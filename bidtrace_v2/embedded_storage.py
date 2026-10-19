# SPDX-License-Identifier: MPL-2.0
"""Embedded desktop records and immutable objects; no network storage services."""
from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from bidtrace.store import event, now as stamp
from bidtrace_core.contracts import ImmutableSnapshotError, LeaseLostError, JobCancelled
from bidtrace_core.storage_utils import _source_payload, _serialize, validate_object_key

def identity(value): return value

def encode(value): return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

class Cursor:
    def __init__(self, cursor): self.cursor = cursor
    def _row(self, row):
        if row is None: return None
        result = dict(row)
        for key in ('payload', 'result'):
            if result.get(key) is not None: result[key] = json.loads(result[key])
        return result
    def fetchone(self): return self._row(self.cursor.fetchone())
    def fetchall(self): return [self._row(row) for row in self.cursor.fetchall()]

class Connection:
    def __init__(self, connection): self.connection = connection
    def execute(self, sql, parameters=()):
        return Cursor(self.connection.execute(sql, tuple(encode(v) if isinstance(v, (dict,list)) else v for v in parameters)))

class SQLiteStore:
    def __init__(self, path, lease_seconds=90):
        self.path = Path(path)
        self.lease_seconds = max(1, int(lease_seconds))
    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=60)
        con.row_factory = sqlite3.Row
        con.create_function('stamp', 0, stamp)
        con.create_function('epoch', 0, time.time)
        con.execute('PRAGMA foreign_keys=ON')
        con.execute('PRAGMA busy_timeout=60000')
        try:
            con.execute('BEGIN IMMEDIATE')
            yield Connection(con)
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally: con.close()
    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as con:
            con.execute('PRAGMA journal_mode=WAL')
            con.execute("CREATE TABLE IF NOT EXISTS bt_projects (workspace_id TEXT NOT NULL,id TEXT NOT NULL,payload TEXT NOT NULL,created_at TEXT DEFAULT CURRENT_TIMESTAMP,updated_at TEXT DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(workspace_id,id))")
            for name in ('documents','events','reviews'):
                con.execute(f"CREATE TABLE IF NOT EXISTS bt_{name} (workspace_id TEXT NOT NULL,id TEXT NOT NULL,tender_id TEXT NOT NULL,payload TEXT NOT NULL,created_at TEXT DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(workspace_id,id),FOREIGN KEY(workspace_id,tender_id) REFERENCES bt_projects(workspace_id,id))")
            for name in ('runs','requirements'):
                con.execute(f"CREATE TABLE IF NOT EXISTS bt_{name} (workspace_id TEXT NOT NULL,tender_id TEXT NOT NULL,id TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(workspace_id,tender_id,id),FOREIGN KEY(workspace_id,tender_id) REFERENCES bt_projects(workspace_id,id))")
            con.execute("""CREATE TABLE IF NOT EXISTS bt_jobs (id TEXT PRIMARY KEY,workspace_id TEXT NOT NULL,tender_id TEXT NOT NULL,run_id TEXT NOT NULL,payload TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'queued',worker_id TEXT,lease_until REAL,attempts INTEGER NOT NULL DEFAULT 0,result TEXT,error TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP,started_at TEXT,completed_at TEXT,UNIQUE(workspace_id,run_id),FOREIGN KEY(workspace_id,tender_id) REFERENCES bt_projects(workspace_id,id))""")
    def health(self):
        try:
            with self.connect() as con: con.execute('SELECT id FROM bt_projects LIMIT 1').fetchone()
            return True
        except sqlite3.Error: return False
    @staticmethod
    def _lock(con, workspace_id, id): pass # The transaction already holds SQLite's writer lock.
    @staticmethod
    def _current(con, row):
        p=row['payload']['project']
        run=con.execute('SELECT payload FROM bt_runs WHERE workspace_id=? AND tender_id=? AND id=?',
                        (row.get('workspace_id','local'),p['id'],p.get('last_run_id'))).fetchone()
        return run['payload'] if run else None
    def _get(self, con, id, workspace_id):
        row=con.execute('SELECT payload FROM bt_projects WHERE workspace_id=? AND id=?',(workspace_id,id)).fetchone()
        if row is None: raise KeyError(id)
        detail=row['payload']
        mode=detail.pop('_run_history_storage',None)
        if mode is not None:
            if mode!='normalized_v1': raise ImmutableSnapshotError('Unsupported run history storage.')
            runs=[]
            for ref in detail.get('runs',[]):
                saved=con.execute('SELECT payload FROM bt_runs WHERE workspace_id=? AND tender_id=? AND id=?',(workspace_id,id,ref['id'])).fetchone()
                if saved is None: raise ImmutableSnapshotError('A complete run snapshot is missing from tender history.')
                runs.append(saved['payload'])
            detail['runs']=runs
        current=detail['project'].get('last_run_id')
        if current and sum(r['id']==current for r in detail.get('runs',[]))!=1:
            raise ImmutableSnapshotError('Current run is missing or duplicated in tender history.')
        return detail
    def get(self, id, workspace_id='local'):
        with self.connect() as con: return self._get(con,id,workspace_id)
    def get_public(self, id, workspace_id='local'):
        with self.connect() as con: return self._public(con,id,workspace_id)
    def _public(self, con, id, workspace_id):
        row=con.execute('SELECT payload FROM bt_projects WHERE workspace_id=? AND id=?',(workspace_id,id)).fetchone()
        if row is None: raise KeyError(id)
        detail=row['payload']; mode=detail.pop('_run_history_storage',None)
        if mode not in (None,'normalized_v1'): raise ImmutableSnapshotError('Unsupported run history storage.')
        source=detail['project'].get('results_run_id'); runs=[]
        for ref in detail.get('runs',[]):
            if mode:
                # JSON extraction drops large caches inside the database, before decoding.
                saved=con.execute("SELECT json_remove(payload,'$.assessment_cache','$.obligation_batch_cache'" + (",'$.requirements'" if ref['id']!=source else '') + ") AS payload FROM bt_runs WHERE workspace_id=? AND tender_id=? AND id=?",(workspace_id,id,ref['id'])).fetchone()
                if saved is None: raise ImmutableSnapshotError('A complete run snapshot is missing from tender history.')
                run=saved['payload']
            else:
                run={k:v for k,v in ref.items() if k not in {'assessment_cache','obligation_batch_cache'} and (k!='requirements' or ref['id']==source)}
            runs.append(run)
        detail['runs']=runs
        return detail
    def all(self, workspace_id='local'):
        with self.connect() as con:
            rows=con.execute('SELECT id FROM bt_projects WHERE workspace_id=? ORDER BY created_at,id',(workspace_id,)).fetchall()
            return [{'project': con.execute("SELECT json_extract(payload,'$.project') AS payload FROM bt_projects WHERE workspace_id=? AND id=?",(workspace_id,row['id'])).fetchone()['payload']} for row in rows]
    def save(self, detail, review=None, workspace_id='local'):
        with self.connect() as con:
            self._lock(con, workspace_id, detail['project']['id'])
            row = con.execute('SELECT workspace_id,payload FROM bt_projects WHERE workspace_id=? AND id=?',
                              (workspace_id, detail['project']['id'])).fetchone()
            if row is not None and self._current(con, row) is not None:
                # Whole-detail saves may carry a snapshot read before a checkpoint.
                # Existing run transitions belong in the locked mutation methods.
                current = self._current(con, row)
                if detail['project'].get('last_run_id') != current['id']:
                    raise ImmutableSnapshotError('Whole save cannot replace the current analysis run.')
                matches = [i for i, run in enumerate(detail.get('runs', [])) if run['id'] == current['id']]
                if len(matches) != 1:
                    raise ImmutableSnapshotError('Whole save must preserve the current analysis run.')
                detail['runs'][matches[0]] = current
                for key in ('analysis_status', 'last_job_id'):
                    if key in row['payload']['project']:
                        detail['project'][key] = row['payload']['project'][key]
            self._save(con, detail, review, workspace_id)
        return detail

    def _save(self, con, detail, review, workspace_id):
        id = detail['project']['id']
        # Full snapshots already have their own canonical bt_runs rows. Repeating
        # them inside one project JSONB hits PostgreSQL's 256 MiB container limit
        # and doubles every publication write. Keep ordered references here and
        # hydrate complete bodies in one consistent joined read. The transaction
        # retains the existing source, audit and ownership rollback boundaries.
        projection = {**detail, 'runs': [{'id': run['id']} for run in detail.get('runs', [])],
                      '_run_history_storage': 'normalized_v1'}
        con.execute('''INSERT INTO bt_projects(workspace_id,id,payload) VALUES(?,?,?)
                       ON CONFLICT(workspace_id,id) DO UPDATE SET payload=EXCLUDED.payload,updated_at=stamp()''',
                    (workspace_id, id, identity(projection)))
        for document in detail.get('documents', []):
            if document.get('tender_id', id) != id:
                raise ValueError('Document belongs to a different tender.')
            con.execute('''INSERT INTO bt_documents(workspace_id,id,tender_id,payload) VALUES(?,?,?,?)
                           ON CONFLICT(workspace_id,id) DO NOTHING''',
                        (workspace_id, document['id'], id, identity(document)))
            # Inspect after conflict arbitration so a simultaneous insert cannot hide a different source.
            old = con.execute('SELECT tender_id,payload FROM bt_documents WHERE workspace_id=? AND id=?',
                              (workspace_id, document['id'])).fetchone()
            if old['tender_id'] != id or _source_payload(old['payload']) != _source_payload(document):
                raise ImmutableSnapshotError('Source snapshot changed: ' + document['id'])
        # Requirements are the current projection; complete prior results remain in runs/detail history.
        con.execute('DELETE FROM bt_requirements WHERE workspace_id=? AND tender_id=?', (workspace_id, id))
        for requirement in detail.get('requirements', []):
            con.execute('INSERT INTO bt_requirements(workspace_id,tender_id,id,payload) VALUES(?,?,?,?)',
                        (workspace_id, id, requirement['id'], identity(requirement)))
        for run in detail.get('runs', []):
            con.execute('''INSERT INTO bt_runs(workspace_id,tender_id,id,payload) VALUES(?,?,?,?)
                           ON CONFLICT(workspace_id,tender_id,id) DO UPDATE SET payload=EXCLUDED.payload''',
                        (workspace_id, id, run['id'], identity(run)))
        reviews = {r['id']: r for r in detail.get('reviews', [])}
        if review:
            reviews[review['id']] = review
        for table, records in [('bt_events', detail.get('events', [])), ('bt_reviews', list(reviews.values()))]:
            for record in records:
                con.execute(f'''INSERT INTO {table}(workspace_id,id,tender_id,payload) VALUES(?,?,?,?)
                                ON CONFLICT(workspace_id,id) DO NOTHING''',
                            (workspace_id, record['id'], id, identity(record)))
                old = con.execute(f'SELECT tender_id,payload FROM {table} WHERE workspace_id=? AND id=?',
                                  (workspace_id, record['id'])).fetchone()
                if old['tender_id'] != id or old['payload'] != record:
                    raise ImmutableSnapshotError('Audit record changed: ' + record['id'])

    def historical_document(self, id, workspace_id='local'):
        with self.connect() as con:
            row = con.execute('SELECT payload FROM bt_documents WHERE workspace_id=? AND id=?',
                              (workspace_id, id)).fetchone()
        return row['payload'] if row else None

    def mutate(self, id, callback, workspace_id='local'):
        with self.connect() as con:
            self._lock(con, workspace_id, id)
            detail = callback(self._get(con, id, workspace_id))
            if not isinstance(detail, dict) or detail['project']['id'] != id:
                raise ValueError('Mutation must return the same tender detail.')
            self._save(con, detail, None, workspace_id)
        return detail

    def enqueue(self, tender_id, run_id, payload):
        workspace_id = payload.get('workspace_id', 'local')
        with self.connect() as con:
            con.execute('''INSERT INTO bt_jobs(id,workspace_id,tender_id,run_id,payload) VALUES(?,?,?,?,?)
                           ON CONFLICT(workspace_id,run_id) DO NOTHING''',
                        ('job_' + uuid.uuid4().hex, workspace_id, tender_id, run_id, identity(payload)))
            row = con.execute('SELECT * FROM bt_jobs WHERE workspace_id=? AND run_id=?',
                              (workspace_id, run_id)).fetchone()
            if row['tender_id'] != tender_id or row['payload'] != payload:
                raise ValueError('Run was already queued with different inputs.')
        return _serialize(row)



    def _locked_job(self, con, job_id, workspace_id):
        # Project -> job is the shared lock order for cancel, resume and worker writes.
        row = con.execute('SELECT tender_id FROM bt_jobs WHERE id=? AND workspace_id=?',
                          (job_id, workspace_id)).fetchone()
        if row is None:
            raise KeyError(job_id)
        self._lock(con, workspace_id, row['tender_id'])
        return con.execute('SELECT *, lease_until >= epoch() AS lease_valid FROM bt_jobs '
                           'WHERE id=? AND workspace_id=?', (job_id, workspace_id)).fetchone()

    @staticmethod
    def _check_owner(job, worker_id, allow_stopping=False):
        if job['worker_id'] != worker_id or not job['lease_valid'] or job['status'] not in {'running', 'stopping'}:
            raise LeaseLostError('Job lease expired or belongs to another worker.')
        if job['status'] == 'stopping' and not allow_stopping:
            raise JobCancelled('Analysis stopped by user request.')

    def mutate_job(self, job_id, worker_id, callback, workspace_id='local', *, allow_stopping=False,
                   result=None, error=None, expected_run_id=None, expected_tender_id=None):
        """Fence every worker projection write, including publication, against Stop."""
        with self.connect() as con:
            job = self._locked_job(con, job_id, workspace_id)
            self._check_owner(job, worker_id, allow_stopping)
            if ((expected_run_id is not None and job['run_id'] != expected_run_id)
                    or (expected_tender_id is not None and job['tender_id'] != expected_tender_id)):
                raise LeaseLostError('Job does not match this run and tender.')
            detail = self._get(con, job['tender_id'], workspace_id)
            if detail['project'].get('last_run_id') != job['run_id']:
                raise LeaseLostError('A different run now owns this project.')
            updated = callback(detail)
            if not isinstance(updated, dict) or updated['project']['id'] != job['tender_id']:
                raise ValueError('Mutation must return the same tender detail.')
            self._save(con, updated, None, workspace_id)
            if result is not None or error is not None:
                # The result and terminal job state commit together; cancellation cannot split them.
                con.execute('UPDATE bt_jobs SET status=?,result=?,error=?,completed_at=stamp(),lease_until=NULL '
                            'WHERE id=?', ('failed' if error else 'completed', identity(result), error, job_id))
            return updated


    def _cancel_projection(self, con, job, status):
        detail = self._get(con, job['tender_id'], job['workspace_id'])
        run = next(r for r in detail['runs'] if r['id'] == job['run_id'])
        run.update(status=status, error='Stopped by user request.', completed_at=stamp() if status == 'cancelled' else None)
        if status == 'cancelled':
            for step in run.get('steps', []):
                if step.get('status') == 'running':
                    step.update(status='cancelled', completed_at=stamp())
        if detail['project'].get('last_run_id') == job['run_id']:
            detail['project'].update(analysis_status=status, updated_at=stamp())
        event(detail, 'analysis_' + status, 'Stop requested; waiting for the current operation.' if status == 'stopping'
              else 'Analysis stopped. Completed work and original sources are saved for resume.')
        self._save(con, detail, None, job['workspace_id'])

    def cancel(self, job_id, workspace_id='local'):
        with self.connect() as con:
            job = self._locked_job(con, job_id, workspace_id)
            if job['status'] not in {'queued', 'running'}:
                return _serialize({k: v for k, v in job.items() if k != 'lease_valid'})
            status = 'cancelled' if job['status'] == 'queued' else 'stopping'
            row = con.execute("""UPDATE bt_jobs SET status=?,error='Stopped by user request.',
                completed_at=CASE WHEN ?='cancelled' THEN stamp() ELSE NULL END,
                lease_until=CASE WHEN ?='cancelled' THEN NULL ELSE lease_until END
                WHERE id=? RETURNING *""", (status, status, status, job_id)).fetchone()
            self._cancel_projection(con, job, status)
            return _serialize(row)

    def complete_cancel(self, job_id, worker_id, workspace_id='local'):
        with self.connect() as con:
            job = self._locked_job(con, job_id, workspace_id)
            self._check_owner(job, worker_id, allow_stopping=True)
            if job['status'] != 'stopping':
                raise ValueError('Only a stopping job can acknowledge cancellation.')
            row = con.execute("UPDATE bt_jobs SET status='cancelled',completed_at=stamp(),lease_until=NULL "
                              'WHERE id=? RETURNING *', (job_id,)).fetchone()
            self._cancel_projection(con, job, 'cancelled')
            return _serialize(row)



    def finish(self, job_id, worker_id, result=None, error=None):
        with self.connect() as con:
            existing = con.execute('SELECT * FROM bt_jobs WHERE id=?', (job_id,)).fetchone()
            if (existing and existing['status'] == 'completed' and existing['worker_id'] == worker_id
                    and existing['result'] == result and error is None):
                return _serialize(existing)
            row = con.execute('''UPDATE bt_jobs SET status=?,result=?,error=?,completed_at=stamp(),lease_until=NULL
                WHERE id=? AND worker_id=? AND status='running' AND lease_until>=epoch() RETURNING *''',
                              ('failed' if error else 'completed', identity(result), str(error) if error else None,
                               job_id, worker_id)).fetchone()
            if row is None:
                raise LeaseLostError('Job lease expired or belongs to another worker.')
        return _serialize(row)

    def retry(self, job_id, workspace_id='local'):
        with self.connect() as con:
            row = con.execute("""UPDATE bt_jobs SET status='queued',worker_id=NULL,lease_until=NULL,
                error=NULL,result=NULL,completed_at=NULL WHERE id=? AND workspace_id=? AND status IN ('failed','cancelled')
                RETURNING *""", (job_id, workspace_id)).fetchone()
            if row is None:
                raise ValueError('Only a failed or cancelled job in this workspace can be retried.')
        return _serialize(row)

    def get_job(self, id):
        with self.connect() as con:
            row = con.execute('SELECT * FROM bt_jobs WHERE id=?', (id,)).fetchone()
        return _serialize(row)

    def queue_analysis(self, tender_id, payload, prepare, workspace_id='local', *, compact=False):
        with self.connect() as con:
            detail = self._public(con,tender_id,workspace_id) if compact else self._get(con,tender_id,workspace_id)
            if compact:
                matching=next((r for r in detail['runs'] if r.get('fingerprint')==payload.get('fingerprint')),None)
                if matching:
                    saved=con.execute('SELECT payload FROM bt_runs WHERE workspace_id=? AND tender_id=? AND id=?',(workspace_id,tender_id,matching['id'])).fetchone()
                    if saved is None: raise ImmutableSnapshotError('Selected run snapshot is missing.')
                    detail['runs'][detail['runs'].index(matching)]=saved['payload']
            run=prepare(detail)
            job=con.execute('SELECT * FROM bt_jobs WHERE workspace_id=? AND run_id=?',(workspace_id,run['id'])).fetchone()
            if job and job['status'] not in {'failed','cancelled'}: return job
            if job:
                if job['tender_id']!=tender_id or job['payload']!=payload: raise ValueError('Run was already queued with different inputs.')
                job=con.execute("UPDATE bt_jobs SET status='queued',worker_id=NULL,lease_until=NULL,error=NULL,result=NULL,completed_at=NULL WHERE id=? RETURNING *",(job['id'],)).fetchone()
            else:
                job=con.execute('INSERT INTO bt_jobs(id,workspace_id,tender_id,run_id,payload) VALUES(?,?,?,?,?) RETURNING *',('job_'+uuid.uuid4().hex,workspace_id,tender_id,run['id'],payload)).fetchone()
            if not any(r['id']==run['id'] for r in detail['runs']): detail['runs'].insert(0,run)
            run.update(status='queued',job_id=job['id'],error=None,completed_at=None)
            detail['project'].update(analysis_status='queued',last_run_id=run['id'],last_job_id=job['id'],updated_at=stamp())
            if compact:
                con.execute('INSERT INTO bt_runs(workspace_id,tender_id,id,payload) VALUES(?,?,?,?) ON CONFLICT(workspace_id,tender_id,id) DO UPDATE SET payload=excluded.payload',(workspace_id,tender_id,run['id'],run))
                projection=con.execute('SELECT payload FROM bt_projects WHERE workspace_id=? AND id=?',(workspace_id,tender_id)).fetchone()['payload']
                projection.update(project=detail['project'],runs=[{'id':r['id']} for r in detail['runs']],_run_history_storage='normalized_v1')
                con.execute('UPDATE bt_projects SET payload=?,updated_at=stamp() WHERE workspace_id=? AND id=?',(projection,workspace_id,tender_id))
            else: self._save(con,detail,None,workspace_id)
            return job

    def mutate_run(self, job_id, worker_id, callback, workspace_id='local', *, allow_stopping=False, expected_run_id=None, expected_tender_id=None):
        mutable={'steps','metrics','model_stats','progress','work_progress','assessment_cache','obligation_batch_cache'}
        with self.connect() as con:
            job=self._locked_job(con,job_id,workspace_id); self._check_owner(job,worker_id,allow_stopping)
            if ((expected_run_id is not None and job['run_id']!=expected_run_id) or (expected_tender_id is not None and job['tender_id']!=expected_tender_id)): raise LeaseLostError('Job does not match this run and tender.')
            projection=con.execute('SELECT payload FROM bt_projects WHERE workspace_id=? AND id=?',(workspace_id,job['tender_id'])).fetchone()['payload']
            saved=con.execute('SELECT payload FROM bt_runs WHERE workspace_id=? AND tender_id=? AND id=?',(workspace_id,job['tender_id'],job['run_id'])).fetchone()
            if not saved or projection['project'].get('last_run_id')!=job['run_id'] or sum(r['id']==job['run_id'] for r in projection['runs'])!=1: raise LeaseLostError('A different run now owns this project or its run snapshot is missing.')
            frozen=deepcopy({k:v for k,v in saved['payload'].items() if k not in mutable})
            updated=callback(saved['payload'])
            if not isinstance(updated,dict) or {k:v for k,v in updated.items() if k not in mutable}!=frozen: raise ValueError('A run checkpoint cannot change identity, configuration or published results.')
            con.execute('UPDATE bt_runs SET payload=? WHERE workspace_id=? AND tender_id=? AND id=?',(updated,workspace_id,job['tender_id'],job['run_id']))
            return updated

    def claim(self, worker_id, workspace_id='local'):
        if not worker_id: raise ValueError('Worker id is required.')
        with self.connect() as con:
            candidate=con.execute("SELECT id FROM bt_jobs WHERE workspace_id=? AND (status='queued' OR (status IN ('running','stopping') AND lease_until<epoch())) ORDER BY created_at,id LIMIT 1",(workspace_id,)).fetchone()
            if not candidate: return None
            return con.execute("UPDATE bt_jobs SET status=CASE WHEN status='stopping' THEN 'stopping' ELSE 'running' END,worker_id=?,lease_until=epoch()+?,attempts=attempts+1,started_at=COALESCE(started_at,stamp()) WHERE id=? RETURNING *",(worker_id,self.lease_seconds,candidate['id'])).fetchone()

    def heartbeat(self, job_id, worker_id, *, allow_stopping=False):
        with self.connect() as con:
            row=con.execute("UPDATE bt_jobs SET lease_until=epoch()+? WHERE id=? AND worker_id=? AND status IN ('running','stopping') AND lease_until>=epoch() RETURNING *",(self.lease_seconds,job_id,worker_id)).fetchone()
            if row is None: raise LeaseLostError('Job lease expired or belongs to another worker.')
        if row['status']=='stopping' and not allow_stopping: raise JobCancelled('Analysis stopped by user request.')
        return row

class FileObjectStore:
    def __init__(self, path): self.path=Path(path)
    def initialize(self): self.path.mkdir(parents=True,exist_ok=True)
    def _path(self,key):
        validate_object_key(key)
        result=self.path.joinpath(key)
        if not result.resolve().is_relative_to(self.path.resolve()): raise ValueError('Object path escapes local storage.')
        return result
    def put(self,key,data,content_type='application/octet-stream'):
        path=self._path(key)
        if hashlib.sha256(data).hexdigest() not in key.split('/'): raise ValueError('Object key must contain its SHA256 digest as a path component.')
        path.parent.mkdir(parents=True,exist_ok=True)
        # Link a fully written temporary object; readers never see a partial source.
        temporary=path.parent/('.'+uuid.uuid4().hex+'.tmp')
        try:
            with temporary.open('xb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            try: os.link(temporary,path)
            except FileExistsError:
                if path.read_bytes()!=data: raise ImmutableSnapshotError('Object key already contains different bytes.')
        finally: temporary.unlink(missing_ok=True)
        return key
    def get(self,key): return self._path(key).read_bytes()
    def health(self): return self.path.is_dir() and os.access(self.path,os.R_OK|os.W_OK)
