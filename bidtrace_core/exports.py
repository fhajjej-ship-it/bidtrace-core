# SPDX-License-Identifier: MPL-2.0
"""Basic evidence/review exports; read existing workspaces without rewriting them."""
import csv
import json
import sqlite3
from pathlib import Path
from bidtrace_v2.provenance import with_results_provenance


def read_project(workspace, tender_id=None):
    path=Path(workspace).resolve()/'records.sqlite'
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as con:
        rows=con.execute('SELECT id,payload FROM bt_projects WHERE workspace_id=?'+(' AND id=?' if tender_id else ''),
                         ('local',tender_id) if tender_id else ('local',)).fetchall()
        if len(rows)!=1:raise ValueError('Select one project with --project; the workspace must contain a matching project.')
        detail=json.loads(rows[0][1]);mode=detail.pop('_run_history_storage',None)
        if mode not in {None,'normalized_v1'}:raise ValueError('Unsupported workspace history format.')
        if mode:
            snapshots=[]
            for ref in detail['runs']:
                row=con.execute('SELECT payload FROM bt_runs WHERE workspace_id=? AND tender_id=? AND id=?',('local',rows[0][0],ref['id'])).fetchone()
                if not row:raise ValueError('A saved run snapshot is missing.')
                snapshots.append(json.loads(row[0]))
            detail['runs']=snapshots
    return with_results_provenance(detail)


def write_export(detail, output, format):
    if format not in {'json','csv'}:raise ValueError('Choose JSON or CSV export.')
    path=Path(output)
    # Exclusive creation protects an existing report, including when CLI paths collide.
    with path.open('x',encoding='utf-8',newline='') as stream:
        if format=='json':
            json.dump({'format':'bidtrace-core-1','detail':detail},stream,ensure_ascii=False,indent=2)
        else:
            writer=csv.DictWriter(stream,fieldnames=['requirement_id','source_text','assessment','stale','citations_json','proposal_json','reviews_json','source_versions_json'])
            writer.writeheader()
            for req in detail['requirements']:
                text=req.get('after') or req.get('before') or ''
                if text.startswith(('=','+','-','@','\t','\r')):text="'"+text
                writer.writerow({'requirement_id':req['id'],'source_text':text,
                    'assessment':(req.get('proposal') or {}).get('assessment',req.get('assessment','unknown')),
                    'stale':bool(req.get('stale')),
                    'citations_json':json.dumps(req.get('citations',[]),ensure_ascii=False),
                    'proposal_json':json.dumps(req.get('proposal'),ensure_ascii=False),
                    'reviews_json':json.dumps([r for r in detail.get('reviews',[]) if r.get('requirement_id')==req['id']],ensure_ascii=False),
                    'source_versions_json':json.dumps([{k:d.get(k) for k in ['id','logical_id','version','role','raw_sha256','sha256','active']} for d in detail['documents']],ensure_ascii=False)})
    return path
