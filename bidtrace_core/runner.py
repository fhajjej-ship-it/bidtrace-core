# SPDX-License-Identifier: MPL-2.0
"""Synchronous local pack analysis using the same durable workflow as the app."""
from pathlib import Path
import mimetypes

from bidtrace.store import uid, digest, now, event
from .workspace import new_detail
from .sources import selected_sources
from .jobs import run_once
from bidtrace_v2.documents import extract_document
from bidtrace_v2.embedded_storage import SQLiteStore, FileObjectStore
from bidtrace_v2.identity import source_fingerprint, execution_signature
from bidtrace_v2.providers import provider_identity


def analyze_pack(config, buyers, suppliers, *, title='Local tender review'):
    """Create a new workspace; never overwrite or reuse an existing customer store."""
    if config.data_dir.exists():
        raise ValueError('Analysis requires a new workspace path. Existing data was not changed.')
    inputs=[]
    for role, paths in [('current', buyers), ('evidence', suppliers)]:
        names=set()
        for path in paths:
            p=Path(path).resolve()
            if not p.is_file():raise ValueError(f'Input file does not exist: {p.name}')
            if p.name in names:raise ValueError('Each source collection needs distinct filenames.')
            names.add(p.name);inputs.append((role,p))
    if not buyers or not suppliers:raise ValueError('Supply buyer documents and supplier evidence.')
    config.data_dir.mkdir(parents=True,exist_ok=False)
    store=SQLiteStore(config.data_dir/'records.sqlite');objects=FileObjectStore(config.data_dir/'objects')
    store.initialize();objects.initialize()
    detail=new_detail({'title':title,'provenance':'user'})
    tid=detail['project']['id']
    for role,path in inputs:
        extracted=extract_document(path.name,path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream',objects,f'{config.workspace_id}/{tid}')
        detail['documents'].append({'id':uid('doc'),'tender_id':tid,'logical_id':uid('file'),
            'title':path.name,'role':role,'version':1,'active':True,'provenance':'user',
            'source_url':None,'created_at':now(),'sha256':digest(extracted['text']),**extracted})
    detail['project'].update(active_version=1,coverage='Local uploaded pack; external attachments are not included.')
    event(detail,'documents','Saved buyer documents and supplier evidence for local analysis.')
    store.save(detail,workspace_id=config.workspace_id)
    fingerprint=source_fingerprint(detail,'text',config)
    def prepare(d):
        return {'id':uid('run'),'fingerprint':fingerprint,'profile':'text','mode':'text',
            'pipeline_version':execution_signature(config)['pipeline'],**provider_identity(config),
            'execution_config':execution_signature(config),'steps':[],'metrics':{},'model_stats':{},
            'source_document_ids':[doc['id'] for doc in selected_sources(d)],
            'started_at':None,'completed_at':None,'error':None}
    job=store.queue_analysis(tid,{'workspace_id':config.workspace_id,'profile':'text','fingerprint':fingerprint},prepare,config.workspace_id,compact=True)
    try:
        if not run_once(config,store,objects,uid('core_worker')):
            raise RuntimeError('The local analysis job could not be claimed.')
    except KeyboardInterrupt:
        store.cancel(job['id'],config.workspace_id)
        raise
    final=store.get(tid,config.workspace_id)
    run=next(r for r in final['runs'] if r['id']==job['run_id'])
    if run['status']!='completed':raise RuntimeError(run.get('error') or 'Analysis did not complete; workspace and progress retained.')
    return final
