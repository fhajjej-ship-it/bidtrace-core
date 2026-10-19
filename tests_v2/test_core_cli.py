# SPDX-License-Identifier: MPL-2.0
"""Standalone configuration and lossless export boundaries; no model doubles claim accuracy."""
from copy import deepcopy
import csv
import json

import pytest

from bidtrace_core.config import CoreSettings
from bidtrace_core.exports import read_project, write_export
from bidtrace_core.runner import analyze_pack
from bidtrace_core.workspace import new_detail
from bidtrace_v2.embedded_storage import SQLiteStore


@pytest.mark.parametrize('url', ['https://api.example.com','http://127.0.0.1:8000/path',
                               'http://user:password@localhost:8000','http://localhost:8000?key=x'])
def test_configuration_refuses_nonlocal_or_credential_bearing_services(tmp_path,url):
    with pytest.raises(ValueError,match='loopback'):
        CoreSettings(tmp_path,url)


def test_analyze_refuses_existing_workspace_before_changing_any_data(tmp_path):
    sentinel=tmp_path/'existing.txt';sentinel.write_text('Existing customer data')
    config=CoreSettings(tmp_path,'http://127.0.0.1:9')
    with pytest.raises(ValueError,match='new workspace'):
        analyze_pack(config,[],[])
    assert sentinel.read_text()=='Existing customer data'
    assert list(tmp_path.iterdir())==[sentinel]


def test_json_csv_preserve_unknown_evidence_history_and_read_only_source(tmp_path):
    store=SQLiteStore(tmp_path/'records.sqlite');store.initialize()
    d=new_detail({'title':'Synthetic export evidence fixture','provenance':'synthetic'})
    d['documents']=[{'id':'buyer','logical_id':'b','title':'buyer.txt','role':'current','version':1,'active':True,'raw_sha256':'buyerhash','sha256':'buyertext','pages':[{'page':1,'text':'Require a 24-hour cloud RPO.'}]}]
    req={'id':'r1','key':'r1','after':'Require a 24-hour cloud RPO.','assessment':'unknown','stale':True,
         'citations':[{'document_id':'buyer','page':1,'quote':'Require a 24-hour cloud RPO.','role':'current'}],
         'proposal':{'assessment':'unknown','rationale':'Monitoring alone does not establish the cloud RPO.','citations':[]}}
    d['requirements']=[req]
    d['reviews']=[{'id':'human1','requirement_id':'r1','decision':'needs_followup','note':'Obtain the promised RPO.','fingerprint':'oldsource'}]
    d['runs']=[{'id':'run1','status':'completed','requirements':[deepcopy(req)],'completed_at':'2026-10-17T10:00:00Z','models':{'generation':'synthetic fixture'}}]
    d['project'].update(last_run_id='run1',results_run_id='run1',analysis_status='completed')
    store.save(d);before=store.get(d['project']['id'])
    detail=read_project(tmp_path)
    assert detail['requirements']==before['requirements']
    out=tmp_path/'export.json';write_export(detail,out,'json')
    exported=json.loads(out.read_text())['detail']
    assert exported['reviews']==before['reviews'] and exported['runs']==before['runs']
    assert exported['requirements'][0]['proposal']['assessment']=='unknown'
    assert exported['results_provenance']['run_id']=='run1'
    csv_path=tmp_path/'export.csv';write_export(detail,csv_path,'csv')
    with csv_path.open() as stream:row=next(csv.DictReader(stream))
    assert json.loads(row['citations_json'])==req['citations']
    assert json.loads(row['reviews_json'])==d['reviews']
    assert row['assessment']=='unknown' and row['stale']=='True'
    assert store.get(d['project']['id'])==before
    saved=out.read_bytes()
    with pytest.raises(FileExistsError):write_export(detail,out,'json')
    assert out.read_bytes()==saved


def test_wheel_manifest_excludes_paid_app_and_evaluation_tools():
    from bidtrace_core.package_manifest import SOURCE_FILES
    assert not set(SOURCE_FILES)&{'bidtrace_v2/app.py','bidtrace_v2/config.py','bidtrace_v2/worker.py',
                                 'bidtrace_v2/storage.py','bidtrace_v2/gemini.py'}
    assert not any(p.startswith(('scripts/','development/','web/','macos/','data')) for p in SOURCE_FILES)
