# SPDX-License-Identifier: MPL-2.0
"""Synthetic transport fixtures verify setup, hash rejection and no overwrite."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

spec=importlib.util.spec_from_file_location('prepare_models',Path(__file__).resolve().parents[1]/'tools/prepare_models.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


def configured(tmp_path,monkeypatch,wrong_hash=False):
    artifact=b'Synthetic model fixture, not weights'
    sha=hashlib.sha256(artifact).hexdigest()
    models=[{'kind':kind,'repository':'synthetic/'+kind,'revision':'0'*40,'folder':folder,
             'files':[{'file':'config.json','bytes':len(artifact),'sha256':'f'*64 if wrong_hash else sha}]}
            for kind,folder in [('assessment','qwen'),('nli','nli'),('embedding','embedding-cache/models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/'+'0'*40)]]
    (tmp_path/'models.json').write_text(json.dumps({'models':models}))
    monkeypatch.setattr(prepare,'__file__',str(tmp_path/'prepare_models.py'))
    allowed={f"https://huggingface.co/{m['repository']}/resolve/{m['revision']}/config.json" for m in models}
    def download(url,timeout):
        assert url in allowed and timeout==60
        return io.BytesIO(artifact)
    monkeypatch.setattr(prepare.urllib.request,'urlopen',download)
    root=tmp_path/'new-models';monkeypatch.setattr(sys,'argv',['prepare_models.py',str(root)])
    return root,{'models':models}


def test_preparation_produces_verified_bge_and_nli_layout(tmp_path,monkeypatch):
    root,models=configured(tmp_path,monkeypatch)
    prepare.main();prepare.verify(root,models)
    assert (root/"embedding-cache/models--qdrant--bge-small-en-v1.5-onnx-q/refs/main").read_bytes()==b"0"*40
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(sys,'argv',['prepare_models.py',str(root),'--verify-only'])
    prepare.main()
    assert before=={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_existing_destination_is_not_overwritten(tmp_path,monkeypatch):
    root,_=configured(tmp_path,monkeypatch);root.mkdir();sentinel=root/'existing.txt';sentinel.write_text('Preserve this')
    with pytest.raises(FileExistsError):prepare.main()
    assert list(root.iterdir())==[sentinel] and sentinel.read_text()=='Preserve this'


def test_download_hash_mismatch_is_rejected(tmp_path,monkeypatch):
    configured(tmp_path,monkeypatch,wrong_hash=True)
    with pytest.raises(ValueError,match='hash mismatch'):prepare.main()


def test_cache_revision_has_no_trailing_newline(tmp_path,monkeypatch):
    root,models=configured(tmp_path,monkeypatch)
    prepare.main()
    ref=root/'embedding-cache/models--qdrant--bge-small-en-v1.5-onnx-q/refs/main'
    ref.write_text(ref.read_text()+'\n')
    with pytest.raises(ValueError,match='BGE revision'):prepare.verify(root,models)


def test_mlx_provenance_is_required(tmp_path,monkeypatch):
    root,models=configured(tmp_path,monkeypatch)
    prepare.main()
    (root/'qwen/.cache/huggingface/download/config.json.metadata').write_text('wrong-revision\n')
    with pytest.raises(ValueError,match='MLX model provenance'):prepare.verify(root,models)
