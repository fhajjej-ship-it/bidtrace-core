# SPDX-License-Identifier: MPL-2.0
"""Download the pinned public models once, or verify an existing prepared directory.

No documents, account credentials or inference requests are transmitted. Models
retain their upstream licenses. The analysis CLI itself never downloads weights.
"""
from pathlib import Path
import argparse,hashlib,json,time,urllib.request


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(root, spec):
    for model in spec['models']:
        for item in model['files']:
            p=root/model['folder']/item['file']
            if p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
                raise ValueError('Model bytes differ: '+model['kind']+'/'+item['file'])
    assessment=next(m for m in spec['models'] if m['kind']=='assessment')
    metadata=root/assessment['folder']/'.cache/huggingface/download'
    for item in assessment['files']:
        if (metadata/(item['file']+'.metadata')).read_text().splitlines()[0]!=assessment['revision']:
            raise ValueError('MLX model provenance differs.')
    bge=next(m for m in spec['models'] if m['kind']=='embedding')
    if (root/'embedding-cache/models--qdrant--bge-small-en-v1.5-onnx-q/refs/main').read_text()!=bge['revision']:
        raise ValueError('BGE revision reference differs.')
    nli=next(m for m in spec['models'] if m['kind']=='nli')
    manifest=json.loads((root/'nli/download-manifest.json').read_text())
    if (manifest['repository'],manifest['revision'],manifest['files'])!=(nli['repository'],nli['revision'],nli['files']):
        raise ValueError('NLI provenance manifest differs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='New destination outside the checkout, or existing prepared directory for verification.')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();root=args.directory.resolve()
    spec=json.loads(Path(__file__).with_name('models.json').read_text())
    if not args.verify_only:
        root.mkdir(parents=True,exist_ok=False)
        for model in spec['models']:
            target=root/model['folder'];target.mkdir(parents=True,exist_ok=True)
            for item in model['files']:
                name=item['file']
                if Path(name).name!=name:raise ValueError('Unsafe model filename')
                url=f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{name}"
                print('Downloading '+model['kind']+'/'+name,flush=True)
                with urllib.request.urlopen(url,timeout=60) as response,(target/name).open('xb') as output:
                    remaining=item['bytes']
                    while chunk:=response.read(min(1024*1024,remaining+1)):
                        if len(chunk)>remaining:raise ValueError('Artifact exceeds pinned size')
                        output.write(chunk);remaining-=len(chunk)
                if remaining or digest(target/name)!=item['sha256']:raise ValueError('Downloaded model hash mismatch: '+name)
        # Emit provenance only after every pinned artifact passed its byte/hash check.
        assessment=next(m for m in spec['models'] if m['kind']=='assessment')
        metadata=root/assessment['folder']/'.cache/huggingface/download';metadata.mkdir(parents=True)
        for item in assessment['files']:
            (metadata/(item['file']+'.metadata')).write_text(f"{assessment['revision']}\n{item['sha256']}\n{time.time()}\n")
        bge=next(m for m in spec['models'] if m['kind']=='embedding')
        ref=root/'embedding-cache/models--qdrant--bge-small-en-v1.5-onnx-q/refs/main';ref.parent.mkdir(parents=True);ref.write_text(bge['revision'])
        nli=next(m for m in spec['models'] if m['kind']=='nli')
        (root/'nli/download-manifest.json').write_text(json.dumps({k:nli[k] for k in ('repository','revision','files')},indent=2)+'\n')
    verify(root,spec)
    print(json.dumps({'verified':True,'directory':str(root),'models':len(spec['models']),'hosted_inference':False}))


if __name__=='__main__':main()
