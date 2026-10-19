# SPDX-License-Identifier: MPL-2.0
"""Development CLI: analyze a new local pack or export existing user-owned results."""
import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(prog='bidtrace-core',description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    analyze=sub.add_parser('analyze',help='Analyze buyer documents against supplied evidence into a NEW workspace.')
    analyze.add_argument('--workspace',type=Path,required=True)
    analyze.add_argument('--buyer',type=Path,action='append',required=True)
    analyze.add_argument('--supplier',type=Path,action='append',required=True)
    analyze.add_argument('--model-url',required=True,help='Existing loopback MLX service URL.')
    analyze.add_argument('--embedding-cache',type=Path,required=True)
    analyze.add_argument('--nli-model',type=Path,required=True)
    analyze.add_argument('--title',default='Local tender review')
    export=sub.add_parser('export',help='Export one project as JSON or CSV without changing its workspace.')
    export.add_argument('--workspace',type=Path,required=True)
    export.add_argument('--project')
    export.add_argument('--format',choices=['json','csv'],default='json')
    export.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='export':
            from .exports import read_project,write_export
            result=write_export(read_project(args.workspace,args.project),args.output,args.format)
            print(json.dumps({'output':str(result),'format':args.format}));return 0
        from .config import CoreSettings
        config=CoreSettings(data_dir=args.workspace.resolve(),model_url=args.model_url.rstrip('/'))
        for path in [args.embedding_cache,args.nli_model]:
            if not path.is_dir():raise ValueError(f'Local model/cache directory does not exist: {path}')
        os.environ.update(BIDTRACE_EMBEDDING_CACHE=str(args.embedding_cache.resolve()),
            BIDTRACE_NLI_MODEL_PATH=str(args.nli_model.resolve()),BIDTRACE_ALLOW_HOSTED_TRACES='0',
            LANGSMITH_TRACING='false',LANGCHAIN_TRACING_V2='false',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        from .runner import analyze_pack
        detail=analyze_pack(config,args.buyer,args.supplier,title=args.title)
        print(json.dumps({'project_id':detail['project']['id'],'run_id':detail['runs'][-1]['id'],
            'status':'completed','workspace':str(config.data_dir),'hosted_inference':False}));return 0
    except KeyboardInterrupt:
        parser.exit(130,'Analysis interrupted; sources and progress remain in the local workspace.\n')
    except (OSError,ValueError,RuntimeError) as exc:
        parser.exit(1,f'bidtrace-core: {exc}\n')


if __name__=='__main__':
    raise SystemExit(main())
