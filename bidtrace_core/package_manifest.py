# SPDX-License-Identifier: MPL-2.0
"""Explicit first-party wheel contents; names retain engine import compatibility."""
ENGINE_MODULES = (
    'documents', 'workflow', 'identity', 'providers', 'embedded_storage', 'provenance',
    'models', 'model_server', 'obligations', 'retrieval', 'parallel', 'clauses', 'segments',
    'buyer_references', 'buyer_scope', 'mlx_client', 'mlx_server', 'mlx_compact', 'facts',
    'condition_assessment', 'condition_evidence', 'local_nli_gate', 'local_mlx_batch_decode',
    'local_generation_queue',
)
CORE_MODULES = ('contracts', 'identity', 'sources', 'storage_utils', 'prompts', 'package_manifest',
                'config', 'runner', 'cli', 'exports', 'jobs', 'workspace', '__main__')
LOCAL_MODULES = ('context', 'grounding', 'content', 'relations', 'spans', 'batching', 'cache',
                 'budget', 'thinking', 'focused_input', 'runtime')
MODULES = (['bidtrace.engine', 'bidtrace.store']
           + ['bidtrace_v2.' + name for name in ENGINE_MODULES]
           + ['bidtrace_core.' + name for name in CORE_MODULES]
           + ['bidtrace_core.local.' + name for name in LOCAL_MODULES])
SOURCE_FILES = tuple(dict.fromkeys(
    ['bidtrace/__init__.py', 'bidtrace_v2/__init__.py', 'bidtrace_core/__init__.py', 'bidtrace_core/local/__init__.py']
    + [name.replace('.', '/') + '.py' for name in MODULES]))
