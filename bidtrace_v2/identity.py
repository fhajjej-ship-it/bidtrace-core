# SPDX-License-Identifier: MPL-2.0
"""Compose shared provenance with the selected installation's inference settings.

This adapter must not import the HTTP application. Physical file locations may
change independently of their logical identities; all declared files are required.
"""
import json
import os
from pathlib import Path

from bidtrace_core.contracts import PIPELINE_VERSION
from bidtrace_core.identity import implementation_hashes, source_fingerprint as bind_sources
from .providers import provider_identity
from bidtrace_core.package_manifest import SOURCE_FILES


# Keep existing logical keys so saved identities remain intelligible. Extraction
# adds new keys and changes hashes honestly; historical run bindings stay intact.
IMPLEMENTATION_FILES = (
    "models.py", "providers.py", "gemini.py", "config.py", "storage.py", "embedded_storage.py", "obligations.py", "retrieval.py",
    "workflow.py", "../scripts/desktop_runtime.py", "parallel.py", "clauses.py", "segments.py", "buyer_references.py", "buyer_scope.py",
    "mlx_client.py", "mlx_server.py", "mlx_compact.py", "facts.py", "condition_assessment.py", "condition_evidence.py", "local_nli_gate.py",
    "local_mlx_batch_decode.py", "local_generation_queue.py",
    "../scripts/local_mlx_trial.py", "../scripts/local_mlx_context.py",
    "../scripts/local_mlx_grounding.py", "../scripts/local_mlx_content.py",
    "../scripts/local_mlx_relations.py", "../scripts/local_mlx_spans.py",
    "../scripts/local_mlx_batching.py", "../scripts/local_mlx_cache.py",
    "../scripts/local_mlx_budget.py", "../scripts/local_mlx_thinking.py",
    "../scripts/local_mlx_focused_input.py", "../bidtrace/engine.py",
    "documents.py", "identity.py",
)
CORE_IMPLEMENTATION_FILES = ("__init__.py", "contracts.py", "sources.py", "identity.py")


def implementation_files(root=None, *, standalone=False):
    root = root or Path(__file__).resolve().parent.parent
    if standalone:
        return {name: root / name for name in SOURCE_FILES}
    if os.getenv('BIDTRACE_CORE_PATH'):
        from bidtrace_app.manifest import APPLICATION_FILES
        core = Path(os.environ['BIDTRACE_CORE_PATH']).resolve()
        app = Path(os.environ['BIDTRACE_APP_CODE_DIR']).resolve()
        return {**{name: core / name for name in SOURCE_FILES},
                **{'app/' + name: app / name for name in APPLICATION_FILES}}
    files = {name: root / "bidtrace_v2" / name for name in IMPLEMENTATION_FILES}
    files.update({"core/" + name: root / "bidtrace_core" / name for name in CORE_IMPLEMENTATION_FILES})
    files.update({name: root / name for name in SOURCE_FILES})
    return files


def execution_signature(config=None):
    if config is None:
        from .config import settings
        config = settings
    identity = provider_identity(config)
    manifest_path = config.data_dir / "model-cache" / "manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if identity["provider"] == "local" and manifest_path.exists() else {})
    return {"pipeline": PIPELINE_VERSION, **identity,
            "revisions": {key: value.get("revision") for key, value in manifest.items()},
            "implementation": implementation_hashes(implementation_files(standalone=getattr(config, 'standalone_core', False)))}


def source_fingerprint(detail, profile, config=None):
    return bind_sources(detail, profile, execution_signature(config))
