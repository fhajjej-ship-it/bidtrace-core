# SPDX-License-Identifier: MPL-2.0
"""Pure source binding and logical-file hashes; runtime configuration is injected."""
from hashlib import sha256
import json


def digest(value):
    # Preserve the existing serialized identity contract, including string handling.
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return sha256(raw.encode()).hexdigest()


def implementation_hashes(files):
    """Keep logical identities stable across moves; missing declared code is an error."""
    return {logical_name: digest(path.read_text()) for logical_name, path in files.items()}


def source_fingerprint(detail, profile, execution):
    # Earlier snapshots still bind comparison results; list order is unchanged.
    return digest({"documents": [(d["id"], d["sha256"], d.get("active", True)) for d in detail["documents"]],
                   "profile": profile, "execution": execution})
