# SPDX-License-Identifier: MPL-2.0
"""Select source versions without changing the stored document snapshots."""
import copy


def active_documents(detail):
    return [d for d in detail["documents"] if d.get("active", True) and d["role"] != "previous"]


def selected_sources(detail):
    active = active_documents(detail)
    sources = list(active)
    for current in [d for d in active if d["role"] == "current"]:
        earlier = [d for d in detail["documents"] if d["logical_id"] == current["logical_id"]
                   and d["version"] < current["version"] and d["role"] != "evidence"]
        if earlier:
            previous = copy.deepcopy(max(earlier, key=lambda d: d["version"]))
            previous["role"] = "previous"  # Comparison-local; stored source metadata stays immutable.
            sources.append(previous)
    return sources
