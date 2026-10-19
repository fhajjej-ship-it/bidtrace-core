# SPDX-License-Identifier: MPL-2.0
"""Bind visible findings to a completed run without changing stored history."""
from bidtrace.store import digest


_REVIEW_FIELDS = {"reviewed", "review_decision", "review_note", "status", "stale"}


def _findings_identity(requirements):
    if not isinstance(requirements, list) or not requirements:
        return None
    if any(not isinstance(req, dict) or not isinstance(req.get("id"), str) for req in requirements):
        return None
    if len({req["id"] for req in requirements}) != len(requirements):
        return None
    return digest(sorted(({key: value for key, value in req.items() if key not in _REVIEW_FIELDS}
                          for req in requirements), key=lambda req: req["id"]))


def results_provenance(detail):
    """Only exact analysis snapshots establish provenance; latest attempt does not.

    Human reviews and stale flags can change after publication. All source,
    assessment, proposal, fingerprint and execution fields must still match.
    Legacy records with no unique matching snapshot stay explicitly unknown.
    """
    unknown = {"run_id": None, "model": None, "completed_at": None}
    identity = _findings_identity(detail.get("requirements"))
    if identity is None:
        return unknown
    matches = [run for run in detail.get("runs", [])
               if run.get("status") == "completed" and run.get("id")
               and _findings_identity(run.get("requirements")) == identity]
    recorded = detail.get("project", {}).get("results_run_id")
    bound = [run for run in matches if run["id"] == recorded]
    if len(bound) == 1:
        run = bound[0]
    elif len(matches) == 1:
        run = matches[0]
    else:
        return unknown
    return {"run_id": run["id"], "model": (run.get("models") or {}).get("generation"),
            "completed_at": run.get("completed_at")}


def with_results_provenance(detail):
    return {**detail, "results_provenance": results_provenance(detail)}
