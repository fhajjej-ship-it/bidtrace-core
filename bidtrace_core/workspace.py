# SPDX-License-Identifier: MPL-2.0
"""Shared new-project schema; no HTTP or UI dependencies."""
from bidtrace.store import uid, counts, now

def new_detail(payload, provenance=None):
    p = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
    return {"project": {"id": uid("tender"), "title": p["title"].strip(), "buyer": p.get("buyer", ""),
            "country": p.get("country", ""), "category": "Tender document review",
            "provenance": provenance or p.get("provenance", "user"), "coverage": "No documents uploaded yet.",
            "active_version": 0, "previous_version": None, "analysis_status": "pending", "counts": counts([]),
            "updated_at": now(), "metadata": {}, "last_run_id": None, "last_job_id": None},
            "documents": [], "requirements": [], "runs": [], "reviews": [], "events": []}
