# SPDX-License-Identifier: MPL-2.0
"""Shared leased execution; callers inject provider-specific public diagnostics."""
from __future__ import annotations
import threading
from bidtrace.store import event, now
from bidtrace_core.contracts import JobCancelled, LeaseLostError
from bidtrace_v2.workflow import execute

def run_once(config, store, objects, worker_id, executor=execute, validation_error=()):
    job = store.claim(worker_id, config.workspace_id)
    if not job:
        return False
    stop, lost = threading.Event(), threading.Event()

    def renew():
        while not stop.wait(20):
            try:
                # Keep ownership while an already admitted model operation drains.
                store.heartbeat(job["id"], worker_id, allow_stopping=True)
            except Exception:
                lost.set()
                return

    keeper = threading.Thread(target=renew, daemon=True)
    keeper.start()
    def acknowledge_stop():
        try:
            store.complete_cancel(job["id"], worker_id, job["workspace_id"])
        except LeaseLostError:
            pass
    try:
        if job["status"] == "stopping":
            raise JobCancelled("Reclaimed stop request; no inference will resume.")
        result = executor(config, store, objects, job, worker_id, cancelled=lost)
        store.finish(job["id"], worker_id, result=result)
    except JobCancelled:
        acknowledge_stop()
        print("Analysis cancelled; saved work is available for resume.", flush=True)
    except LeaseLostError:
        print("Worker lease expired; another worker may resume the checkpoint.", flush=True)
    except Exception as exc:
        if not lost.is_set():
            message = f"Analysis stopped during {type(exc).__name__}. Sources and checkpoints are saved. Inspect the failed stage and retry."
            # Domain errors are curated by the workflow; external payloads are never returned to users.
            if isinstance(exc, validation_error):
                message = (exc.public_message + " Sources and validated progress are saved. "
                           "Review the failed stage, then resume after the issue is corrected.")
            elif isinstance(exc, ValueError):
                message = str(exc)[:400]
            def fail(d):
                run = next(r for r in d["runs"] if r["id"] == job["run_id"])
                run.update(status="failed", completed_at=now(), error=message)
                d["project"]["analysis_status"] = "failed"
                event(d, "analysis_failed", message)
                return d
            try:
                store.mutate_job(job["id"], worker_id, fail, job["workspace_id"], error=message,
                                 expected_run_id=job["run_id"], expected_tender_id=job["tender_id"])
            except JobCancelled:
                acknowledge_stop()
            except LeaseLostError:
                pass
        print(f"Analysis failure: {type(exc).__name__}", flush=True)
    finally:
        stop.set()
        keeper.join(timeout=2)
    return True
