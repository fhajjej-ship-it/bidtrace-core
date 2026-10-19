# SPDX-License-Identifier: MPL-2.0
"""Bounded admission for independent cloud work; drain started work on failure."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import RLock


class PeerAborted(RuntimeError):
    """A peer already failed; this work must not begin another request."""


class WorkAdmission:
    def __init__(self):
        self._lock = RLock()
        self.error = None

    def stop(self, error):
        with self._lock:
            if self.error is None or (isinstance(self.error, PeerAborted) and not isinstance(error, PeerAborted)):
                self.error = error

    def raise_if_stopped(self):
        with self._lock:
            if self.error is not None:
                raise PeerAborted("Independent cloud work stopped after a peer failed.") from self.error


def bounded_map(fn, items, *, max_workers, check=None, admission=None):
    """Map in input order, with only a bounded number of submitted items.

    Workers latch the first failure immediately, before the coordinator observes
    the future. Callers may share admission with request cancellation callbacks.
    All started functions finish (including durable receipts) before an error is
    surfaced. A peer-abort never replaces the original error.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    admission = admission or WorkAdmission()
    source, results, pending = iter(items), [], {}

    def boundary():
        admission.raise_if_stopped()
        if check:
            check()

    def run(item):
        try:
            boundary()
            return fn(item)
        except BaseException as exc:
            admission.stop(exc)
            raise

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bidtrace-cloud") as pool:
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < max_workers:
                    # Stop publication and task admission share the same lock.
                    with admission._lock:
                        boundary()
                        try:
                            item = next(source)
                        except StopIteration:
                            exhausted = True
                            break
                        index = len(results)
                        results.append(None)
                        pending[pool.submit(run, item)] = index
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    results[pending.pop(future)] = future.result()
        except BaseException as exc:
            admission.stop(exc)
            # The context manager drains every admitted item before returning.
            # Cancellation checks prevent peers from beginning another request.
    if admission.error is not None:
        raise admission.error
    return results
