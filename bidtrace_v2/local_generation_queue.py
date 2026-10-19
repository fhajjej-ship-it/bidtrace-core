# SPDX-License-Identifier: MPL-2.0
"""One GPU owner coalesces bounded independent requests, without sharing answers."""
from concurrent.futures import Future
from queue import Queue, Empty
from threading import Thread
import time


class LocalGenerationQueue:
    def __init__(self, generate, *, limit=8, collection_seconds=.05):
        if type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError('Local GPU queue limit must be one to eight')
        self.generate, self.limit, self.collection_seconds = generate, limit, collection_seconds
        self.queue = Queue(maxsize=limit * 2)
        self.worker = Thread(target=self._work, name='bidtrace-local-gpu', daemon=True)
        self.worker.start()

    def submit(self, job):
        future = Future()
        self.queue.put((job, future))
        return future.result()

    def _work(self):
        while True:
            first = self.queue.get()
            batch = [first]
            deadline = time.monotonic() + self.collection_seconds
            while len(batch) < self.limit:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except Empty:
                    break
            try:
                results = self.generate([job for job, _ in batch])
                if len(results) != len(batch):
                    raise ValueError('Native batch must return one result per request')
                for (_, future), result in zip(batch, results):
                    future.set_result(result)
            except BaseException as exc:
                for _, future in batch:
                    future.set_exception(exc)
            finally:
                for _ in batch:
                    self.queue.task_done()
