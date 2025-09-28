"""Async inference orchestration using a worker queue."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from fastapi import WebSocket

from observability import update_queue_depth

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceJob:
    job_id: str
    websocket: WebSocket
    audio_data: bytes
    sample_rate: int


class InferenceOrchestrator:
    """Coordinates inference jobs across worker tasks.

    Designed to support both in-process batching and future external queue
    backends (FastStream/Redis). The `backend_name` label feeds Prometheus
    metrics so alternate dispatchers can be wired in with minimal code changes.
    """

    def __init__(
        self,
        processor: Callable[[InferenceJob], Awaitable[None]],
        workers: int = 1,
        backend_name: str = "in-memory",
        batch_size: int = 1,
    ) -> None:
        self._processor = processor
        self._workers_count = max(1, workers)
        self._queue: asyncio.Queue[Optional[InferenceJob]] = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._backend_name = backend_name
        self._batch_size = max(1, batch_size)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        LOGGER.info("Starting inference orchestrator with %s worker(s)", self._workers_count)
        for _ in range(self._workers_count):
            task = asyncio.create_task(self._worker_loop())
            self._workers.append(task)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        LOGGER.info("Inference orchestrator stopped")
        update_queue_depth(0, self._backend_name)

    async def enqueue(self, websocket: WebSocket, audio_data: bytes, sample_rate: int) -> str:
        job_id = str(uuid.uuid4())
        job = InferenceJob(job_id=job_id, websocket=websocket, audio_data=audio_data, sample_rate=sample_rate)
        await self._queue.put(job)
        update_queue_depth(self._queue.qsize(), self._backend_name)
        LOGGER.info("Job %s enqueued", job_id)
        return job_id

    async def _worker_loop(self) -> None:
        while True:
            job = await self._queue.get()
            if job is None:
                break

            jobs_batch = [job]
            if self._batch_size > 1:
                try:
                    for _ in range(self._batch_size - 1):
                        next_job = self._queue.get_nowait()
                        if next_job is None:
                            await self._queue.put(None)
                            break
                        jobs_batch.append(next_job)
                except asyncio.QueueEmpty:
                    pass

            update_queue_depth(self._queue.qsize(), self._backend_name)

            for job_item in jobs_batch:
                try:
                    LOGGER.info("Worker processing job %s", job_item.job_id)
                    await self._processor(job_item)
                    LOGGER.info("Job %s completed", job_item.job_id)
                except Exception as exc:  # pragma: no cover - runtime logging only
                    LOGGER.exception("Job %s failed: %s", job_item.job_id, exc)
