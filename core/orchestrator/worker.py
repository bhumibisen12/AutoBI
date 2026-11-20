"""Thread-pool background worker so long AutoBI orchestrator steps never freeze Streamlit."""

from __future__ import annotations

import contextlib
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from core.logging import get_logger
from core.schemas.orchestrator import BackgroundJobStatus

logger = get_logger("orchestrator.worker")


class BackgroundWorker:
    """Submit long-running orchestrator tasks to a thread pool with live progress tracking."""

    def __init__(self, max_workers: int = 2) -> None:
        """Initialize the background worker pool."""
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="autobi")
        self._lock = threading.Lock()
        self._statuses: dict[str, BackgroundJobStatus] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._results: dict[str, Any] = {}
        self._latest_by_session: dict[str, str] = {}

    def submit(
        self,
        session_id: str,
        stage: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> BackgroundJobStatus:
        """Submit ``fn`` to run asynchronously and return its initial ``BackgroundJobStatus``."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        status = BackgroundJobStatus(
            job_id=job_id,
            session_id=session_id,
            stage=stage,
            status="queued",
            progress=0.05,
            message=f"Queued {stage}...",
        )
        with self._lock:
            self._statuses[job_id] = status
            self._latest_by_session[session_id] = job_id

        def _update_progress(progress: float, message: str) -> None:
            with self._lock:
                cur = self._statuses[job_id]
                self._statuses[job_id] = cur.model_copy(
                    update={
                        "status": "running",
                        "progress": max(0.0, min(1.0, progress)),
                        "message": message,
                    }
                )

        def _runner() -> Any:
            _update_progress(0.15, f"Running {stage}...")
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                logger.exception("Background job failed: %s (%s)", job_id, stage)
                with self._lock:
                    cur = self._statuses[job_id]
                    self._statuses[job_id] = cur.model_copy(
                        update={
                            "status": "failed",
                            "progress": 1.0,
                            "message": f"{stage} failed: {exc}",
                            "error": str(exc),
                            "finished_at": datetime.now(UTC),
                        }
                    )
                raise

            with self._lock:
                self._results[job_id] = result
                cur = self._statuses[job_id]
                self._statuses[job_id] = cur.model_copy(
                    update={
                        "status": "completed",
                        "progress": 1.0,
                        "message": f"{stage} completed.",
                        "finished_at": datetime.now(UTC),
                    }
                )
            return result

        future = self._executor.submit(_runner)
        with self._lock:
            self._futures[job_id] = future
        return status

    def get_status(self, job_id: str) -> BackgroundJobStatus | None:
        """Return the latest ``BackgroundJobStatus`` snapshot for ``job_id``."""
        with self._lock:
            status = self._statuses.get(job_id)
            return status.model_copy(deep=True) if status is not None else None

    def latest_for_session(self, session_id: str) -> BackgroundJobStatus | None:
        """Return the most recent job status submitted for ``session_id``."""
        with self._lock:
            job_id = self._latest_by_session.get(session_id)
            if job_id is None:
                return None
            status = self._statuses.get(job_id)
            return status.model_copy(deep=True) if status is not None else None

    def get_result(self, job_id: str) -> Any:
        """Return the completed task result for ``job_id``, or ``None`` if not ready."""
        with self._lock:
            return self._results.get(job_id)

    def wait(self, job_id: str, timeout: float | None = None) -> BackgroundJobStatus:
        """Block until ``job_id`` completes (primarily for tests) and return final status."""
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            with contextlib.suppress(Exception):
                future.result(timeout=timeout)
        status = self.get_status(job_id)
        if status is None:
            raise KeyError(f"Unknown background job: {job_id}")
        return status

    def shutdown(self, *, wait: bool = False) -> None:
        """Shut down the worker thread pool."""
        self._executor.shutdown(wait=wait)
