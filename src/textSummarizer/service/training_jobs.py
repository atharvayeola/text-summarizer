import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from textSummarizer.logging import logger
from textSummarizer.pipeline.training import run_training_pipeline


class TrainingJobManager:
    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        dataset_id: Optional[str] = None,
        enable_hyperparameter_search: Optional[bool] = None,
        hyperparameter_trials: Optional[int] = None,
        experiment_name: Optional[str] = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        job_info = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "status": "queued",
            "submitted_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "experiment_name": experiment_name,
        }
        with self._lock:
            self._jobs[job_id] = job_info
        self._executor.submit(
            self._run_job,
            job_id,
            dataset_id,
            enable_hyperparameter_search,
            hyperparameter_trials,
            experiment_name,
        )
        return job_id

    def _run_job(
        self,
        job_id: str,
        dataset_id: Optional[str],
        enable_hyperparameter_search: Optional[bool],
        hyperparameter_trials: Optional[int],
        experiment_name: Optional[str],
    ) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["started_at"] = time.time()
        try:
            run_training_pipeline(
                dataset_id=dataset_id,
                enable_hyperparameter_search=enable_hyperparameter_search,
                hyperparameter_trials=hyperparameter_trials,
                experiment_name=experiment_name,
            )
            status = "completed"
            error = None
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Training job %s failed", job_id)
            status = "failed"
            error = str(exc)
        with self._lock:
            self._jobs[job_id]["status"] = status
            self._jobs[job_id]["finished_at"] = time.time()
            self._jobs[job_id]["error"] = error

    def get(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            return dict(job)

    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {job_id: dict(info) for job_id, info in self._jobs.items()}
