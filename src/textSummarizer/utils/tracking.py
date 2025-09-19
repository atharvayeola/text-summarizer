import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from transformers.trainer_callback import TrainerCallback

from textSummarizer.entity import ExperimentTrackingConfig
from textSummarizer.logging import logger


class ExperimentTracker:
    """Lightweight abstraction over experiment tracking backends."""

    def __init__(self, config: ExperimentTrackingConfig):
        self.config = config
        self.enabled = bool(config.enabled)
        self.backend = (config.backend or "local").lower()
        self._run_started = False
        self._log_file: Optional[Path] = None
        self._metadata_file: Optional[Path] = None
        self._wandb_run = None

        if self.enabled and self.backend not in {"local", "wandb"}:
            raise ValueError(f"Unsupported experiment tracking backend: {self.backend}")

    @property
    def run_dir(self) -> Path:
        return Path(self.config.root_dir) / self.config.run_name

    def start_run(self, params: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self._run_started:
            return

        params = params or {}
        if self.backend == "local":
            run_dir = self.run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = run_dir / "metrics.jsonl"
            self._metadata_file = run_dir / "metadata.json"
            metadata = {
                "run_name": self.config.run_name,
                "dataset_id": self.config.dataset_id,
                "backend": self.backend,
                "tags": list(self.config.tags),
                "project": self.config.project,
                "entity": self.config.entity,
                "mode": self.config.mode,
                "params": params,
                "started_at": datetime.utcnow().isoformat(),
            }
            with open(self._metadata_file, "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, indent=2)
        elif self.backend == "wandb":
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError("W&B backend requested but package 'wandb' is not installed.") from exc
            if self.config.mode:
                os.environ.setdefault("WANDB_MODE", self.config.mode)
            self._wandb_run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.config.run_name,
                tags=list(self.config.tags) or None,
                reinit=True,
            )
            if params:
                self._wandb_run.config.update(params, allow_val_change=True)
        self._run_started = True
        logger.info("Experiment tracking run '%s' started (backend=%s)", self.config.run_name, self.backend)

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled or not params:
            return
        if self.backend == "local" and self._metadata_file:
            with open(self._metadata_file, "r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
            metadata.setdefault("params", {}).update(params)
            with open(self._metadata_file, "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, indent=2)
        elif self.backend == "wandb" and self._wandb_run:
            self._wandb_run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or not metrics:
            return
        if self.backend == "local" and self._log_file:
            entry = {
                "step": step,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }
            with open(self._log_file, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(entry) + "\n")
        elif self.backend == "wandb" and self._wandb_run:
            import wandb

            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if not self.enabled or not self._run_started:
            return
        if self.backend == "local" and self._metadata_file:
            with open(self._metadata_file, "r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
            metadata["finished_at"] = datetime.utcnow().isoformat()
            with open(self._metadata_file, "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, indent=2)
        elif self.backend == "wandb" and self._wandb_run:
            import wandb

            wandb.finish()
            self._wandb_run = None
        self._run_started = False
        logger.info("Experiment tracking run '%s' finished", self.config.run_name)


class ExperimentTrackerCallback(TrainerCallback):
    """Hook Trainer logs into the experiment tracker."""

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs:
            self.tracker.log_metrics(logs, step=state.global_step)
