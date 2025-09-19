import json
from pathlib import Path
from typing import Dict, Optional

from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from textSummarizer.entity import ExperimentTrackingConfig, ModelTrainerConfig
from textSummarizer.logging import logger
from textSummarizer.utils.tracking import ExperimentTracker, ExperimentTrackerCallback


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, tracking_config: ExperimentTrackingConfig):
        self.config = config
        self.tracker = ExperimentTracker(tracking_config)
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt)
        self._data_collator = DataCollatorForSeq2Seq(self._tokenizer)

    def _model_init(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)

    def _build_trainer(self, training_args: Dict, train_dataset, eval_dataset) -> Trainer:
        args = TrainingArguments(**training_args)
        callbacks = [ExperimentTrackerCallback(self.tracker)] if self.tracker.enabled else []
        return Trainer(
            args=args,
            tokenizer=self._tokenizer,
            data_collator=self._data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=self._model_init,
            callbacks=callbacks,
        )

    def _run_hyperparameter_search(self, trainer: Trainer) -> Optional[Dict[str, float]]:
        hp_config = self.config.hyperparameter_search
        if not hp_config.enabled or hp_config.n_trials <= 0 or not hp_config.params:
            return None

        logger.info(
            "Starting hyperparameter search (%s trials, direction=%s)",
            hp_config.n_trials,
            hp_config.direction,
        )

        def hp_space(trial):  # type: ignore[override]
            space = {}
            for name, spec in hp_config.params.items():
                distribution = str(spec.get("distribution", "float")).lower()
                if distribution == "float":
                    space[name] = trial.suggest_float(
                        name,
                        float(spec["low"]),
                        float(spec["high"]),
                        log=bool(spec.get("log", False)),
                    )
                elif distribution == "int":
                    space[name] = trial.suggest_int(
                        name,
                        int(spec["low"]),
                        int(spec["high"]),
                        step=int(spec.get("step", 1)),
                        log=bool(spec.get("log", False)),
                    )
                elif distribution == "categorical":
                    space[name] = trial.suggest_categorical(name, list(spec.get("values", [])))
                else:
                    raise ValueError(f"Unsupported distribution '{distribution}' for hyperparameter '{name}'")
            return space

        best_run = trainer.hyperparameter_search(
            direction=hp_config.direction,
            n_trials=hp_config.n_trials,
            hp_space=hp_space,
        )
        best_params = dict(best_run.hyperparameters)
        logger.info("Best hyperparameters: %s (objective=%.4f)", best_params, best_run.objective)
        self.tracker.log_params({f"best_{k}": v for k, v in best_params.items()})
        search_summary = {
            "best_params": best_params,
            "objective": best_run.objective,
            "direction": hp_config.direction,
        }
        with open(Path(self.config.run_dir) / "hyperparameter_search.json", "w", encoding="utf-8") as file:
            json.dump(search_summary, file, indent=2)
        return best_params

    def train(self) -> None:
        training_args = dict(self.config.training_args)
        dataset = load_from_disk(self.config.data_path)
        train_dataset = dataset[self.config.train_split]
        eval_dataset = dataset[self.config.eval_split]
        run_metadata = {
            "dataset_id": self.config.dataset_id,
            "model_ckpt": self.config.model_ckpt,
            "train_split": self.config.train_split,
            "eval_split": self.config.eval_split,
        }
        self.tracker.start_run(run_metadata)
        self.tracker.log_params({"initial_training_args": training_args})

        trainer = self._build_trainer(training_args, train_dataset, eval_dataset)
        best_params: Optional[Dict[str, float]] = None
        try:
            best_params = self._run_hyperparameter_search(trainer)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Hyperparameter search failed: %s", exc)
        if best_params:
            logger.info("Rebuilding trainer with best hyperparameters")
            training_args.update(best_params)
            self.tracker.log_params({"final_training_args": training_args})
            trainer = self._build_trainer(training_args, train_dataset, eval_dataset)
        else:
            self.tracker.log_params({"final_training_args": training_args})

        try:
            trainer.train()
            trainer.save_model(self.config.model_dir)
            self._tokenizer.save_pretrained(self.config.tokenizer_dir)
            with open(Path(self.config.run_dir) / "training_args.json", "w", encoding="utf-8") as file:
                json.dump(training_args, file, indent=2)
            logger.info("Model saved to %s", self.config.model_dir)
        finally:
            self.tracker.finish()
