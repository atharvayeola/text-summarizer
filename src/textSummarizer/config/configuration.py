from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textSummarizer.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ExperimentTrackingConfig,
    HyperparameterSearchConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from textSummarizer.utils.common import create_directories, read_yaml


class ConfigurationManager:
    """Hydrates strongly typed configuration objects for each pipeline stage."""

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        dataset_id: Optional[str] = None,
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

        self._available_datasets = list(self.config.datasets.keys())
        self.dataset_id = dataset_id or self.config.get("default_dataset")
        if not self.dataset_id:
            raise ValueError("Dataset identifier is required. Set default_dataset in the config or pass dataset_id explicitly.")
        if self.dataset_id not in self.config.datasets:
            raise ValueError(
                f"Dataset '{self.dataset_id}' is not defined in config/config.yaml. Available options: {', '.join(self._available_datasets)}"
            )
        self.dataset_config = self.config.datasets[self.dataset_id]

        self._data_ingestion_config: Optional[DataIngestionConfig] = None
        self._data_validation_config: Optional[DataValidationConfig] = None
        self._data_transformation_config: Optional[DataTransformationConfig] = None
        self._model_trainer_config: Optional[ModelTrainerConfig] = None
        self._model_evaluation_config: Optional[ModelEvaluationConfig] = None

    @property
    def available_datasets(self) -> List[str]:
        return list(self._available_datasets)

    def _stage_root(self, stage_root: str) -> Path:
        root = Path(stage_root) / self.dataset_id
        create_directories([root])
        return root

    def _raw_dataset_path(self) -> Path:
        ingestion_cfg = self.dataset_config.ingestion
        stage_root = self._stage_root(self.config.data_ingestion.root_dir)
        raw_root = stage_root / "raw"
        create_directories([raw_root])
        if ingestion_cfg.type.lower() == "zip":
            extracted = ingestion_cfg.get("extracted_folder_name")
            if extracted:
                return raw_root / extracted
        return raw_root

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        if self._data_ingestion_config:
            return self._data_ingestion_config

        ingestion_root = self._stage_root(self.config.data_ingestion.root_dir)
        download_dir = ingestion_root / "downloads"
        raw_root = ingestion_root / "raw"
        create_directories([download_dir, raw_root])

        dataset_path = self._raw_dataset_path()
        create_directories([dataset_path.parent])

        ingestion_cfg = self.dataset_config.ingestion
        ingestion_type = ingestion_cfg.type.lower()

        local_data_file = None
        if ingestion_type == "zip":
            archive_filename = ingestion_cfg.get("archive_filename") or f"{self.dataset_id}.zip"
            local_data_file = download_dir / archive_filename
        elif ingestion_type == "huggingface":
            archive_filename = None
        else:
            raise ValueError(f"Unsupported ingestion type '{ingestion_type}' for dataset '{self.dataset_id}'.")

        config = DataIngestionConfig(
            dataset_id=self.dataset_id,
            root_dir=ingestion_root,
            ingestion_type=ingestion_type,
            download_dir=download_dir,
            raw_data_dir=raw_root,
            dataset_path=dataset_path,
            source_url=ingestion_cfg.get("source_url"),
            local_data_file=local_data_file,
            extracted_folder_name=ingestion_cfg.get("extracted_folder_name"),
            huggingface_dataset=ingestion_cfg.get("dataset_name"),
            huggingface_subset=ingestion_cfg.get("subset"),
        )
        self._data_ingestion_config = config
        return config

    def get_data_validation_config(self) -> DataValidationConfig:
        if self._data_validation_config:
            return self._data_validation_config

        validation_root = self._stage_root(self.config.data_validation.root_dir)
        status_file = validation_root / self.config.data_validation.status_file_name
        data_path = self.get_data_ingestion_config().dataset_path
        required_splits = list(self.config.data_validation.required_splits)
        splits = dict(self.dataset_config.splits)

        config = DataValidationConfig(
            dataset_id=self.dataset_id,
            root_dir=validation_root,
            data_path=data_path,
            required_splits=required_splits,
            splits=splits,
            status_file=status_file,
        )
        self._data_validation_config = config
        return config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        if self._data_transformation_config:
            return self._data_transformation_config

        transformation_root = self._stage_root(self.config.data_transformation.root_dir)
        transformed_path = transformation_root / "tokenized"
        create_directories([transformation_root])

        dataset_meta = self.dataset_config
        config = DataTransformationConfig(
            dataset_id=self.dataset_id,
            root_dir=transformation_root,
            data_path=self.get_data_ingestion_config().dataset_path,
            tokenizer_name=self.config.data_transformation.tokenizer_name,
            input_column=dataset_meta.columns.input_text,
            target_column=dataset_meta.columns.target_text,
            max_input_length=int(dataset_meta.max_input_length),
            max_target_length=int(dataset_meta.max_target_length),
            transformed_data_path=transformed_path,
            preprocessing_fn=dataset_meta.get("preprocessing_fn"),
        )
        self._data_transformation_config = config
        return config

    def get_experiment_tracking_config(
        self,
        run_name_override: Optional[str] = None,
        stage: str = "train",
    ) -> ExperimentTrackingConfig:
        tracking_root = self._stage_root(self.config.experiment_tracking.root_dir)
        params = self.params.get("experiment_tracking", {})
        enabled = bool(params.get("enabled", False))
        backend = params.get("backend", "local")
        project = params.get("project")
        entity = params.get("entity")
        configured_run_name = params.get("run_name")
        tags = tuple(params.get("tags", []))
        mode = params.get("mode")

        base_name = run_name_override or configured_run_name
        if base_name:
            run_name = f"{base_name}-{stage}" if stage and not base_name.endswith(stage) else base_name
        else:
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            run_name = f"{self.dataset_id}-{stage}-{timestamp}"

        return ExperimentTrackingConfig(
            enabled=enabled,
            backend=backend,
            project=project,
            entity=entity,
            run_name=run_name,
            tags=tags,
            mode=mode,
            root_dir=tracking_root,
            dataset_id=self.dataset_id,
        )

    def get_model_trainer_config(
        self,
        run_name: Optional[str] = None,
        enable_hyperparameter_search: Optional[bool] = None,
        hyperparameter_trials: Optional[int] = None,
    ) -> ModelTrainerConfig:
        if self._model_trainer_config:
            return self._model_trainer_config

        trainer_root = self._stage_root(self.config.model_trainer.root_dir)
        model_dir = trainer_root / "model"
        tokenizer_dir = trainer_root / "tokenizer"
        run_artifacts_dir = trainer_root / "runs"
        create_directories([model_dir, tokenizer_dir, run_artifacts_dir])

        run_subdir = run_name or "latest"
        run_dir = run_artifacts_dir / run_subdir
        create_directories([run_dir])

        training_args = self.params.training.arguments.to_dict()
        training_args["output_dir"] = str(run_dir)
        training_args.setdefault("logging_dir", str(run_dir / "logs"))
        training_args.setdefault("overwrite_output_dir", True)

        hp_params = self.params.training.hyperparameter_search.to_dict()
        if enable_hyperparameter_search is not None:
            hp_params["enabled"] = bool(enable_hyperparameter_search)
        if hyperparameter_trials is not None:
            hp_params["n_trials"] = int(hyperparameter_trials)

        hyperparameter_search_config = HyperparameterSearchConfig(
            enabled=bool(hp_params.get("enabled", False)),
            backend=hp_params.get("backend", "optuna"),
            direction=hp_params.get("direction", "minimize"),
            n_trials=int(hp_params.get("n_trials", 0)),
            params=dict(hp_params.get("params", {})),
        )

        dataset_meta = self.dataset_config
        splits = dict(dataset_meta.splits)

        config = ModelTrainerConfig(
            dataset_id=self.dataset_id,
            root_dir=trainer_root,
            data_path=self.get_data_transformation_config().transformed_data_path,
            model_ckpt=self.config.model_trainer.model_ckpt,
            train_split=splits.get("train", "train"),
            eval_split=splits.get("validation", "validation"),
            training_args=training_args,
            hyperparameter_search=hyperparameter_search_config,
            model_dir=model_dir,
            tokenizer_dir=tokenizer_dir,
            run_dir=run_dir,
            columns=dict(dataset_meta.columns),
        )
        self._model_trainer_config = config
        return config

    def get_model_evaluation_config(self, run_name: Optional[str] = None) -> ModelEvaluationConfig:
        if self._model_evaluation_config:
            return self._model_evaluation_config

        evaluation_root = self._stage_root(self.config.model_evaluation.root_dir)
        metric_file = evaluation_root / "metrics.csv"

        dataset_meta = self.dataset_config
        generation_kwargs = dict(self.config.model_evaluation.generation)
        metric_section = self.config.model_evaluation.metrics
        metric_names = tuple(metric_section.names)
        sample_size = metric_section.get("sample_size")

        config = ModelEvaluationConfig(
            dataset_id=self.dataset_id,
            root_dir=evaluation_root,
            data_path=self.get_data_transformation_config().transformed_data_path,
            model_path=self.get_model_trainer_config(run_name=run_name).model_dir,
            tokenizer_path=self.get_model_trainer_config(run_name=run_name).tokenizer_dir,
            metric_file_name=metric_file,
            splits=dict(dataset_meta.splits),
            input_column=dataset_meta.columns.input_text,
            target_column=dataset_meta.columns.target_text,
            generation_kwargs=generation_kwargs,
            metric_names=metric_names,
            sample_size=sample_size,
            input_max_length=int(dataset_meta.max_input_length),
            target_max_length=int(dataset_meta.max_target_length),
        )
        self._model_evaluation_config = config
        return config
