from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DataIngestionConfig:
    dataset_id: str
    root_dir: Path
    ingestion_type: str
    download_dir: Path
    raw_data_dir: Path
    dataset_path: Path
    source_url: Optional[str] = None
    local_data_file: Optional[Path] = None
    extracted_folder_name: Optional[str] = None
    huggingface_dataset: Optional[str] = None
    huggingface_subset: Optional[str] = None


@dataclass(frozen=True)
class DataValidationConfig:
    dataset_id: str
    root_dir: Path
    data_path: Path
    required_splits: List[str]
    splits: Dict[str, str]
    status_file: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    dataset_id: str
    root_dir: Path
    data_path: Path
    tokenizer_name: str
    input_column: str
    target_column: str
    max_input_length: int
    max_target_length: int
    transformed_data_path: Path
    preprocessing_fn: Optional[str] = None


@dataclass(frozen=True)
class HyperparameterSearchConfig:
    enabled: bool
    backend: str
    direction: str
    n_trials: int
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentTrackingConfig:
    enabled: bool
    backend: str
    project: Optional[str]
    entity: Optional[str]
    run_name: str
    tags: Tuple[str, ...]
    mode: Optional[str]
    root_dir: Path
    dataset_id: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    dataset_id: str
    root_dir: Path
    data_path: Path
    model_ckpt: str
    train_split: str
    eval_split: str
    training_args: Dict[str, Any]
    hyperparameter_search: HyperparameterSearchConfig
    model_dir: Path
    tokenizer_dir: Path
    run_dir: Path
    columns: Dict[str, str]


@dataclass(frozen=True)
class ModelEvaluationConfig:
    dataset_id: str
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
    splits: Dict[str, str]
    input_column: str
    target_column: str
    generation_kwargs: Dict[str, Any]
    metric_names: Tuple[str, ...]
    sample_size: Optional[int]
    input_max_length: int
    target_max_length: int
