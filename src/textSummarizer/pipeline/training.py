from typing import Optional

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger
from textSummarizer.pipeline.stage1_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage2_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage3_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage4_model_trainer import ModelTrainerTrainingPipeline
from textSummarizer.pipeline.stage5_model_evaluation import ModelEvaluationTrainingPipeline


def run_training_pipeline(
    dataset_id: Optional[str] = None,
    enable_hyperparameter_search: Optional[bool] = None,
    hyperparameter_trials: Optional[int] = None,
    experiment_name: Optional[str] = None,
) -> str:
    """Execute the end-to-end training workflow for the selected dataset."""

    configuration_manager = ConfigurationManager(dataset_id=dataset_id)
    dataset_label = configuration_manager.dataset_id
    logger.info("Starting training pipeline for dataset '%s'", dataset_label)

    ingestion_pipeline = DataIngestionTrainingPipeline(configuration_manager)
    ingestion_pipeline.main()

    validation_pipeline = DataValidationTrainingPipeline(configuration_manager)
    validation_pipeline.main()

    transformation_pipeline = DataTransformationTrainingPipeline(configuration_manager)
    transformation_pipeline.main()

    trainer_pipeline = ModelTrainerTrainingPipeline(
        configuration=configuration_manager,
        run_name=experiment_name,
        enable_hyperparameter_search=enable_hyperparameter_search,
        hyperparameter_trials=hyperparameter_trials,
    )
    trainer_pipeline.main()

    evaluation_pipeline = ModelEvaluationTrainingPipeline(
        configuration=configuration_manager,
        run_name=experiment_name,
    )
    evaluation_pipeline.main()

    logger.info("Training pipeline finished for dataset '%s'", dataset_label)
    return dataset_label
