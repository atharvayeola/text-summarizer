from typing import Optional

from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(
        self,
        configuration: ConfigurationManager,
        run_name: Optional[str] = None,
        enable_hyperparameter_search: Optional[bool] = None,
        hyperparameter_trials: Optional[int] = None,
    ):
        self.configuration = configuration
        self.run_name = run_name
        self.enable_hyperparameter_search = enable_hyperparameter_search
        self.hyperparameter_trials = hyperparameter_trials

    def main(self) -> None:
        tracking_config = self.configuration.get_experiment_tracking_config(
            run_name_override=self.run_name,
            stage="train",
        )
        trainer_config = self.configuration.get_model_trainer_config(
            run_name=tracking_config.run_name,
            enable_hyperparameter_search=self.enable_hyperparameter_search,
            hyperparameter_trials=self.hyperparameter_trials,
        )
        trainer = ModelTrainer(config=trainer_config, tracking_config=tracking_config)
        trainer.train()
        logger.info("Model training completed for dataset '%s'", trainer_config.dataset_id)
