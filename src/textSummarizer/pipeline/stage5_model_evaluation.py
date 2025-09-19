from typing import Optional

from textSummarizer.components.model_evaluation import ModelEvaluation
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger


class ModelEvaluationTrainingPipeline:
    def __init__(self, configuration: ConfigurationManager, run_name: Optional[str] = None):
        self.configuration = configuration
        self.run_name = run_name

    def main(self) -> None:
        evaluation_config = self.configuration.get_model_evaluation_config(run_name=self.run_name)
        tracking_config = self.configuration.get_experiment_tracking_config(
            run_name_override=self.run_name,
            stage="evaluation",
        )
        evaluator = ModelEvaluation(config=evaluation_config, tracking_config=tracking_config)
        metrics = evaluator.evaluate()
        logger.info("Evaluation metrics for dataset '%s': %s", evaluation_config.dataset_id, metrics)
