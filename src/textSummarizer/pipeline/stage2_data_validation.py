from textSummarizer.components.data_validation import DataValidation
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger


class DataValidationTrainingPipeline:
    def __init__(self, configuration: ConfigurationManager):
        self.configuration = configuration

    def main(self) -> None:
        validation_config = self.configuration.get_data_validation_config()
        validator = DataValidation(config=validation_config)
        status = validator.validate_all_files_exist()
        if not status:
            raise RuntimeError("Data validation failed. Please ensure all required splits are available.")
        logger.info("Data validation successful for dataset '%s'", validation_config.dataset_id)
