from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self, configuration: ConfigurationManager):
        self.configuration = configuration

    def main(self) -> None:
        transformation_config = self.configuration.get_data_transformation_config()
        transformer = DataTransformation(config=transformation_config)
        transformer.convert()
        logger.info("Data transformation completed for dataset '%s'", transformation_config.dataset_id)
