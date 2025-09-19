from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger


class DataIngestionTrainingPipeline:
    def __init__(self, configuration: ConfigurationManager):
        self.configuration = configuration

    def main(self) -> None:
        data_ingestion_config = self.configuration.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.run()
        logger.info("Data ingestion stage completed for dataset '%s'", data_ingestion_config.dataset_id)
