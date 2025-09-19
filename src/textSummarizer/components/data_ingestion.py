import urllib.request as request
import zipfile
from pathlib import Path

from datasets import load_dataset

from textSummarizer.entity import DataIngestionConfig
from textSummarizer.logging import logger
from textSummarizer.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _download_zip(self) -> None:
        if not self.config.source_url or not self.config.local_data_file:
            raise ValueError("Zip ingestion requires a source_url and local_data_file path.")
        local_path = Path(self.config.local_data_file)
        if local_path.exists():
            logger.info("Dataset archive already exists: %s (size: %s)", local_path, get_size(local_path))
            return
        logger.info("Downloading dataset from %s", self.config.source_url)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        request.urlretrieve(url=self.config.source_url, filename=local_path)
        logger.info("Download complete: %s (size: %s)", local_path, get_size(local_path))

    def _extract_zip(self) -> None:
        if not self.config.local_data_file:
            raise ValueError("Zip ingestion requires a local_data_file path.")
        dataset_path = Path(self.config.dataset_path)
        if dataset_path.exists():
            logger.info("Extracted dataset already available at %s", dataset_path)
            return
        archive_path = Path(self.config.local_data_file)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found at {archive_path}. Download it before extraction.")
        logger.info("Extracting archive %s to %s", archive_path, self.config.raw_data_dir)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(self.config.raw_data_dir)
        logger.info("Extraction complete. Dataset available at %s", dataset_path)

    def _ingest_zip_dataset(self) -> None:
        dataset_path = Path(self.config.dataset_path)
        if dataset_path.exists():
            logger.info("Dataset already prepared at %s", dataset_path)
            return
        self._download_zip()
        self._extract_zip()

    def _ingest_huggingface_dataset(self) -> None:
        if not self.config.huggingface_dataset:
            raise ValueError("Hugging Face ingestion requires a dataset name.")
        dataset_path = Path(self.config.dataset_path)
        if dataset_path.exists():
            logger.info("Hugging Face dataset already cached at %s", dataset_path)
            return
        logger.info(
            "Downloading Hugging Face dataset %s (subset=%s)",
            self.config.huggingface_dataset,
            self.config.huggingface_subset,
        )
        dataset = load_dataset(self.config.huggingface_dataset, self.config.huggingface_subset)
        dataset.save_to_disk(dataset_path)
        logger.info("Saved Hugging Face dataset to %s", dataset_path)

    def run(self) -> None:
        if self.config.ingestion_type == "zip":
            self._ingest_zip_dataset()
        elif self.config.ingestion_type == "huggingface":
            self._ingest_huggingface_dataset()
        else:
            raise ValueError(f"Unsupported ingestion type '{self.config.ingestion_type}'.")
        logger.info("Data ingestion complete for dataset '%s'", self.config.dataset_id)
