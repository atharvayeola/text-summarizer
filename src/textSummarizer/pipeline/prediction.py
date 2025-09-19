from threading import Lock
from typing import Any, Dict, Optional

from transformers import AutoTokenizer, pipeline

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger


class PredictionPipeline:
    _pipelines: Dict[str, Any] = {}
    _locks: Dict[str, Lock] = {}
    _registry_lock = Lock()

    def __init__(self, dataset_id: Optional[str] = None):
        self.configuration_manager = ConfigurationManager(dataset_id=dataset_id)
        self.config = self.configuration_manager.get_model_evaluation_config()

    @classmethod
    def warmup(cls, dataset_id: Optional[str] = None) -> None:
        instance = cls(dataset_id=dataset_id)
        instance._get_or_create_pipeline()

    @classmethod
    def _get_dataset_lock(cls, dataset_id: str) -> Lock:
        with cls._registry_lock:
            if dataset_id not in cls._locks:
                cls._locks[dataset_id] = Lock()
            return cls._locks[dataset_id]

    def _get_or_create_pipeline(self):
        dataset_id = self.config.dataset_id
        if dataset_id in self._pipelines:
            return self._pipelines[dataset_id]
        dataset_lock = self._get_dataset_lock(dataset_id)
        with dataset_lock:
            if dataset_id not in self._pipelines:
                logger.info("Loading summarization pipeline for dataset '%s'", dataset_id)
                tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
                summarizer = pipeline(
                    "summarization",
                    model=str(self.config.model_path),
                    tokenizer=tokenizer,
                )
                self._pipelines[dataset_id] = summarizer
        return self._pipelines[dataset_id]

    def predict(self, text: str) -> str:
        summarizer = self._get_or_create_pipeline()
        logger.debug("Running prediction for dataset '%s'", self.config.dataset_id)
        result = summarizer(text, **self.config.generation_kwargs)
        summary = result[0]["summary_text"] if result else ""
        return summary
