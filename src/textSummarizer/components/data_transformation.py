from typing import Callable, Optional

from datasets import load_from_disk
from transformers import AutoTokenizer

from textSummarizer.entity import DataTransformationConfig
from textSummarizer.logging import logger
from textSummarizer.utils.common import import_from_string


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self._preprocess_fn: Optional[Callable] = None
        if config.preprocessing_fn:
            try:
                self._preprocess_fn = import_from_string(config.preprocessing_fn)
                logger.info("Loaded custom preprocessing function '%s'", config.preprocessing_fn)
            except ImportError as exc:
                logger.warning("Unable to import preprocessing function '%s': %s", config.preprocessing_fn, exc)

    def _tokenize_batch(self, example_batch):
        inputs = example_batch[self.config.input_column]
        targets = example_batch[self.config.target_column]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_input_length,
            truncation=True,
            padding="max_length",
        )
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.config.max_target_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def convert(self) -> None:
        dataset = load_from_disk(self.config.data_path)
        if self._preprocess_fn:
            logger.info("Applying custom preprocessing function before tokenization")
            dataset = dataset.map(self._preprocess_fn)
        logger.info("Tokenizing dataset for columns '%s' -> '%s'", self.config.input_column, self.config.target_column)
        tokenized_dataset = dataset.map(self._tokenize_batch, batched=True, remove_columns=None)
        tokenized_dataset.save_to_disk(self.config.transformed_data_path)
        logger.info("Saved tokenized dataset to %s", self.config.transformed_data_path)
