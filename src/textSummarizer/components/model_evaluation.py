import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from datasets import load_from_disk, load_metric
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.entity import ExperimentTrackingConfig, ModelEvaluationConfig
from textSummarizer.logging import logger
from textSummarizer.utils.tracking import ExperimentTracker


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, tracking_config: Optional[ExperimentTrackingConfig] = None):
        self.config = config
        self.tracker = ExperimentTracker(tracking_config) if tracking_config else ExperimentTracker(
            ExperimentTrackingConfig(
                enabled=False,
                backend="local",
                project=None,
                entity=None,
                run_name="evaluation",
                tags=(),
                mode=None,
                root_dir=config.root_dir,
                dataset_id=config.dataset_id,
            )
        )

    def _generate_batches(self, items, batch_size: int):
        for index in range(0, len(items), batch_size):
            yield items[index : index + batch_size]

    def _calculate_metrics(
        self,
        dataset_split,
        model,
        tokenizer,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        metric = load_metric("rouge")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        input_column = self.config.input_column
        target_column = self.config.target_column
        generation_kwargs = dict(self.config.generation_kwargs)

        article_batches = list(self._generate_batches(dataset_split[input_column], batch_size))
        target_batches = list(self._generate_batches(dataset_split[target_column], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches),
            total=len(article_batches),
            desc="Evaluating",
        ):
            inputs = tokenizer(
                article_batch,
                max_length=self.config.input_max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                **generation_kwargs,
            )
            decoded = [
                tokenizer.decode(summary, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for summary in summaries
            ]
            metric.add_batch(predictions=decoded, references=target_batch)

        scores = metric.compute()
        rouge_scores = {}
        for name in self.config.metric_names:
            score = scores[name]
            rouge_scores[name] = score.mid.fmeasure if hasattr(score, "mid") else score
        return rouge_scores

    def evaluate(self, batch_size: int = 8) -> Dict[str, float]:
        logger.info("Starting model evaluation for dataset '%s'", self.config.dataset_id)
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
        dataset = load_from_disk(self.config.data_path)
        test_split = dataset[self.config.splits.get("test", "test")]
        if self.config.sample_size:
            limit = min(int(self.config.sample_size), len(test_split))
            logger.info("Subsampling evaluation dataset to %s examples", limit)
            test_split = test_split.select(range(limit))

        self.tracker.start_run({
            "dataset_id": self.config.dataset_id,
            "model_path": str(self.config.model_path),
        })
        self.tracker.log_params({
            "evaluation_sample_size": len(test_split),
            "generation_kwargs": self.config.generation_kwargs,
        })

        try:
            rouge_scores = self._calculate_metrics(test_split, model, tokenizer, batch_size=batch_size)
            metrics_path = Path(self.config.metric_file_name)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame([rouge_scores])
            df.to_csv(metrics_path, index=False)
            with open(metrics_path.with_suffix(".json"), "w", encoding="utf-8") as file:
                json.dump({"dataset_id": self.config.dataset_id, "metrics": rouge_scores}, file, indent=2)
            logger.info("Evaluation metrics written to %s", metrics_path)
            self.tracker.log_metrics({f"evaluation/{k}": v for k, v in rouge_scores.items()})
            return rouge_scores
        finally:
            self.tracker.finish()
