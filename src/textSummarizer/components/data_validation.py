from pathlib import Path
from typing import List

from textSummarizer.entity import DataValidationConfig
from textSummarizer.logging import logger


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def _record_status(self, status: bool, missing: List[str]) -> None:
        status_message = f"Validation status: {status}"
        if missing:
            status_message += f" | missing splits: {', '.join(missing)}"
        self.config.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.status_file, "w", encoding="utf-8") as file:
            file.write(status_message)
        logger.info(status_message)

    def validate_all_files_exist(self) -> bool:
        dataset_root = Path(self.config.data_path)
        missing_splits: List[str] = []
        for required_split in self.config.required_splits:
            split_name = self.config.splits.get(required_split, required_split)
            split_path = dataset_root / split_name
            if not split_path.exists():
                logger.warning("Missing expected split '%s' at %s", split_name, split_path)
                missing_splits.append(split_name)
        status = not missing_splits
        self._record_status(status, missing_splits)
        return status
