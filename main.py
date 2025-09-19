import argparse

from textSummarizer.logging import logger
from textSummarizer.pipeline.training import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the text summarization training pipeline.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset identifier defined in config/config.yaml (defaults to config.default_dataset).",
    )
    parser.add_argument(
        "--hyperparameter-search",
        action="store_true",
        help="Enable hyperparameter search using the configured backend.",
    )
    parser.add_argument(
        "--sweep-trials",
        type=int,
        default=None,
        help="Override the number of hyperparameter search trials.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment/run name used for tracking artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("CLI arguments: %s", args)
    run_training_pipeline(
        dataset_id=args.dataset,
        enable_hyperparameter_search=args.hyperparameter_search,
        hyperparameter_trials=args.sweep_trials,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
