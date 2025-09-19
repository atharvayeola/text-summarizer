# Text Summarizer Project

## Overview
This repository implements an end-to-end abstractive text summarization system. It now supports:

- **Config-driven datasets** – plug in multiple summarization corpora with per-dataset schema mappings and preprocessing hooks.
- **Reproducible ML pipeline** – ingestion, validation, transformation, training, and evaluation stages orchestrated through a single entry point.
- **Experiment tracking & sweeps** – optional Optuna-based hyperparameter search and pluggable tracking backends (local JSONL logs by default, with optional Weights & Biases support).
- **Async API service** – FastAPI application with cached inference pipelines and background training jobs with status reporting.

## Repository highlights
- `config/config.yaml` – global settings, dataset registry, and generation defaults.
- `params.yaml` – training arguments, hyperparameter search space, and tracking preferences.
- `src/textSummarizer/components/*` – implementation of each pipeline stage.
- `src/textSummarizer/pipeline/training.py` – orchestrates all stages for a selected dataset.
- `app.py` – FastAPI service exposing dataset discovery, async training, and cached prediction endpoints.

## Setup
```bash
# clone the repo
git clone https://github.com/atharvayeola/text-summarizer
cd text-summarizer

# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Configuring datasets
`config/config.yaml` ships with `samsum` and `cnn_dailymail` entries. Each dataset defines:

- `ingestion.type` – `zip` for remote archives or `huggingface` for datasets downloaded via the 🤗 `datasets` hub.
- `columns` – mapping between canonical `input_text`/`target_text` keys and dataset-specific fields.
- `splits` – logical split names (`train`, `validation`, `test`).
- `max_input_length` / `max_target_length` – sequence lengths used during tokenization and evaluation.

Add new datasets by copying an entry and adjusting these fields. Optional `preprocessing_fn` values can point to custom cleaning functions using dotted import paths.

## Running the pipeline from the CLI
```
python main.py \
  --dataset samsum \
  --hyperparameter-search \
  --sweep-trials 5 \
  --experiment-name samsum-baseline
```
Arguments:
- `--dataset` – dataset identifier from `config/config.yaml` (defaults to `default_dataset`).
- `--hyperparameter-search` – enable Optuna sweeps using the search space defined in `params.yaml`.
- `--sweep-trials` – optionally override the number of trials.
- `--experiment-name` – tag the run for tracking artifacts and logs.

Artifacts are written under `artifacts/<stage>/<dataset_id>/`, including tokenized datasets, fine-tuned models, and evaluation metrics (`metrics.csv` + JSON).

## Experiment tracking
Tracking settings live in `params.yaml` under `experiment_tracking`. By default the `local` backend writes JSONL logs to `artifacts/experiments/<dataset>/<run_name>/`. Set `backend: wandb` (and install `wandb`) to stream metrics to Weights & Biases.

Hyperparameter search is powered by Hugging Face `Trainer.hyperparameter_search` with an Optuna backend. The best configuration is logged to `hyperparameter_search.json` and used to retrain the final model automatically.

## FastAPI service
```
uvicorn app:app --reload
```
Key endpoints:
- `GET /datasets` – discover available datasets and their metadata.
- `POST /train` – enqueue a background training job. Body fields mirror the CLI flags.
- `GET /train/{job_id}` – retrieve job status (`queued`, `running`, `completed`, `failed`).
- `GET /train` – list all submitted jobs.
- `POST /predict` – summarize text using the cached model (`{"text": "...", "dataset_id": "samsum"}`).

The service warms up the default dataset model on startup and caches pipelines per dataset for low-latency inference.

## Evaluation pipeline
The evaluation stage reloads the fine-tuned model, generates summaries with the configured decoding settings, computes ROUGE metrics, and logs the results to both CSV and JSON artifacts. Metrics are also pushed to the active experiment tracker.

## AWS CI/CD (existing instructions)
Refer to the original deployment notes below for provisioning AWS infrastructure, building Docker images, and wiring up GitHub Actions self-hosted runners.

### AWS CI/CD Deployment with GitHub Actions

1. **Login to AWS console.**
2. **Create IAM user for deployment with specific access:**
   - EC2 access: Virtual machine
   - ECR: Elastic Container Registry to save your Docker image in AWS

   **Deployment flow:**
   1. Build Docker image of the source code
   2. Push your Docker image to ECR
   3. Launch your EC2
   4. Pull your image from ECR in EC2
   5. Launch your Docker image in EC2

   **Required policies:**
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonEC2FullAccess`

3. **Create ECR repository to store/save Docker image** and note the repository URI.
4. **Create an EC2 machine (Ubuntu)** and install Docker:
   ```bash
   sudo apt-get update -y
   sudo apt-get upgrade
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```
5. **Configure the EC2 instance as a self-hosted GitHub Actions runner.** Navigate to `Settings → Actions → Runners` in your GitHub repository and follow the prompts.
6. **Set up GitHub secrets** used by the CI/CD workflow:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION` (e.g., `us-east-1`)
   - `AWS_ECR_LOGIN_URI`
   - `ECR_REPOSITORY_NAME`
