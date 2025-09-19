from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.pipeline.prediction import PredictionPipeline
from textSummarizer.service.training_jobs import TrainingJobManager

app = FastAPI(title="Text Summarizer Service", version="2.0.0")
job_manager = TrainingJobManager(max_workers=1)


class TrainingRequest(BaseModel):
    dataset_id: Optional[str] = None
    hyperparameter_search: bool = False
    sweep_trials: Optional[int] = None
    experiment_name: Optional[str] = None


class TrainingResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    dataset_id: Optional[str]
    status: str
    submitted_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    error: Optional[str]
    experiment_name: Optional[str]


class PredictionRequest(BaseModel):
    text: str
    dataset_id: Optional[str] = None


class PredictionResponse(BaseModel):
    dataset_id: str
    summary: str


@app.on_event("startup")
async def startup_event() -> None:
    manager = ConfigurationManager()
    PredictionPipeline.warmup(manager.dataset_id)


@app.get("/", tags=["root"])
async def index() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/datasets")
async def list_datasets() -> Dict[str, object]:
    manager = ConfigurationManager()
    datasets = []
    for dataset_id in manager.available_datasets:
        dataset_cfg = manager.config.datasets[dataset_id]
        datasets.append(
            {
                "id": dataset_id,
                "description": dataset_cfg.get("description", ""),
                "ingestion_type": dataset_cfg.ingestion.type,
                "input_column": dataset_cfg.columns.input_text,
                "target_column": dataset_cfg.columns.target_text,
                "max_input_length": int(dataset_cfg.max_input_length),
                "max_target_length": int(dataset_cfg.max_target_length),
            }
        )
    return {"default": manager.dataset_id, "datasets": datasets}


@app.post("/train", response_model=TrainingResponse)
async def enqueue_training(request: TrainingRequest) -> TrainingResponse:
    try:
        manager = ConfigurationManager(dataset_id=request.dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = job_manager.submit(
        dataset_id=manager.dataset_id,
        enable_hyperparameter_search=request.hyperparameter_search,
        hyperparameter_trials=request.sweep_trials,
        experiment_name=request.experiment_name,
    )
    return TrainingResponse(job_id=job_id, status="queued")


@app.get("/train/{job_id}", response_model=JobStatusResponse)
async def training_status(job_id: str) -> JobStatusResponse:
    try:
        job = job_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found") from exc
    return JobStatusResponse(**job)


@app.get("/train", response_model=Dict[str, JobStatusResponse])
async def list_jobs() -> Dict[str, JobStatusResponse]:
    jobs = job_manager.list_jobs()
    return {job_id: JobStatusResponse(**info) for job_id, info in jobs.items()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required for prediction.")
    predictor = PredictionPipeline(dataset_id=request.dataset_id)
    summary = predictor.predict(request.text)
    return PredictionResponse(dataset_id=predictor.config.dataset_id, summary=summary)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
