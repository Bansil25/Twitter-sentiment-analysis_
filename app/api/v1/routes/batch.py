"""
Batch prediction endpoints.
- POST /batch/predict   → synchronous, up to 512 texts in JSON body
- POST /batch/upload    → async CSV upload, returns job_id for polling
- GET  /batch/jobs/{id} → poll job status and get download URL
"""

import io
import time
import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status

from app.core.security import CurrentUser
from app.schemas.sentiment import (
    BatchSentimentRequest,
    BatchSentimentResponse,
    CSVBatchJobResponse,
    CSVBatchStatus,
    ModelName,
    SentimentResponse,
)
from app.services.model_service import ModelService

log = structlog.get_logger()
router = APIRouter()

# In-memory job store — replace with Redis in production
_jobs: dict[str, dict] = {}


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


@router.post(
    "/batch/predict",
    response_model=BatchSentimentResponse,
    summary="Batch sentiment prediction (JSON, up to 512 texts)",
)
async def batch_predict(
    payload: BatchSentimentRequest,
    model_service: ModelServiceDep,
    current_user: CurrentUser,
) -> BatchSentimentResponse:
    """
    Submit a list of texts for bulk inference.
    More efficient than calling /predict N times.

    **Example:**
    ```json
    {
      "texts": ["Great product!", "Total waste of money.", "It arrived today."],
      "model": "distilbert"
    }
    ```
    """
    start = time.perf_counter()
    log.info("batch.request", count=len(payload.texts), model=payload.model, user=current_user.sub)

    try:
        results = await model_service.batch_predict(
            texts=payload.texts,
            model_name=payload.model,
        )
    except Exception as e:
        log.error("batch.failed", error=str(e))
        raise HTTPException(status_code=503, detail="Batch inference failed")

    total_ms = (time.perf_counter() - start) * 1000

    responses = [
        SentimentResponse(
            text=r["text"],
            sentiment=r["sentiment"],
            confidence=r["confidence"],
            probabilities=r["probabilities"],
            model_used=payload.model,
            processing_time_ms=round(total_ms / len(results), 2),
        )
        for r in results
    ]

    return BatchSentimentResponse(
        results=responses,
        total=len(responses),
        model_used=payload.model,
        total_processing_time_ms=round(total_ms, 2),
    )


@router.post(
    "/batch/upload",
    response_model=CSVBatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload CSV for async bulk prediction",
    description=(
        "Upload a CSV file with a 'text' column (max 10,000 rows). "
        "Returns a job_id immediately — poll /batch/jobs/{job_id} for results. "
        "Results are available as a downloadable CSV."
    ),
)
async def upload_csv(
    background_tasks: BackgroundTasks,
    model_service: ModelServiceDep,
    current_user: CurrentUser,
    file: UploadFile = File(..., description="CSV file with 'text' column"),
    model: ModelName = ModelName.DISTILBERT,
) -> CSVBatchJobResponse:
    # Validate file type
    if not file.filename.endswith(".csv"):  # type: ignore[union-attr]
        raise HTTPException(status_code=415, detail="Only CSV files are accepted")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": CSVBatchStatus.PENDING, "rows_total": None, "rows_processed": 0}

    background_tasks.add_task(
        _process_csv_job,
        job_id=job_id,
        content=content,
        model_service=model_service,
        model=model,
        user=current_user.sub,
    )

    log.info("batch.csv.accepted", job_id=job_id, filename=file.filename, user=current_user.sub)
    return CSVBatchJobResponse(
        job_id=job_id,
        status=CSVBatchStatus.PENDING,
        message="Job queued. Poll /batch/jobs/{job_id} for status.",
    )


@router.get(
    "/batch/jobs/{job_id}",
    response_model=CSVBatchJobResponse,
    summary="Poll CSV batch job status",
)
async def get_job_status(
    job_id: str,
    current_user: CurrentUser,
) -> CSVBatchJobResponse:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    return CSVBatchJobResponse(
        job_id=job_id,
        status=job["status"],
        message=job.get("message", ""),
        rows_total=job.get("rows_total"),
        rows_processed=job.get("rows_processed"),
        download_url=job.get("download_url"),
    )


# ── Background task ────────────────────────────────────────────────────────

async def _process_csv_job(
    job_id: str,
    content: bytes,
    model_service: ModelService,
    model: ModelName,
    user: str,
) -> None:
    """Runs in FastAPI's background task queue."""
    import csv

    log.info("batch.csv.start", job_id=job_id, user=user)
    _jobs[job_id]["status"] = CSVBatchStatus.PROCESSING

    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8", errors="replace")))
        rows = list(reader)

        if "text" not in (reader.fieldnames or []):
            _jobs[job_id].update(status=CSVBatchStatus.FAILED, message="CSV must have a 'text' column")
            return

        texts = [row["text"] for row in rows if row.get("text", "").strip()]
        _jobs[job_id]["rows_total"] = len(texts)

        # Process in chunks to avoid memory pressure
        CHUNK = 64
        results = []
        for i in range(0, len(texts), CHUNK):
            chunk_results = await model_service.batch_predict(texts[i:i+CHUNK], model)
            results.extend(chunk_results)
            _jobs[job_id]["rows_processed"] = min(i + CHUNK, len(texts))

        # Write output CSV to temp file (in production: upload to S3/GCS)
        import tempfile, os
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=["text", "sentiment", "confidence"])
        writer.writeheader()
        for r in results:
            writer.writerow({"text": r["text"], "sentiment": r["sentiment"], "confidence": round(r["confidence"], 4)})

        _jobs[job_id].update(
            status=CSVBatchStatus.COMPLETE,
            message=f"Processed {len(results)} rows",
            rows_processed=len(results),
            download_url=f"/api/v1/batch/jobs/{job_id}/download",
            _csv_content=out.getvalue(),
        )
        log.info("batch.csv.complete", job_id=job_id, rows=len(results))

    except Exception as e:
        log.error("batch.csv.error", job_id=job_id, error=str(e))
        _jobs[job_id].update(status=CSVBatchStatus.FAILED, message=str(e))
