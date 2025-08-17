from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
import uuid

from src.presentation.models.request_models import SyncRequest
from src.presentation.models.response_models import SyncResponse
from src.presentation.api.dependencies import get_sync_use_case
from src.application.use_cases.sync_data import SyncDataUseCase

router = APIRouter()

# Store for tracking sync jobs
sync_jobs = {}

async def run_sync_task(
    job_id: str,
    schema_name: str,
    table_name: str,
    sync_use_case: SyncDataUseCase
):
    """Background task for data sync"""
    try:
        sync_jobs[job_id] = {"status": "running"}
        result = await sync_use_case.sync_table(schema_name, table_name)
        sync_jobs[job_id] = {"status": "completed", **result}
    except Exception as e:
        sync_jobs[job_id] = {"status": "failed", "error": str(e)}

@router.post("/", response_model=SyncResponse)
async def sync_table(
    request: SyncRequest,
    background_tasks: BackgroundTasks,
    sync_use_case: SyncDataUseCase = Depends(get_sync_use_case)
):
    """
    Sync a database table to vector store
    """
    job_id = str(uuid.uuid4())
    
    # Start background task
    background_tasks.add_task(
        run_sync_task,
        job_id,
        request.schema_name,
        request.table_name,
        sync_use_case
    )
    
    collection_name = f"{request.schema_name}_{request.table_name}"
    
    return SyncResponse(
        status="started",
        processed_items=0,
        collection_name=collection_name,
        duration=0,
        job_id=job_id
    )

@router.get("/status/{job_id}")
async def get_sync_status(job_id: str):
    """
    Get status of a sync job
    """
    if job_id not in sync_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return sync_jobs[job_id]