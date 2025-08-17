from fastapi import APIRouter
from typing import Dict, Any
import psutil
import os

from src.presentation.api.dependencies import (
    get_postgres_repository,
    get_vector_store,
    get_embedding_service,
    get_llm_service
)

router = APIRouter()

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check
    """
    health_status = {
        "status": "healthy",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "checks": {}
    }
    
    # Check database
    try:
        db = get_postgres_repository()
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        health_status["checks"]["database"] = "ok"
    except Exception as e:
        health_status["checks"]["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check vector store
    try:
        vector_store = get_vector_store()
        health_status["checks"]["vector_store"] = "ok"
    except Exception as e:
        health_status["checks"]["vector_store"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check embedding service
    try:
        embedder = get_embedding_service()
        health_status["checks"]["embedding_service"] = "ok"
    except Exception as e:
        health_status["checks"]["embedding_service"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # Check LLM service
    try:
        llm = get_llm_service()
        health_status["checks"]["llm_service"] = "ok"
    except Exception as e:
        health_status["checks"]["llm_service"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # System metrics
    health_status["metrics"] = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    return health_status

@router.get("/ready")
async def readiness_check():
    """
    Simple readiness check
    """
    return {"ready": True}

@router.get("/live")
async def liveness_check():
    """
    Simple liveness check
    """
    return {"alive": True}