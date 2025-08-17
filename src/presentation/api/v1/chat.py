from fastapi import APIRouter, Depends, HTTPException

from src.presentation.models.request_models import ChatRequest
from src.presentation.models.response_models import ChatResponse
from src.presentation.api.dependencies import get_orchestrator
from src.application.services.orchestrator_service import RAGOrchestrator

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
):
    """
    Process user query and return AI-generated answer
    """
    try:
        answer, sources, processing_time = await orchestrator.process_query(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k
        )
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            processing_time=processing_time,
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(
    session_id: str,
    rating: int,
    comment: str = None
):
    """
    Submit feedback for a chat session
    """
    # TODO: Implement feedback storage
    return {"status": "Feedback received", "session_id": session_id}