from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatResponse(BaseModel):
    answer: str = Field(..., description="AI-generated answer")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents used")
    processing_time: float = Field(..., description="Time taken to process query in seconds")
    session_id: Optional[str] = Field(default=None, description="Session ID for tracking")

class SyncResponse(BaseModel):
    status: str = Field(..., description="Sync operation status")
    processed_items: int = Field(..., description="Number of items processed")
    collection_name: str = Field(..., description="Vector store collection name")
    duration: float = Field(..., description="Sync duration in seconds")
    job_id: str = Field(..., description="ID of the sync job")

class SchemaResponse(BaseModel):
    schema_name: str
    tables: List[Dict[str, Any]]
    total_tables: int