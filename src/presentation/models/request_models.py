from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    query: str = Field(..., description="User's question in natural language")
    collection_name: str = Field(..., description="Target collection in vector store")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of relevant documents to retrieve")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation tracking")

class SyncRequest(BaseModel):
    schema_name: str = Field(..., description="Database schema name")
    table_name: str = Field(..., description="Table name to sync")
    force_update: bool = Field(default=False, description="Force re-sync even if data exists")

class SchemaRequest(BaseModel):
    schema_name: str = Field(..., description="Database schema to inspect")