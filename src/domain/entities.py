from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class Document:
    """Entity representing a document in vector store"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None

@dataclass
class TableColumn:
    """Entity representing a database table column"""
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    foreign_table: Optional[str] = None
    foreign_column: Optional[str] = None

@dataclass
class Table:
    """Entity representing a database table"""
    schema_name: str
    table_name: str
    columns: List[TableColumn]
    row_count: Optional[int] = None

@dataclass
class SearchResult:
    """Entity representing a search result"""
    document: Document
    score: float
    relevance: float

@dataclass
class ChatSession:
    """Entity representing a chat session"""
    session_id: str
    user_query: str
    context: List[Document]
    answer: str
    created_at: datetime
    processing_time: float