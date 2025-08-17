from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .entities import Document, Table, SearchResult

class IEmbeddingService(ABC):
    """Interface for embedding generation"""
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass

class IVectorStore(ABC):
    """Interface for vector store operations"""
    
    @abstractmethod
    async def upsert_documents(self, collection: str, documents: List[Document]) -> None:
        pass
    
    @abstractmethod
    async def search(self, collection: str, embedding: List[float], top_k: int) -> List[SearchResult]:
        pass
    
    @abstractmethod
    async def delete_collection(self, collection: str) -> None:
        pass

class IDatabaseRepository(ABC):
    """Interface for database operations"""
    
    @abstractmethod
    async def get_schemas(self) -> List[str]:
        pass
    
    @abstractmethod
    async def get_tables(self, schema: str) -> List[Table]:
        pass
    
    @abstractmethod
    async def get_table_data(self, schema: str, table: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        pass

class ILLMService(ABC):
    """Interface for LLM operations"""
    
    @abstractmethod
    async def generate(self, prompt: str, context: str, max_tokens: int = 500) -> str:
        pass