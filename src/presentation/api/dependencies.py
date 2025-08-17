import os
from functools import lru_cache

from src.infrastructure.database.postgres_repository import PostgresRepository
from src.infrastructure.vector_store.chroma_repository import ChromaRepository
from src.infrastructure.embedding.sentence_transformer import SentenceTransformerEmbedder
from src.infrastructure.llm.poe_client import PoeClient
from src.application.services.orchestrator_service import RAGOrchestrator
from src.application.use_cases.sync_data import SyncDataUseCase

# Cache instances for better performance
@lru_cache()
def get_postgres_repository() -> PostgresRepository:
    """Get PostgreSQL repository instance"""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL is not configured")
    return PostgresRepository(db_url)

@lru_cache()
def get_vector_store() -> ChromaRepository:
    """Get vector store instance"""
    path = os.getenv("VECTOR_DB_PATH", "./vector_db_data")
    return ChromaRepository(path)

@lru_cache()
def get_embedding_service() -> SentenceTransformerEmbedder:
    """Get embedding service instance"""
    model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformerEmbedder(model)

@lru_cache()
def get_llm_service() -> PoeClient:
    """Get LLM service instance"""
    api_key = os.getenv("POE_API_KEY")
    if not api_key:
        raise ValueError("POE_API_KEY is not configured")
    model = os.getenv("LLM_MODEL", "Claude-3-Haiku")
    return PoeClient(api_key, model)

def get_orchestrator() -> RAGOrchestrator:
    """Get RAG orchestrator instance"""
    return RAGOrchestrator(
        vector_store=get_vector_store(),
        llm_service=get_llm_service(),
        embedding_service=get_embedding_service()
    )

def get_sync_use_case() -> SyncDataUseCase:
    """Get sync data use case instance"""
    return SyncDataUseCase(
        db_repository=get_postgres_repository(),
        vector_store=get_vector_store(),
        embedding_service=get_embedding_service()
    )