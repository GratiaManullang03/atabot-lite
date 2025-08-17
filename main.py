import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import logging

from src.presentation.api.v1 import chat, schema, sync, health
from src.infrastructure.database.postgres_repository import PostgresRepository
from src.infrastructure.vector_store.chroma_repository import ChromaRepository
from src.infrastructure.embedding.sentence_transformer import SentenceTransformerEmbedder

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO if not os.getenv("DEBUG", "False").lower() == "true" else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Atabot Lite...")
    
    # Initialize infrastructure components
    try:
        # Test database connection
        db_repo = PostgresRepository(os.getenv("DATABASE_URL"))
        with db_repo.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        logger.info("Database connection established")
        
        # Initialize vector store
        vector_repo = ChromaRepository(os.getenv("VECTOR_DB_PATH", "./vector_db_data"))
        logger.info("Vector store initialized")
        
        # Load embedding model
        embedder = SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        logger.info("Embedding model loaded")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down Atabot Lite...")

# Create FastAPI application
app = FastAPI(
    title=os.getenv("APP_NAME", "Atabot Lite"),
    description="Asisten Bisnis Cerdas - Unified Service",
    version=os.getenv("APP_VERSION", "1.0.0"),
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(schema.router, prefix="/api/v1/schema", tags=["Schema"])
app.include_router(sync.router, prefix="/api/v1/sync", tags=["Data Sync"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "Atabot Lite",
        "version": "1.0.0",
        "description": "Asisten Bisnis Cerdas di Ujung Jari Anda",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )