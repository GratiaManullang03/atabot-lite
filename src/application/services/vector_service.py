from typing import List, Dict, Any
import logging
import hashlib
from datetime import datetime

from src.domain.interfaces import IVectorStore, IEmbeddingService
from src.domain.entities import Document, SearchResult

logger = logging.getLogger(__name__)

class VectorService:
    """
    Application service for vector store operations
    """
    
    def __init__(
        self,
        vector_store: IVectorStore,
        embedding_service: IEmbeddingService
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def index_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        text_field: str = "content",
        batch_size: int = 64
    ) -> int:
        """
        Index documents into vector store
        """
        if not documents:
            return 0
        
        # Prepare documents for indexing
        doc_objects = []
        texts_to_embed = []
        
        for doc in documents:
            # Extract text for embedding
            text = doc.get(text_field, "")
            if not text:
                logger.warning(f"Document missing text field '{text_field}'")
                continue
            
            texts_to_embed.append(text)
            
            # Create document object (without embedding yet)
            doc_id = doc.get("id") or hashlib.md5(text.encode()).hexdigest()
            doc_objects.append(Document(
                id=doc_id,
                content=text,
                metadata=doc,
                created_at=datetime.now()
            ))
        
        if not texts_to_embed:
            logger.warning("No valid documents to index")
            return 0
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts_to_embed)} documents")
        embeddings = []
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            batch_embeddings = await self.embedding_service.embed_batch(batch)
            embeddings.extend(batch_embeddings)
            
            logger.info(f"Processed {min(i + batch_size, len(texts_to_embed))}/{len(texts_to_embed)} embeddings")
        
        # Add embeddings to documents
        for doc, embedding in zip(doc_objects, embeddings):
            doc.embedding = embedding
        
        # Upsert to vector store
        await self.vector_store.upsert_documents(collection_name, doc_objects)
        
        logger.info(f"Successfully indexed {len(doc_objects)} documents to '{collection_name}'")
        return len(doc_objects)
    
    async def semantic_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Perform semantic search on a collection
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Search in vector store
        results = await self.vector_store.search(
            collection_name,
            query_embedding,
            top_k
        )
        
        # Filter by score threshold
        if score_threshold > 0:
            results = [r for r in results if r.score >= score_threshold]
        
        return results
    
    async def hybrid_search(
        self,
        collection_name: str,
        query: str,
        filters: Dict[str, Any] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Perform hybrid search (semantic + metadata filtering)
        """
        # First do semantic search
        results = await self.semantic_search(collection_name, query, top_k * 2)
        
        # Apply metadata filters if provided
        if filters:
            filtered_results = []
            for result in results:
                match = True
                for key, value in filters.items():
                    if result.document.metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_results.append(result)
            results = filtered_results
        
        # Return top_k results
        return results[:top_k]
    
    async def get_collection_stats(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Get statistics about a collection
        """
        try:
            # Try to get collection info
            # This is a simplified version - actual implementation would depend on vector store
            sample_results = await self.vector_store.search(
                collection_name,
                [0.0] * 384,  # Dummy embedding
                1
            )
            
            stats = {
                "collection": collection_name,
                "exists": True,
                "sample_document": None
            }
            
            if sample_results:
                stats["sample_document"] = {
                    "id": sample_results[0].document.id,
                    "content_preview": sample_results[0].document.content[:100] + "..."
                    if len(sample_results[0].document.content) > 100
                    else sample_results[0].document.content
                }
            
            return stats
            
        except Exception as e:
            return {
                "collection": collection_name,
                "exists": False,
                "error": str(e)
            }
    
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from vector store
        """
        try:
            await self.vector_store.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False