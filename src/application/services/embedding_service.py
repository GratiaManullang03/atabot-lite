from typing import List
import logging

from src.domain.interfaces import IEmbeddingService

logger = logging.getLogger(__name__)

class EmbeddingServiceAdapter:
    """
    Application service adapter for embedding operations
    Provides additional business logic on top of infrastructure embedding service
    """
    
    def __init__(self, embedding_service: IEmbeddingService):
        self.embedding_service = embedding_service
    
    async def generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with validation
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Truncate very long texts to prevent memory issues
        max_length = 5000
        if len(text) > max_length:
            logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        return await self.embedding_service.embed_text(text)
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str],
        batch_size: int = 64
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
        
        # Process in batches if needed
        if len(valid_texts) <= batch_size:
            return await self.embedding_service.embed_batch(valid_texts)
        
        # Process large batches
        all_embeddings = []
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            batch_embeddings = await self.embedding_service.embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            if i % (batch_size * 5) == 0:
                logger.info(f"Processed {i}/{len(valid_texts)} embeddings")
        
        return all_embeddings
    
    async def generate_schema_embedding(
        self,
        schema_description: str,
        table_name: str
    ) -> List[float]:
        """
        Generate specialized embedding for database schema
        """
        # Enhance schema description for better semantic search
        enhanced_text = f"""
        Table: {table_name}
        {schema_description}
        
        This data can be used to answer questions about {table_name}.
        """
        
        return await self.generate_text_embedding(enhanced_text)