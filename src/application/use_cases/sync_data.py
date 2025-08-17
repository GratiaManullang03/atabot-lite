import hashlib
from typing import List, Dict, Any
import logging
from datetime import datetime

from src.domain.entities import Document
from src.domain.interfaces import (
    IDatabaseRepository,
    IVectorStore,
    IEmbeddingService
)

logger = logging.getLogger(__name__)

class SyncDataUseCase:
    """
    Use case for syncing database data to vector store
    """
    
    def __init__(
        self,
        db_repository: IDatabaseRepository,
        vector_store: IVectorStore,
        embedding_service: IEmbeddingService
    ):
        self.db_repository = db_repository
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def sync_table(
        self,
        schema_name: str,
        table_name: str,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """
        Sync a database table to vector store
        """
        start_time = datetime.now()
        collection_name = f"{schema_name}_{table_name}"
        
        try:
            logger.info(f"Starting sync for {schema_name}.{table_name}")
            
            # Get table data
            table_data = await self.db_repository.get_table_data(
                schema_name,
                table_name
            )
            
            if not table_data:
                logger.warning(f"No data found in {schema_name}.{table_name}")
                return {
                    "status": "completed",
                    "processed_items": 0,
                    "collection_name": collection_name,
                    "duration": 0
                }
            
            # Prepare documents
            documents = await self._prepare_documents(
                table_data,
                schema_name,
                table_name,
                batch_size
            )
            
            # Upsert to vector store
            await self.vector_store.upsert_documents(
                collection_name,
                documents
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Sync completed for {collection_name}: "
                f"{len(documents)} documents in {duration:.2f}s"
            )
            
            return {
                "status": "completed",
                "processed_items": len(documents),
                "collection_name": collection_name,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Error syncing {schema_name}.{table_name}: {e}")
            raise
    
    async def _prepare_documents(
        self,
        table_data: List[Dict[str, Any]],
        schema_name: str,
        table_name: str,
        batch_size: int
    ) -> List[Document]:
        """
        Prepare documents with embeddings
        """
        documents = []
        
        # Find primary key column
        primary_key = self._find_primary_key(table_data[0])
        
        # Prepare texts for embedding
        texts = []
        metadatas = []
        ids = []
        
        for row in table_data:
            # Create searchable text from row data
            text = self._create_searchable_text(row, table_name)
            texts.append(text)
            
            # Prepare metadata
            metadata = {
                **row,
                "_schema": schema_name,
                "_table": table_name
            }
            metadatas.append(metadata)
            
            # Generate document ID
            doc_id = row.get(primary_key, "")
            if not doc_id:
                doc_id = hashlib.md5(text.encode()).hexdigest()
            else:
                doc_id = hashlib.md5(str(doc_id).encode()).hexdigest()
            ids.append(doc_id)
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = await self.embedding_service.embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            if i % (batch_size * 5) == 0:
                logger.info(f"Processed {i}/{len(texts)} embeddings...")
        
        # Create Document objects
        for i in range(len(texts)):
            documents.append(Document(
                id=ids[i],
                content=texts[i],
                metadata=metadatas[i],
                embedding=all_embeddings[i],
                created_at=datetime.now()
            ))
        
        return documents
    
    def _find_primary_key(self, row: Dict[str, Any]) -> str:
        """Find primary key column"""
        # Common primary key patterns
        for key in row.keys():
            if key.lower() in ['id', 'uuid', 'guid']:
                return key
            if key.lower().endswith('_id'):
                return key
        
        # Default to first column
        return list(row.keys())[0] if row else 'id'
    
    def _create_searchable_text(
        self,
        row: Dict[str, Any],
        table_name: str
    ) -> str:
        """Create searchable text from row data"""
        parts = [f"Data dari tabel {table_name}:"]
        
        for key, value in row.items():
            if value is not None:
                # Skip metadata fields
                if key.startswith('_'):
                    continue
                
                # Format field name
                field_name = key.replace('_', ' ').title()
                
                # Add to searchable text
                parts.append(f"{field_name}: {value}")
        
        return ". ".join(parts)