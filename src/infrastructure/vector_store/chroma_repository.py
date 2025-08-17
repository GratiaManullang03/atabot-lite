import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List
import logging
import hashlib

from src.domain.interfaces import IVectorStore
from src.domain.entities import Document, SearchResult

logger = logging.getLogger(__name__)

class ChromaRepository(IVectorStore):
    """
    Repository for ChromaDB vector store operations
    """
    
    def __init__(self, persist_path: str):
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        logger.info(f"Initialized ChromaDB at {persist_path}")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a collection"""
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
    
    async def upsert_documents(
        self, 
        collection: str, 
        documents: List[Document]
    ) -> None:
        """Upsert documents to collection"""
        if not documents:
            return
        
        try:
            coll = self._get_or_create_collection(collection)
            
            # Prepare data for ChromaDB
            ids = []
            contents = []
            metadatas = []
            embeddings = []
            
            for doc in documents:
                # Generate ID if not provided
                if not doc.id:
                    doc.id = hashlib.md5(doc.content.encode()).hexdigest()
                
                ids.append(doc.id)
                contents.append(doc.content)

                # Pastikan tidak ada nilai None di metadata
                clean_metadata = {k: v if v is not None else "" for k, v in (doc.metadata or {}).items()}
                metadatas.append(clean_metadata)

                if doc.embedding:
                    embeddings.append(doc.embedding)
            
            # Upsert to ChromaDB
            if embeddings:
                coll.upsert(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                coll.upsert(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas
                )
            
            logger.info(f"Upserted {len(documents)} documents to collection '{collection}'")
            
        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            raise
    
    async def search(
        self, 
        collection: str, 
        embedding: List[float], 
        top_k: int
    ) -> List[SearchResult]:
        """Search similar documents"""
        try:
            coll = self._get_or_create_collection(collection)
            
            # Check if collection is empty
            if coll.count() == 0:
                logger.warning(f"Collection '{collection}' is empty")
                return []
            
            # Perform search
            results = coll.query(
                query_embeddings=[embedding],
                n_results=min(top_k, coll.count()),
                include=['metadatas', 'documents', 'distances']
            )
            
            # Parse results
            search_results = []
            if results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    doc = Document(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] or {}
                    )
                    
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - results['distances'][0][i]
                    
                    search_results.append(SearchResult(
                        document=doc,
                        score=score,
                        relevance=score
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching collection '{collection}': {e}")
            raise
    
    async def delete_collection(self, collection: str) -> None:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection)
            logger.info(f"Deleted collection '{collection}'")
        except Exception as e:
            logger.warning(f"Could not delete collection '{collection}': {e}")