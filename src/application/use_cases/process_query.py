from typing import List
import logging
import time
from datetime import datetime

from src.domain.interfaces import IVectorStore, ILLMService, IEmbeddingService
from src.domain.entities import SearchResult, ChatSession

logger = logging.getLogger(__name__)

class ProcessQueryUseCase:
    """
    Use case for processing user queries through RAG pipeline
    """
    
    def __init__(
        self,
        vector_store: IVectorStore,
        llm_service: ILLMService,
        embedding_service: IEmbeddingService
    ):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.embedding_service = embedding_service
    
    async def execute(
        self,
        query: str,
        collection_name: str,
        session_id: str = None,
        top_k: int = 3,
        min_score: float = 0.3
    ) -> ChatSession:
        """
        Execute the query processing pipeline
        """
        start_time = time.time()
        
        try:
            # Step 1: Generate query embedding
            logger.info(f"Processing query: {query[:100]}...")
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Step 2: Search for relevant documents
            search_results = await self.vector_store.search(
                collection_name,
                query_embedding,
                top_k
            )
            
            # Step 3: Filter by minimum score
            relevant_results = [
                r for r in search_results 
                if r.score >= min_score
            ]
            
            if not relevant_results:
                logger.warning(f"No relevant documents found for query: {query}")
                # Provide a default response when no data is found
                answer = "Maaf, saya tidak menemukan data yang relevan untuk menjawab pertanyaan Anda. Pastikan data sudah di-sync ke sistem."
                context_docs = []
            else:
                # Step 4: Prepare context from search results
                context = self._prepare_context(relevant_results)
                context_docs = [r.document for r in relevant_results]
                
                # Step 5: Generate answer using LLM
                answer = await self.llm_service.generate(
                    prompt=query,
                    context=context,
                    max_tokens=500
                )
                
                # Step 6: Validate answer
                answer = self._validate_answer(answer, query)
            
            # Step 7: Create chat session
            processing_time = time.time() - start_time
            
            session = ChatSession(
                session_id=session_id or self._generate_session_id(),
                user_query=query,
                context=context_docs,
                answer=answer,
                created_at=datetime.now(),
                processing_time=processing_time
            )
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return session
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    async def process_complex_query(
        self,
        query: str,
        collection_name: str,
        session_id: str = None
    ) -> ChatSession:
        """
        Process complex queries by decomposing them
        """
        start_time = time.time()
        
        # Check if query needs decomposition
        if not self._is_complex_query(query):
            return await self.execute(query, collection_name, session_id)
        
        logger.info("Detected complex query, decomposing...")
        
        # Decompose query
        sub_queries = await self._decompose_query(query)
        
        # Process each sub-query
        all_contexts = []
        all_answers = []
        
        for sub_query in sub_queries:
            session = await self.execute(
                sub_query,
                collection_name,
                session_id
            )
            all_contexts.extend(session.context)
            all_answers.append(session.answer)
        
        # Combine answers
        final_answer = self._combine_answers(all_answers, query)
        
        # Remove duplicate contexts
        unique_contexts = []
        seen_ids = set()
        for doc in all_contexts:
            if doc.id not in seen_ids:
                unique_contexts.append(doc)
                seen_ids.add(doc.id)
        
        processing_time = time.time() - start_time
        
        return ChatSession(
            session_id=session_id or self._generate_session_id(),
            user_query=query,
            context=unique_contexts,
            answer=final_answer,
            created_at=datetime.now(),
            processing_time=processing_time
        )
    
    def _prepare_context(self, search_results: List[SearchResult]) -> str:
        """
        Prepare context string from search results
        """
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            score = f"(relevance: {result.score:.2%})"
            content = result.document.content
            
            # Add metadata if available
            metadata_info = []
            if result.document.metadata:
                for key, value in result.document.metadata.items():
                    if not key.startswith('_'):
                        metadata_info.append(f"{key}: {value}")
            
            if metadata_info:
                metadata_str = ", ".join(metadata_info[:3])  # Limit metadata items
                context_parts.append(f"{i}. {content} [{metadata_str}] {score}")
            else:
                context_parts.append(f"{i}. {content} {score}")
        
        return "\n".join(context_parts)
    
    def _validate_answer(self, answer: str, query: str) -> str:
        """
        Validate and clean the generated answer
        """
        # Check if answer is too short or generic
        if len(answer) < 10 or answer.lower().strip() in ['ya', 'tidak', 'ok']:
            return f"Untuk pertanyaan '{query}': {answer}"
        
        # Check if answer seems incomplete
        if answer.endswith(('...', '..', '.')):
            return answer
        
        # Add period if missing
        if not answer[-1] in '.!?':
            answer += '.'
        
        return answer
    
    def _is_complex_query(self, query: str) -> bool:
        """
        Check if query is complex and needs decomposition
        """
        indicators = [
            ' dan ', ' serta ', ' atau ',
            'bandingkan', 'berapa masing-masing',
            'semuanya', 'semua', 'list', 'daftar'
        ]
        
        query_lower = query.lower()
        
        # Check for multiple question marks
        if query.count('?') > 1:
            return True
        
        # Check for conjunctions and list requests
        return any(indicator in query_lower for indicator in indicators)
    
    async def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler ones
        """
        # This would typically use the LLM to decompose
        # For now, using simple heuristics
        
        sub_queries = []
        
        # Split by conjunctions
        parts = query.replace(' dan ', '|').replace(' serta ', '|').split('|')
        
        for part in parts:
            part = part.strip()
            if part:
                # Ensure each part is a complete question
                if '?' not in part:
                    part += '?'
                sub_queries.append(part)
        
        return sub_queries if len(sub_queries) > 1 else [query]
    
    def _combine_answers(self, answers: List[str], original_query: str) -> str:
        """
        Combine multiple answers into a coherent response
        """
        if not answers:
            return "Tidak ada data yang ditemukan."
        
        if len(answers) == 1:
            return answers[0]
        
        # Remove duplicate answers
        unique_answers = []
        for answer in answers:
            if answer not in unique_answers:
                unique_answers.append(answer)
        
        # Combine with proper formatting
        combined = f"Berdasarkan pertanyaan '{original_query}', berikut informasinya:\n\n"
        
        for i, answer in enumerate(unique_answers, 1):
            if len(unique_answers) > 1:
                combined += f"{i}. {answer}\n"
            else:
                combined += answer
        
        return combined.strip()
    
    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID
        """
        import uuid
        return str(uuid.uuid4())