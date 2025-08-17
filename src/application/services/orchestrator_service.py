import json
import re
import time
from typing import List, Dict, Any, Tuple
import logging

from src.domain.interfaces import IVectorStore, ILLMService, IEmbeddingService
from src.domain.entities import SearchResult

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    """
    Service untuk mengorkestrasi RAG pipeline
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
    
    async def process_query(
        self,
        query: str,
        collection_name: str,
        top_k: int = 3
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Process user query through RAG pipeline
        """
        start_time = time.time()
        
        try:
            # Decompose complex queries if needed
            sub_queries = await self._decompose_query(query)
            
            all_answers = []
            all_sources = {}
            
            for sub_query in sub_queries:
                # Get embedding for the query
                query_embedding = await self.embedding_service.embed_text(sub_query)
                
                # Search relevant documents
                search_results = await self.vector_store.search(
                    collection_name,
                    query_embedding,
                    top_k
                )
                
                # Format context from search results
                context = self._format_context(search_results)
                
                # Generate answer using LLM
                answer = await self.llm_service.generate(
                    prompt=sub_query,
                    context=context
                )
                
                all_answers.append(answer)
                
                # Collect unique sources
                for result in search_results:
                    doc_id = result.document.id
                    if doc_id not in all_sources:
                        all_sources[doc_id] = result.document.metadata
            
            # Combine answers
            final_answer = self._combine_answers(all_answers)
            sources = list(all_sources.values())
            processing_time = time.time() - start_time
            
            return final_answer, sources, processing_time
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    async def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler sub-queries
        """
        # Check if query needs decomposition
        if not self._is_complex_query(query):
            return [query]
        
        decompose_prompt = f"""
        Analisis pertanyaan berikut dan pecah menjadi pertanyaan-pertanyaan sederhana.
        Setiap pertanyaan harus fokus pada satu item atau topik.
        Jawab HANYA dengan format JSON: {{"questions": ["q1", "q2", ...]}}
        
        Pertanyaan: "{query}"
        """
        
        try:
            response = await self.llm_service.generate(
                prompt=decompose_prompt,
                context="",
                max_tokens=200
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                questions = result.get("questions", [])
                return questions if questions else [query]
        except Exception as e:
            logger.warning(f"Failed to decompose query: {e}")
        
        return [query]
    
    def _is_complex_query(self, query: str) -> bool:
        """
        Check if query is complex and needs decomposition
        """
        # Simple heuristic: check for multiple items or conjunctions
        indicators = [' dan ', ' atau ', ' serta ', ',', ';', 'bandingkan', 'berapa masing-masing']
        return any(indicator in query.lower() for indicator in indicators)
    
    def _format_context(self, search_results: List[SearchResult]) -> str:
        """
        Format search results into context string
        """
        if not search_results:
            return "Tidak ada data relevan yang ditemukan."
        
        context_parts = []
        for result in search_results:
            content = result.document.content
            score = f"(relevance: {result.score:.2f})"
            context_parts.append(f"- {content} {score}")
        
        return "\n".join(context_parts)
    
    def _combine_answers(self, answers: List[str]) -> str:
        """
        Combine multiple answers into final response
        """
        if len(answers) == 1:
            return answers[0]
        
        # Remove duplicate information and combine
        unique_answers = []
        for answer in answers:
            if answer not in unique_answers:
                unique_answers.append(answer)
        
        return "\n\n".join(unique_answers)