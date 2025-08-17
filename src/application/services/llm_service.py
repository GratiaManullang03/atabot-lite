import logging
import re

from src.domain.interfaces import ILLMService

logger = logging.getLogger(__name__)

class LLMServiceAdapter:
    """
    Application service adapter for LLM operations
    Provides additional prompt engineering and response processing
    """
    
    def __init__(self, llm_service: ILLMService):
        self.llm_service = llm_service
    
    async def generate_answer(
        self,
        query: str,
        context: str,
        max_tokens: int = 500,
        temperature: float = 0.1
    ) -> str:
        """
        Generate answer with enhanced prompt engineering
        """
        # Validate inputs
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Generate response
        answer = await self.llm_service.generate(
            prompt=query,
            context=context,
            max_tokens=max_tokens
        )
        
        # Post-process answer
        answer = self._clean_answer(answer)
        
        return answer
    
    async def decompose_complex_query(self, query: str) -> list[str]:
        """
        Decompose complex query into simpler sub-queries
        """
        # Check if decomposition is needed
        if not self._is_complex(query):
            return [query]
        
        decompose_prompt = f"""
        Pecah pertanyaan kompleks berikut menjadi pertanyaan-pertanyaan sederhana.
        Setiap pertanyaan harus fokus pada SATU item atau aspek saja.
        
        Format output HARUS JSON: {{"questions": ["pertanyaan1", "pertanyaan2", ...]}}
        
        Contoh:
        Input: "Berapa stok laptop dan mouse, serta lokasi penyimpanannya?"
        Output: {{"questions": ["Berapa stok laptop?", "Berapa stok mouse?", "Dimana lokasi penyimpanan laptop?", "Dimana lokasi penyimpanan mouse?"]}}
        
        Pertanyaan: "{query}"
        """
        
        try:
            response = await self.llm_service.generate(
                prompt=decompose_prompt,
                context="",
                max_tokens=300
            )
            
            # Extract JSON from response
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                questions = result.get("questions", [])
                if questions:
                    return questions
        except Exception as e:
            logger.warning(f"Failed to decompose query: {e}")
        
        return [query]
    
    async def summarize_context(
        self,
        context: str,
        focus: str,
        max_length: int = 500
    ) -> str:
        """
        Summarize context focusing on specific aspect
        """
        if len(context) <= max_length:
            return context
        
        summary_prompt = f"""
        Rangkum informasi berikut dengan fokus pada: {focus}
        Pertahankan data penting seperti angka, nama, dan fakta spesifik.
        Maksimal {max_length} karakter.
        
        Informasi:
        {context}
        
        Rangkuman:
        """
        
        summary = await self.llm_service.generate(
            prompt=summary_prompt,
            context="",
            max_tokens=200
        )
        
        return summary[:max_length]
    
    def _is_complex(self, query: str) -> bool:
        """Check if query is complex"""
        indicators = [
            ' dan ', ' atau ', ' serta ', ',', ';',
            'bandingkan', 'berapa masing-masing',
            'semuanya', 'daftar', 'list'
        ]
        query_lower = query.lower()
        return any(ind in query_lower for ind in indicators)
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and format answer"""
        # Remove excessive whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Remove incomplete sentences at the end
        if answer and not answer[-1] in '.!?':
            # Try to find last complete sentence
            last_period = answer.rfind('.')
            if last_period > 0:
                answer = answer[:last_period + 1]
        
        return answer