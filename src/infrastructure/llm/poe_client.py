import openai
import tiktoken
import logging

from src.domain.interfaces import ILLMService

logger = logging.getLogger(__name__)

class PoeClient(ILLMService):
    """
    LLM service using Poe API (OpenAI compatible)
    """
    
    def __init__(self, api_key: str, model: str = "Claude-3-Haiku"):
        if not api_key:
            raise ValueError("POE_API_KEY is required")
        
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.poe.com/v1"
        )
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = tiktoken.get_encoding("gpt2")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build optimized prompt for the LLM"""
        # Limit context length to save tokens
        if len(context) > 3000:
            context = context[:3000] + "..."
        
        prompt = f"""Anda adalah Atabot, asisten bisnis cerdas yang membantu menjawab pertanyaan berdasarkan data yang tersedia.

KONTEKS DATA:
{context}

PERTANYAAN:
{query}

INSTRUKSI:
1. Jawab HANYA berdasarkan data konteks yang tersedia
2. Jika data spesifik tersedia, sebutkan angka atau detail dengan tepat
3. Jika informasi tidak tersedia dalam konteks, katakan "Data yang diminta tidak tersedia dalam sistem"
4. Gunakan bahasa Indonesia yang jelas dan profesional
5. Berikan jawaban yang langsung dan informatif

JAWABAN:"""
        
        return prompt
    
    async def generate(
        self, 
        prompt: str, 
        context: str, 
        max_tokens: int = 500
    ) -> str:
        """Generate response from LLM"""
        try:
            # Build full prompt
            full_prompt = self._build_prompt(prompt, context)
            
            # Log token usage
            input_tokens = self._count_tokens(full_prompt)
            logger.debug(f"Input tokens: {input_tokens}")
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Anda adalah Atabot, asisten AI untuk bisnis yang memberikan jawaban akurat berdasarkan data yang tersedia."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for factual responses
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            # Log output tokens
            output_tokens = self._count_tokens(answer)
            logger.debug(f"Output tokens: {output_tokens}, Total: {input_tokens + output_tokens}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise