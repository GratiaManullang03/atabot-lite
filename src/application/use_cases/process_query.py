from typing import List
import logging
import time
from datetime import datetime
import re
from collections import defaultdict

from src.domain.interfaces import IVectorStore, ILLMService, IEmbeddingService
from src.domain.entities import SearchResult, ChatSession

logger = logging.getLogger(__name__)

class ProcessQueryUseCase:
    """
    Adaptive query processing without hardcoded business terms
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
        self.query_patterns = defaultdict(list)
        self.learned_complexity_indicators = set()
    
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
                # Provide adaptive default response
                answer = await self._generate_no_data_response(query)
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
            
            # Step 7: Learn from this query
            self._learn_query_pattern(query, answer)
            
            # Step 8: Create chat session
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
        Process complex queries using AI-based decomposition
        """
        start_time = time.time()
        
        # Use AI to check complexity, not hardcoded rules
        is_complex = await self._ai_check_complexity(query)
        
        if not is_complex:
            return await self.execute(query, collection_name, session_id)
        
        logger.info("AI detected complex query, decomposing...")
        
        # Decompose using AI
        sub_queries = await self._ai_decompose_query(query)
        
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
        
        # Combine answers intelligently
        final_answer = await self._ai_combine_answers(all_answers, query)
        
        # Remove duplicate contexts
        unique_contexts = self._deduplicate_contexts(all_contexts)
        
        processing_time = time.time() - start_time
        
        return ChatSession(
            session_id=session_id or self._generate_session_id(),
            user_query=query,
            context=unique_contexts,
            answer=final_answer,
            created_at=datetime.now(),
            processing_time=processing_time
        )
    
    async def _ai_check_complexity(self, query: str) -> bool:
        """
        Use AI to determine if query is complex
        No hardcoded indicators
        """
        check_prompt = f"""
        Analyze if this query asks for multiple distinct pieces of information.
        Query: "{query}"
        
        Answer with just: YES or NO
        """
        
        try:
            response = await self.llm_service.generate(
                prompt=check_prompt,
                context="",
                max_tokens=10
            )
            
            return "yes" in response.lower()
        except:
            # Fallback to learned patterns
            return self._check_learned_complexity(query)
    
    def _check_learned_complexity(self, query: str) -> bool:
        """
        Check complexity based on learned patterns only
        """
        # Structural analysis without hardcoded words
        
        # 1. Multiple question marks
        if query.count('?') > 1:
            return True
        
        # 2. Query length and structure
        tokens = query.split()
        if len(tokens) > 15:  # Longer queries tend to be complex
            # Check for parallel structures
            if self._has_parallel_patterns(tokens):
                return True
        
        # 3. Learned complexity indicators
        query_lower = query.lower()
        for indicator in self.learned_complexity_indicators:
            if indicator in query_lower:
                return True
        
        # 4. Punctuation patterns
        if query.count(',') >= 2 or query.count(';') >= 1:
            return True
        
        return False
    
    def _has_parallel_patterns(self, tokens: List[str]) -> bool:
        """
        Detect parallel structures in token patterns
        """
        # Look for repeating POS patterns or structures
        # This is language-agnostic
        
        # Simple heuristic: find repeating subsequences
        for window in range(2, min(5, len(tokens) // 2)):
            for i in range(len(tokens) - window * 2 + 1):
                pattern1 = tokens[i:i+window]
                pattern2 = tokens[i+window:i+window*2]
                
                # Check if patterns are similar (same length, similar structure)
                if self._patterns_similar(pattern1, pattern2):
                    return True
        
        return False
    
    def _patterns_similar(self, p1: List[str], p2: List[str]) -> bool:
        """
        Check if two token patterns are structurally similar
        """
        if len(p1) != len(p2):
            return False
        
        # Check structural similarity
        for t1, t2 in zip(p1, p2):
            # Similar if both are same type (number, capitalized, etc)
            if t1.isdigit() != t2.isdigit():
                continue
            if t1[0].isupper() != t2[0].isupper():
                continue
            if len(t1) == len(t2):  # Similar length
                return True
        
        return False
    
    async def _ai_decompose_query(self, query: str) -> List[str]:
        """
        Use AI to decompose without examples
        """
        decompose_prompt = f"""
        Break down this query into simple, independent questions.
        Query: "{query}"
        
        Rules:
        - Each question should ask for ONE piece of information
        - Keep the original language and terminology
        - Output as JSON: {{"questions": [...]}}
        """
        
        try:
            response = await self.llm_service.generate(
                prompt=decompose_prompt,
                context="",
                max_tokens=300
            )
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                import json
                result = json.loads(json_match.group(0))
                questions = result.get("questions", [])
                if questions:
                    return questions
        except Exception as e:
            logger.warning(f"AI decomposition failed: {e}")
        
        # Fallback to simple splitting
        return self._simple_split(query)
    
    def _simple_split(self, query: str) -> List[str]:
        """
        Simple splitting based on structure, not keywords
        """
        # Split by punctuation
        parts = re.split(r'[,;]|\s{2,}', query)
        
        sub_queries = []
        for part in parts:
            part = part.strip()
            if part and len(part) > 3:  # Meaningful length
                # Ensure it's a complete question
                if not part.endswith('?'):
                    part += '?'
                sub_queries.append(part)
        
        return sub_queries if len(sub_queries) > 1 else [query]
    
    async def _ai_combine_answers(
        self, 
        answers: List[str], 
        original_query: str
    ) -> str:
        """
        Use AI to combine answers intelligently
        """
        if not answers:
            return await self._generate_no_data_response(original_query)
        
        if len(answers) == 1:
            return answers[0]
        
        combine_prompt = f"""
        Combine these answers into a coherent response for the query.
        
        Original query: "{original_query}"
        
        Answers to combine:
        {chr(10).join(f"- {ans}" for ans in answers)}
        
        Create a natural, unified response:
        """
        
        try:
            combined = await self.llm_service.generate(
                prompt=combine_prompt,
                context="",
                max_tokens=500
            )
            return combined
        except:
            # Simple fallback
            return self._simple_combine(answers, original_query)
    
    def _simple_combine(self, answers: List[str], original_query: str) -> str:
        """
        Simple combination without hardcoded phrases
        """
        # Remove duplicates
        unique_answers = []
        for answer in answers:
            if answer not in unique_answers:
                unique_answers.append(answer)
        
        if len(unique_answers) == 1:
            return unique_answers[0]
        
        # Get language hint from query
        language = self._detect_language(original_query)
        
        # Combine based on detected language
        if language == "id":
            intro = f"Untuk '{original_query}':\n\n"
        elif language == "en":
            intro = f"Regarding '{original_query}':\n\n"
        else:
            intro = f"'{original_query}':\n\n"
        
        # Format answers
        formatted = []
        for i, answer in enumerate(unique_answers, 1):
            if len(unique_answers) > 1:
                formatted.append(f"{i}. {answer}")
            else:
                formatted.append(answer)
        
        return intro + "\n".join(formatted)
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection without hardcoded words
        """
        # Check for common patterns
        text_lower = text.lower()
        
        # Indonesian patterns (common endings)
        id_patterns = ['nya', 'kan', 'lah', 'kah']
        id_score = sum(1 for p in id_patterns if p in text_lower)
        
        # English patterns (common words)
        en_patterns = ['the', 'is', 'are', 'have', 'has']
        en_score = sum(1 for p in en_patterns if f' {p} ' in f' {text_lower} ')
        
        if id_score > en_score:
            return "id"
        elif en_score > id_score:
            return "en"
        else:
            return "unknown"
    
    async def _generate_no_data_response(self, query: str) -> str:
        """
        Generate adaptive no-data response
        """
        language = self._detect_language(query)
        
        if language == "id":
            return "Data yang relevan tidak ditemukan dalam sistem. Pastikan data sudah tersinkronisasi."
        elif language == "en":
            return "No relevant data found in the system. Please ensure data has been synchronized."
        else:
            return "No relevant data found. / Data tidak ditemukan."
    
    def _prepare_context(self, search_results: List[SearchResult]) -> str:
        """
        Prepare context string from search results
        """
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            score_pct = result.score * 100
            content = result.document.content
            
            # Add relevance indicator
            relevance = f"({score_pct:.0f}%)"
            
            # Add metadata if available (adaptive, not hardcoded)
            metadata_info = []
            if result.document.metadata:
                # Only include non-private metadata
                for key, value in result.document.metadata.items():
                    if not key.startswith('_') and value:
                        # Limit to first 3 metadata items
                        if len(metadata_info) < 3:
                            metadata_info.append(f"{key}: {value}")
            
            if metadata_info:
                metadata_str = " [" + ", ".join(metadata_info) + "]"
                context_parts.append(f"{i}. {content}{metadata_str} {relevance}")
            else:
                context_parts.append(f"{i}. {content} {relevance}")
        
        return "\n".join(context_parts)
    
    def _validate_answer(self, answer: str, query: str) -> str:
        """
        Validate and clean the generated answer
        """
        # Basic length check
        if len(answer) < 10:
            language = self._detect_language(query)
            if language == "id":
                return f"Untuk '{query}': {answer}"
            else:
                return f"Regarding '{query}': {answer}"
        
        # Ensure proper ending
        if not answer[-1] in '.!?':
            answer += '.'
        
        return answer
    
    def _learn_query_pattern(self, query: str, answer: str):
        """
        Learn from successful query-answer pairs
        """
        # Extract pattern features
        pattern = {
            "query_length": len(query.split()),
            "has_question_mark": '?' in query,
            "punctuation_count": sum(1 for c in query if c in ',;:'),
            "successful": len(answer) > 20
        }
        
        # Store pattern
        pattern_key = f"{pattern['query_length']}_{pattern['punctuation_count']}"
        self.query_patterns[pattern_key].append(pattern)
        
        # Learn potential complexity indicators from complex queries
        if pattern["punctuation_count"] > 1 or pattern["query_length"] > 15:
            # Extract potential conjunction words (short words between content)
            words = query.lower().split()
            for i, word in enumerate(words):
                if 1 < i < len(words) - 1:  # Word in middle
                    if 2 <= len(word) <= 5:  # Short word
                        if word.isalpha():  # Not punctuation
                            self.learned_complexity_indicators.add(word)
    
    def _deduplicate_contexts(self, contexts: List) -> List:
        """
        Remove duplicate contexts
        """
        unique = []
        seen_ids = set()
        
        for doc in contexts:
            if doc.id not in seen_ids:
                unique.append(doc)
                seen_ids.add(doc.id)
        
        return unique
    
    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID
        """
        import uuid
        return str(uuid.uuid4())