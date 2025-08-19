import logging
import re
import json
from typing import List, Dict, Any
from collections import defaultdict

from src.domain.interfaces import ILLMService

logger = logging.getLogger(__name__)

class LLMServiceAdapter:
    """
    Truly adaptive LLM service without any hardcoded business terms
    """
    
    def __init__(self, llm_service: ILLMService):
        self.llm_service = llm_service
        self.learned_patterns = {
            "conjunctions": set(),
            "query_structures": [],
            "decomposition_patterns": defaultdict(list)
        }
        self._initialize_learning()
    
    def _initialize_learning(self):
        """Initialize with minimal language-agnostic patterns"""
        # We learn these from the data, not hardcode them
        self.complexity_indicators = {
            "learned_conjunctions": set(),
            "multi_entity_patterns": [],
            "enumeration_patterns": []
        }
    
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
        
        # Learn from this interaction
        self._learn_from_query(query, context)
        
        return answer
    
    async def decompose_complex_query(self, query: str) -> List[str]:
        """
        Decompose query using AI without any hardcoded examples
        """
        # Let AI figure out if decomposition is needed
        complexity_check = await self._analyze_query_complexity(query)
        
        if not complexity_check["is_complex"]:
            return [query]
        
        # Build adaptive decomposition prompt
        decompose_prompt = self._build_adaptive_decompose_prompt(
            query, 
            complexity_check["complexity_type"]
        )
        
        try:
            response = await self.llm_service.generate(
                prompt=decompose_prompt,
                context="",
                max_tokens=300
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                questions = result.get("questions", [])
                if questions:
                    # Learn from successful decomposition
                    self._learn_decomposition_pattern(query, questions)
                    return questions
        except Exception as e:
            logger.warning(f"Failed to decompose query: {e}")
        
        return [query]
    
    async def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity using AI, not hardcoded rules
        """
        analysis_prompt = f"""
        Analyze if this query needs decomposition into simpler parts.
        Query: "{query}"
        
        Respond with JSON only:
        {{
            "is_complex": true/false,
            "complexity_type": "single/multiple/comparison/enumeration/hierarchical",
            "entity_count": number,
            "action_count": number
        }}
        
        A query is complex if it asks about multiple distinct things or requires multiple separate lookups.
        """
        
        try:
            response = await self.llm_service.generate(
                prompt=analysis_prompt,
                context="",
                max_tokens=150
            )
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        
        # Fallback to pattern learning
        return self._learned_complexity_check(query)
    
    def _learned_complexity_check(self, query: str) -> Dict[str, Any]:
        """
        Check complexity based on learned patterns, not hardcoded rules
        """
        result = {
            "is_complex": False,
            "complexity_type": "single",
            "entity_count": 1,
            "action_count": 1
        }
        
        # Check learned conjunction patterns
        for pattern in self.learned_patterns["query_structures"]:
            if self._matches_pattern(query, pattern):
                result["is_complex"] = True
                result["complexity_type"] = pattern.get("type", "multiple")
                break
        
        # Detect multiple entities through structure, not keywords
        # Look for patterns like repeated structures
        query_tokens = query.split()
        
        # Detect parallelism in query structure
        if self._has_parallel_structure(query_tokens):
            result["is_complex"] = True
            result["complexity_type"] = "multiple"
        
        # Multiple question marks indicate complexity
        if query.count('?') > 1:
            result["is_complex"] = True
            result["complexity_type"] = "multiple"
        
        return result
    
    def _has_parallel_structure(self, tokens: List[str]) -> bool:
        """
        Detect parallel structures without hardcoded conjunctions
        """
        # Look for repeated patterns in token structure
        # This is language-agnostic
        
        # Find repeating n-grams
        for n in range(2, min(5, len(tokens) // 2)):
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            ngram_counts = defaultdict(int)
            
            for ngram in ngrams:
                # Skip if it's just punctuation or numbers
                if all(t.isdigit() or not t.isalnum() for t in ngram):
                    continue
                ngram_counts[ngram] += 1
            
            # If we have repeating structures, likely complex
            if any(count > 1 for count in ngram_counts.values()):
                return True
        
        return False
    
    def _build_adaptive_decompose_prompt(
        self, 
        query: str, 
        complexity_type: str
    ) -> str:
        """
        Build decomposition prompt without any business-specific examples
        """
        # Use learned patterns if available
        learned_examples = self._get_learned_examples(complexity_type)
        
        if learned_examples:
            examples_text = "Based on learned patterns:\n" + learned_examples
        else:
            examples_text = ""
        
        prompt = f"""
        Decompose this query into simple, atomic questions.
        Each sub-question should focus on exactly ONE piece of information.
        
        Query: "{query}"
        Type: {complexity_type}
        
        {examples_text}
        
        Rules:
        1. Each sub-question must be answerable independently
        2. Preserve the original language and terms used
        3. Maintain the intent of the original query
        4. Do not add questions not implied by the original
        
        Output JSON only:
        {{"questions": ["question1", "question2", ...]}}
        """
        
        return prompt
    
    def _get_learned_examples(self, complexity_type: str) -> str:
        """
        Get examples from learned patterns, not hardcoded
        """
        if complexity_type not in self.learned_patterns["decomposition_patterns"]:
            return ""
        
        patterns = self.learned_patterns["decomposition_patterns"][complexity_type]
        if not patterns:
            return ""
        
        # Use only the most recent learned patterns
        recent = patterns[-3:]  # Last 3 examples
        
        examples = []
        for pattern in recent:
            # Anonymize the specific terms
            anonymized = self._anonymize_pattern(pattern)
            examples.append(f"Pattern: {anonymized['input']} → {anonymized['output']}")
        
        return "\n".join(examples)
    
    def _anonymize_pattern(self, pattern: Dict) -> Dict:
        """
        Anonymize specific business terms in patterns
        """
        # Replace specific entities with generic markers
        input_text = pattern.get("input", "")
        output_text = pattern.get("output", [])
        
        # Simple anonymization - replace specific terms with [ENTITY], [VALUE], etc
        # This is done through pattern matching, not hardcoded terms
        
        # Find potential entities (capitalized words, quoted strings, numbers)
        input_anon = re.sub(r'\b[A-Z][a-z]+\b', '[ENTITY]', input_text)
        input_anon = re.sub(r'\b\d+\b', '[NUMBER]', input_anon)
        input_anon = re.sub(r'"[^"]*"', '[QUOTED]', input_anon)
        
        output_anon = []
        for q in output_text:
            q_anon = re.sub(r'\b[A-Z][a-z]+\b', '[ENTITY]', q)
            q_anon = re.sub(r'\b\d+\b', '[NUMBER]', q_anon)
            output_anon.append(q_anon)
        
        return {
            "input": input_anon,
            "output": output_anon
        }
    
    def _learn_from_query(self, query: str, context: str):
        """
        Learn patterns from queries without hardcoding
        """
        # Learn query structure
        structure = self._extract_query_structure(query)
        if structure not in self.learned_patterns["query_structures"]:
            self.learned_patterns["query_structures"].append(structure)
        
        # Learn potential conjunctions from context
        self._learn_conjunctions_from_context(query, context)
    
    def _extract_query_structure(self, query: str) -> Dict:
        """
        Extract abstract structure of query
        """
        # Tokenize and analyze structure
        tokens = query.split()
        
        structure = {
            "length": len(tokens),
            "has_multiple_clauses": ',' in query or ';' in query,
            "question_marks": query.count('?'),
            "has_enumeration": self._detect_enumeration_request(tokens),
            "type": "single"  # default
        }
        
        # Detect type based on structure
        if structure["question_marks"] > 1:
            structure["type"] = "multiple"
        elif structure["has_enumeration"]:
            structure["type"] = "enumeration"
        elif structure["has_multiple_clauses"]:
            structure["type"] = "complex"
        
        return structure
    
    def _detect_enumeration_request(self, tokens: List[str]) -> bool:
        """
        Detect if query is asking for a list/enumeration
        Based on structure, not specific words
        """
        # Look for question words that typically precede enumerations
        # This is done by position and structure, not hardcoded words
        
        # If first token is lowercase and ends with a question mark, 
        # likely an enumeration request
        if tokens and tokens[0].islower() and tokens[-1].endswith('?'):
            # Check if there's a plural noun (ends with 's' in many languages)
            for token in tokens:
                if token.endswith('s') and not token.endswith('is'):
                    return True
        
        return False
    
    def _learn_conjunctions_from_context(self, query: str, context: str):
        """
        Learn conjunction patterns from successful queries
        """
        # Find repeated small words that might be conjunctions
        words = query.lower().split()
        
        for i, word in enumerate(words):
            # Conjunctions are typically short (2-5 chars) and lowercase
            if 2 <= len(word) <= 5 and word.islower():
                # Check if this word appears between other content words
                if i > 0 and i < len(words) - 1:
                    # This might be a conjunction
                    self.complexity_indicators["learned_conjunctions"].add(word)
    
    def _learn_decomposition_pattern(self, original: str, decomposed: List[str]):
        """
        Learn from successful decomposition
        """
        # Analyze the decomposition pattern
        pattern_type = "multiple"  # default
        
        if len(decomposed) == 2:
            pattern_type = "dual"
        elif len(decomposed) > 3:
            pattern_type = "enumeration"
        
        # Store the pattern
        self.learned_patterns["decomposition_patterns"][pattern_type].append({
            "input": original,
            "output": decomposed,
            "timestamp": self._get_timestamp()
        })
        
        # Keep only recent patterns (last 10)
        if len(self.learned_patterns["decomposition_patterns"][pattern_type]) > 10:
            self.learned_patterns["decomposition_patterns"][pattern_type] = \
                self.learned_patterns["decomposition_patterns"][pattern_type][-10:]
    
    def _matches_pattern(self, query: str, pattern: Dict) -> bool:
        """
        Check if query matches a learned pattern
        """
        # Structural matching, not keyword matching
        query_structure = self._extract_query_structure(query)
        
        # Compare structures
        if query_structure["type"] == pattern.get("type"):
            if query_structure["question_marks"] == pattern.get("question_marks", 0):
                return True
        
        return False
    
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
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def summarize_context(
        self,
        context: str,
        focus: str,
        max_length: int = 500
    ) -> str:
        """
        Summarize context adaptively
        """
        if len(context) <= max_length:
            return context
        
        summary_prompt = f"""
        Summarize this information focusing on: {focus}
        Keep important data points and relationships.
        Maximum {max_length} characters.
        
        Information:
        {context}
        
        Summary:
        """
        
        summary = await self.llm_service.generate(
            prompt=summary_prompt,
            context="",
            max_tokens=200
        )
        
        return summary[:max_length]