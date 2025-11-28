"""Critic Agent for SmartDoc Analyst.

The CriticAgent provides quality assurance by evaluating responses
and detecting potential issues like hallucinations or inconsistencies.
"""

from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState


class CriticAgent(BaseAgent):
    """Quality assurance agent for response validation.
    
    The CriticAgent is responsible for:
    - Response quality scoring
    - Hallucination detection
    - Consistency checking
    - Improvement suggestions
    
    Attributes:
        quality_threshold: Minimum quality score to pass (0-1).
        check_hallucinations: Whether to check for hallucinations.
        
    Example:
        >>> critic = CriticAgent(llm=my_llm)
        >>> result = await critic.process(context, {
        ...     "query": "Original question",
        ...     "response": synthesized_response
        ... })
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        quality_threshold: float = 0.7,
        check_hallucinations: bool = True
    ):
        """Initialize the critic agent.
        
        Args:
            llm: Language model interface.
            quality_threshold: Minimum quality score (0-1).
            check_hallucinations: Enable hallucination detection.
        """
        super().__init__(
            name="Critic",
            description="Quality assurance and validation",
            llm=llm
        )
        self.quality_threshold = quality_threshold
        self.check_hallucinations = check_hallucinations
        
    async def process(
        self,
        context: AgentContext,
        input_data: Any
    ) -> AgentResult:
        """Evaluate the quality of a response.
        
        Args:
            context: Task context with trace information.
            input_data: Dictionary with query and response.
            
        Returns:
            AgentResult: Quality evaluation results.
        """
        self.set_state(AgentState.RUNNING)
        
        try:
            query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
            response = input_data.get("response", {}) if isinstance(input_data, dict) else {}
            
            # Get response text
            response_text = self._extract_response_text(response)
            
            # Evaluate quality dimensions
            evaluation = {
                "completeness": await self._check_completeness(query, response_text),
                "relevance": await self._check_relevance(query, response_text),
                "coherence": await self._check_coherence(response_text),
                "accuracy": await self._check_accuracy(response_text, context),
                "hallucinations": await self._check_hallucinations_impl(response_text, context) if self.check_hallucinations else [],
            }
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(evaluation)
            evaluation["score"] = overall_score
            
            # Determine if improvement is needed
            needs_improvement = overall_score < self.quality_threshold
            evaluation["needs_improvement"] = needs_improvement
            
            # Generate suggestions if needed
            suggestions = []
            if needs_improvement:
                suggestions = self._generate_suggestions(evaluation)
            evaluation["suggestions"] = suggestions
            
            self.set_state(AgentState.COMPLETED)
            self.add_to_history({
                "task_id": context.task_id,
                "quality_score": overall_score,
                "passed": not needs_improvement
            })
            
            return AgentResult(
                success=True,
                data=evaluation,
                suggestions=suggestions,
                metrics={
                    "quality_score": overall_score,
                    "completeness": evaluation["completeness"],
                    "relevance": evaluation["relevance"],
                    "coherence": evaluation["coherence"]
                }
            )
            
        except Exception as e:
            self.set_state(AgentState.ERROR)
            return AgentResult(
                success=False,
                error=f"Critique failed: {str(e)}"
            )
            
    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from response.
        
        Args:
            response: Response object or string.
            
        Returns:
            str: Response text.
        """
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return response.get("response", response.get("content", str(response)))
        else:
            return str(response)
            
    async def _check_completeness(self, query: str, response: str) -> float:
        """Check if response fully addresses the query.
        
        Args:
            query: Original query.
            response: Response text.
            
        Returns:
            float: Completeness score (0-1).
        """
        if not response:
            return 0.0
            
        # Basic heuristics
        score = 0.0
        
        # Check minimum length
        word_count = len(response.split())
        if word_count >= 50:
            score += 0.3
        elif word_count >= 20:
            score += 0.2
        elif word_count >= 10:
            score += 0.1
            
        # Check for structure (sections, lists)
        if "##" in response or "- " in response:
            score += 0.2
            
        # Check for query keywords in response
        query_words = set(query.lower().split())
        response_lower = response.lower()
        keyword_coverage = sum(1 for w in query_words if w in response_lower)
        if query_words:
            score += 0.3 * (keyword_coverage / len(query_words))
            
        # Check for conclusion/summary
        if "conclusion" in response_lower or "summary" in response_lower:
            score += 0.2
            
        return min(score, 1.0)
        
    async def _check_relevance(self, query: str, response: str) -> float:
        """Check relevance of response to query.
        
        Args:
            query: Original query.
            response: Response text.
            
        Returns:
            float: Relevance score (0-1).
        """
        if not query or not response:
            return 0.0
            
        # Extract key terms from query
        query_terms = set(
            word.lower().strip("?.,!") 
            for word in query.split() 
            if len(word) > 3
        )
        
        # Check overlap with response
        response_lower = response.lower()
        matched = sum(1 for term in query_terms if term in response_lower)
        
        if not query_terms:
            return 0.5
            
        relevance = matched / len(query_terms)
        
        # Use LLM for more sophisticated check if available
        if self.llm and relevance < 0.8:
            try:
                prompt = f"""Rate how relevant this response is to the query.
                
Query: {query}

Response: {response[:1000]}

Respond with a single number from 0 to 10."""
                
                llm_response = await self.llm.generate(prompt)
                if llm_response:
                    # Parse number from response
                    import re
                    numbers = re.findall(r'\b(\d+)\b', llm_response)
                    if numbers:
                        llm_score = min(int(numbers[0]) / 10, 1.0)
                        # Average with heuristic score
                        relevance = (relevance + llm_score) / 2
            except Exception:
                pass
                
        return min(relevance, 1.0)
        
    async def _check_coherence(self, response: str) -> float:
        """Check structural coherence of response.
        
        Args:
            response: Response text.
            
        Returns:
            float: Coherence score (0-1).
        """
        if not response:
            return 0.0
            
        score = 0.5  # Base score
        
        # Check for proper sentence structure
        sentences = response.split(".")
        if len(sentences) >= 3:
            score += 0.1
            
        # Check for paragraph structure
        paragraphs = [p for p in response.split("\n\n") if p.strip()]
        if len(paragraphs) >= 2:
            score += 0.1
            
        # Check for logical connectors
        connectors = ["therefore", "however", "moreover", "additionally", 
                      "furthermore", "in conclusion", "as a result"]
        connector_count = sum(1 for c in connectors if c in response.lower())
        score += min(connector_count * 0.05, 0.2)
        
        # Check for consistent formatting
        has_headers = response.count("##") > 0
        has_lists = response.count("- ") > 0 or response.count("1.") > 0
        if has_headers or has_lists:
            score += 0.1
            
        return min(score, 1.0)
        
    async def _check_accuracy(
        self,
        response: str,
        context: AgentContext
    ) -> float:
        """Check accuracy against source information.
        
        Args:
            response: Response text.
            context: Task context with sources.
            
        Returns:
            float: Accuracy score (0-1).
        """
        # Get source documents from context
        retrieved = context.intermediate_results.get("retrieved", {})
        if isinstance(retrieved, dict):
            documents = retrieved.get("documents", [])
        else:
            documents = []
            
        if not documents:
            return 0.7  # Default score if no sources
            
        # Check if response content aligns with sources
        source_content = " ".join(
            doc.get("content", "")[:500] for doc in documents[:3]
        ).lower()
        
        response_sentences = [s.strip() for s in response.split(".") if s.strip()]
        
        if not response_sentences:
            return 0.5
            
        # Simple overlap check
        aligned = 0
        for sentence in response_sentences[:10]:
            # Check if key phrases from sentence appear in sources
            words = sentence.lower().split()
            key_words = [w for w in words if len(w) > 4]
            if key_words:
                matches = sum(1 for w in key_words if w in source_content)
                if matches / len(key_words) > 0.3:
                    aligned += 1
                    
        return min(aligned / max(len(response_sentences[:10]), 1), 1.0)
        
    async def _check_hallucinations_impl(
        self,
        response: str,
        context: AgentContext
    ) -> List[Dict[str, Any]]:
        """Detect potential hallucinations in response.
        
        Args:
            response: Response text.
            context: Task context with sources.
            
        Returns:
            List[Dict]: Detected hallucinations.
        """
        hallucinations = []
        
        # Get source content
        retrieved = context.intermediate_results.get("retrieved", {})
        if isinstance(retrieved, dict):
            documents = retrieved.get("documents", [])
        else:
            documents = []
            
        source_content = " ".join(
            doc.get("content", "") for doc in documents
        ).lower()
        
        # Check for specific claims that might be hallucinated
        # Look for numbers, dates, percentages not in sources
        import re
        
        # Check percentages
        percentages = re.findall(r'\b(\d+(?:\.\d+)?%)', response)
        for pct in percentages:
            if pct not in source_content:
                hallucinations.append({
                    "type": "unsupported_statistic",
                    "content": pct,
                    "severity": "medium"
                })
                
        # Check specific dates/years
        years = re.findall(r'\b(19|20)\d{2}\b', response)
        for year in years:
            if year not in source_content and len(documents) > 0:
                hallucinations.append({
                    "type": "unsupported_date",
                    "content": year,
                    "severity": "low"
                })
                
        # Check for strong claims
        strong_claims = [
            "proven", "guaranteed", "always", "never",
            "definitely", "certainly", "absolutely"
        ]
        for claim in strong_claims:
            if claim in response.lower():
                hallucinations.append({
                    "type": "strong_claim",
                    "content": claim,
                    "severity": "low"
                })
                
        return hallucinations[:10]  # Limit results
        
    def _calculate_overall_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate overall quality score.
        
        Args:
            evaluation: Individual dimension scores.
            
        Returns:
            float: Overall quality score (0-1).
        """
        weights = {
            "completeness": 0.25,
            "relevance": 0.30,
            "coherence": 0.20,
            "accuracy": 0.25
        }
        
        score = sum(
            evaluation.get(dim, 0.5) * weight
            for dim, weight in weights.items()
        )
        
        # Penalize for hallucinations
        hallucinations = evaluation.get("hallucinations", [])
        penalty = len(hallucinations) * 0.05
        
        return max(score - penalty, 0.0)
        
    def _generate_suggestions(
        self,
        evaluation: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement suggestions based on evaluation.
        
        Args:
            evaluation: Quality evaluation results.
            
        Returns:
            List[str]: Improvement suggestions.
        """
        suggestions = []
        
        if evaluation.get("completeness", 1) < 0.6:
            suggestions.append("Expand response to more fully address the query")
            
        if evaluation.get("relevance", 1) < 0.6:
            suggestions.append("Focus more directly on the query topic")
            
        if evaluation.get("coherence", 1) < 0.6:
            suggestions.append("Improve logical flow and structure")
            
        if evaluation.get("accuracy", 1) < 0.6:
            suggestions.append("Verify claims against source documents")
            
        hallucinations = evaluation.get("hallucinations", [])
        if hallucinations:
            suggestions.append(f"Review {len(hallucinations)} potential unsupported claims")
            
        return suggestions
        
    def get_capabilities(self) -> List[str]:
        """Return the critic's capabilities.
        
        Returns:
            List[str]: List of capabilities.
        """
        return [
            "quality_scoring",
            "hallucination_detection",
            "consistency_checking",
            "completeness_evaluation",
            "relevance_assessment",
            "improvement_suggestions"
        ]
