"""Analyzer Agent for SmartDoc Analyst.

The AnalyzerAgent performs deep analysis on retrieved documents,
including pattern detection, fact verification, and insight generation.
"""

from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState


class AnalyzerAgent(BaseAgent):
    """Deep analysis agent for document content analysis.
    
    The AnalyzerAgent is responsible for:
    - Pattern detection across documents
    - Fact verification and validation
    - Insight generation and key point extraction
    - Code execution for calculations and data analysis
    
    Attributes:
        code_executor: Tool for safe code execution.
        fact_checker: Tool for fact verification.
        
    Example:
        >>> analyzer = AnalyzerAgent(llm=my_llm)
        >>> result = await analyzer.process(context, {
        ...     "query": "Analyze trends",
        ...     "documents": retrieved_docs
        ... })
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        code_executor: Optional[Any] = None,
        fact_checker: Optional[Any] = None
    ):
        """Initialize the analyzer agent.
        
        Args:
            llm: Language model interface.
            code_executor: Tool for safe code execution.
            fact_checker: Tool for fact verification.
        """
        super().__init__(
            name="Analyzer",
            description="Deep analysis and insight generation",
            llm=llm
        )
        self.code_executor = code_executor
        self.fact_checker = fact_checker
        
    async def process(
        self,
        context: AgentContext,
        input_data: Any
    ) -> AgentResult:
        """Perform deep analysis on documents.
        
        Args:
            context: Task context with trace information.
            input_data: Dictionary with query and documents.
            
        Returns:
            AgentResult: Analysis results with insights.
        """
        self.set_state(AgentState.RUNNING)
        
        try:
            query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
            documents = input_data.get("documents", {}) if isinstance(input_data, dict) else {}
            
            analysis = {
                "key_insights": [],
                "patterns": [],
                "facts": [],
                "calculations": None,
                "summary": ""
            }
            
            # Extract key insights from documents
            doc_list = documents.get("documents", []) if isinstance(documents, dict) else []
            if doc_list:
                analysis["key_insights"] = await self._extract_insights(query, doc_list)
                
            # Detect patterns across documents
            analysis["patterns"] = await self._detect_patterns(doc_list)
            
            # Verify facts if checker available
            if self.fact_checker and analysis["key_insights"]:
                analysis["facts"] = await self._verify_facts(analysis["key_insights"], doc_list)
                
            # Execute calculations if needed
            if self._needs_calculation(query):
                analysis["calculations"] = await self._perform_calculations(query, doc_list)
                
            # Generate summary
            analysis["summary"] = self._generate_summary(analysis)
            
            self.set_state(AgentState.COMPLETED)
            self.add_to_history({
                "task_id": context.task_id,
                "query": query,
                "insights_found": len(analysis["key_insights"]),
                "patterns_found": len(analysis["patterns"])
            })
            
            return AgentResult(
                success=True,
                data=analysis,
                metrics={
                    "insights_extracted": len(analysis["key_insights"]),
                    "patterns_detected": len(analysis["patterns"]),
                    "facts_verified": len(analysis["facts"])
                }
            )
            
        except Exception as e:
            self.set_state(AgentState.ERROR)
            return AgentResult(
                success=False,
                error=f"Analysis failed: {str(e)}"
            )
            
    async def _extract_insights(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract key insights from documents.
        
        Args:
            query: Analysis query.
            documents: List of documents to analyze.
            
        Returns:
            List[Dict]: Extracted insights.
        """
        insights = []
        
        if self.llm:
            # Use LLM to extract insights
            prompt = f"""Analyze the following documents and extract key insights relevant to: {query}

Documents:
{self._format_documents(documents)}

Extract the most important insights as a list of key points."""

            try:
                response = await self.llm.generate(prompt)
                # Parse response into structured insights
                if response:
                    insights = self._parse_insights(response)
            except Exception:
                pass
                
        # Fallback: Basic extraction
        if not insights:
            for i, doc in enumerate(documents[:5]):
                content = doc.get("content", "")
                if content:
                    insights.append({
                        "id": f"insight_{i+1}",
                        "content": content[:200] + "..." if len(content) > 200 else content,
                        "source": doc.get("metadata", {}).get("source", "Unknown"),
                        "confidence": 0.7
                    })
                    
        return insights
        
    async def _detect_patterns(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect patterns across multiple documents.
        
        Args:
            documents: List of documents to analyze.
            
        Returns:
            List[Dict]: Detected patterns.
        """
        patterns = []
        
        # Extract common themes/topics
        all_content = " ".join(doc.get("content", "") for doc in documents)
        
        if self.llm and all_content:
            try:
                prompt = f"""Identify recurring patterns, themes, and common topics in the following content:

{all_content[:4000]}

List the main patterns found."""

                response = await self.llm.generate(prompt)
                if response:
                    patterns = self._parse_patterns(response)
            except Exception:
                pass
                
        return patterns
        
    async def _verify_facts(
        self,
        insights: List[Dict[str, Any]],
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Verify facts against source documents.
        
        Args:
            insights: Extracted insights to verify.
            documents: Source documents for verification.
            
        Returns:
            List[Dict]: Verified facts with confidence.
        """
        verified = []
        
        if self.fact_checker:
            for insight in insights:
                result = await self.fact_checker.verify(
                    claim=insight.get("content", ""),
                    sources=documents
                )
                verified.append({
                    "claim": insight.get("content", ""),
                    "verified": result.get("verified", False),
                    "confidence": result.get("confidence", 0.0),
                    "supporting_evidence": result.get("evidence", [])
                })
        else:
            # Basic verification - check if content appears in documents
            for insight in insights:
                content = insight.get("content", "").lower()
                found_in = sum(
                    1 for doc in documents
                    if content[:50].lower() in doc.get("content", "").lower()
                )
                verified.append({
                    "claim": insight.get("content", ""),
                    "verified": found_in > 0,
                    "confidence": min(found_in / max(len(documents), 1), 1.0),
                    "supporting_documents": found_in
                })
                
        return verified
        
    async def _perform_calculations(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Perform calculations or data analysis.
        
        Args:
            query: Query that may require calculation.
            documents: Documents with potential data.
            
        Returns:
            Optional[Dict]: Calculation results.
        """
        if not self.code_executor:
            return None
            
        # Extract numeric data from documents
        data = self._extract_numeric_data(documents)
        
        if data:
            try:
                result = await self.code_executor.execute(
                    code=self._generate_analysis_code(query, data),
                    timeout=30
                )
                return result
            except Exception:
                return None
                
        return None
        
    def _needs_calculation(self, query: str) -> bool:
        """Determine if query requires numerical calculation.
        
        Args:
            query: User query.
            
        Returns:
            bool: Whether calculation is needed.
        """
        calc_keywords = [
            "calculate", "compute", "average", "sum", "total",
            "percentage", "growth", "trend", "statistics", "numbers"
        ]
        return any(kw in query.lower() for kw in calc_keywords)
        
    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents for LLM prompt.
        
        Args:
            documents: List of documents.
            
        Returns:
            str: Formatted document text.
        """
        formatted = []
        for i, doc in enumerate(documents[:5], 1):
            content = doc.get("content", "")[:1000]
            source = doc.get("metadata", {}).get("source", "Unknown")
            formatted.append(f"[Document {i}] Source: {source}\n{content}")
        return "\n\n".join(formatted)
        
    def _parse_insights(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured insights.
        
        Args:
            response: LLM response text.
            
        Returns:
            List[Dict]: Parsed insights.
        """
        insights = []
        lines = response.strip().split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith("#"):
                # Remove list markers
                clean = line.lstrip("0123456789.-) ").strip()
                if clean:
                    insights.append({
                        "id": f"insight_{i+1}",
                        "content": clean,
                        "confidence": 0.8
                    })
        return insights[:10]  # Limit to 10 insights
        
    def _parse_patterns(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured patterns.
        
        Args:
            response: LLM response text.
            
        Returns:
            List[Dict]: Parsed patterns.
        """
        patterns = []
        lines = response.strip().split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.startswith("#"):
                clean = line.lstrip("0123456789.-) ").strip()
                if clean:
                    patterns.append({
                        "id": f"pattern_{i+1}",
                        "description": clean,
                        "frequency": "common"
                    })
        return patterns[:5]  # Limit to 5 patterns
        
    def _extract_numeric_data(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[float]:
        """Extract numeric data from documents.
        
        Args:
            documents: Source documents.
            
        Returns:
            List[float]: Extracted numbers.
        """
        import re
        numbers = []
        for doc in documents:
            content = doc.get("content", "")
            # Find numbers in content
            found = re.findall(r'\b\d+\.?\d*\b', content)
            numbers.extend(float(n) for n in found[:10])
        return numbers[:100]  # Limit data points
        
    def _generate_analysis_code(
        self,
        query: str,
        data: List[float]
    ) -> str:
        """Generate Python code for data analysis.
        
        Args:
            query: Analysis query.
            data: Numeric data to analyze.
            
        Returns:
            str: Python code for analysis.
        """
        return f"""
import statistics
data = {data}
result = {{
    'count': len(data),
    'sum': sum(data),
    'mean': statistics.mean(data) if data else 0,
    'median': statistics.median(data) if data else 0,
    'stdev': statistics.stdev(data) if len(data) > 1 else 0
}}
result
"""
        
    def _generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a summary of the analysis.
        
        Args:
            analysis: Complete analysis results.
            
        Returns:
            str: Analysis summary.
        """
        parts = []
        
        if analysis["key_insights"]:
            parts.append(f"Found {len(analysis['key_insights'])} key insights")
            
        if analysis["patterns"]:
            parts.append(f"Detected {len(analysis['patterns'])} patterns")
            
        if analysis["facts"]:
            verified = sum(1 for f in analysis["facts"] if f.get("verified"))
            parts.append(f"Verified {verified}/{len(analysis['facts'])} facts")
            
        if analysis["calculations"]:
            parts.append("Performed numerical analysis")
            
        return ". ".join(parts) + "." if parts else "Analysis complete."
        
    def get_capabilities(self) -> List[str]:
        """Return the analyzer's capabilities.
        
        Returns:
            List[str]: List of capabilities.
        """
        return [
            "insight_extraction",
            "pattern_detection",
            "fact_verification",
            "code_execution",
            "numerical_analysis",
            "trend_identification"
        ]
