"""Synthesizer Agent for SmartDoc Analyst.

The SynthesizerAgent generates structured reports and responses
by fusing information from multiple sources.
"""

from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState


class SynthesizerAgent(BaseAgent):
    """Report synthesis agent for generating structured outputs.
    
    The SynthesizerAgent is responsible for:
    - Multi-source information fusion
    - Structured report generation
    - Executive summary creation
    - Citation formatting
    
    Attributes:
        output_format: Default output format (markdown, json, plain).
        include_citations: Whether to include citations by default.
        
    Example:
        >>> synthesizer = SynthesizerAgent(llm=my_llm)
        >>> result = await synthesizer.process(context, {
        ...     "query": "Summarize findings",
        ...     "analysis": analysis_results
        ... })
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        output_format: str = "markdown",
        include_citations: bool = True
    ):
        """Initialize the synthesizer agent.
        
        Args:
            llm: Language model interface.
            output_format: Default output format.
            include_citations: Include citations in output.
        """
        super().__init__(
            name="Synthesizer",
            description="Report synthesis and output generation",
            llm=llm
        )
        self.output_format = output_format
        self.include_citations = include_citations
        
    async def process(
        self,
        context: AgentContext,
        input_data: Any
    ) -> AgentResult:
        """Synthesize a structured response from analysis results.
        
        Args:
            context: Task context with trace information.
            input_data: Dictionary with query and analysis results.
            
        Returns:
            AgentResult: Synthesized report.
        """
        self.set_state(AgentState.RUNNING)
        
        try:
            query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
            analysis = input_data.get("analysis", {}) if isinstance(input_data, dict) else {}
            feedback = input_data.get("feedback", []) if isinstance(input_data, dict) else []
            
            # Generate the main response
            response = await self._generate_response(query, analysis, feedback)
            
            # Format citations if available
            citations = self._format_citations(context.intermediate_results)
            
            # Structure the output
            output = {
                "response": response,
                "executive_summary": self._generate_executive_summary(response),
                "citations": citations if self.include_citations else [],
                "format": self.output_format,
                "word_count": len(response.split()),
                "sections": self._extract_sections(response)
            }
            
            self.set_state(AgentState.COMPLETED)
            self.add_to_history({
                "task_id": context.task_id,
                "query": query,
                "output_length": len(response)
            })
            
            return AgentResult(
                success=True,
                data=output,
                metrics={
                    "response_length": len(response),
                    "citation_count": len(citations),
                    "sections_generated": len(output["sections"])
                }
            )
            
        except Exception as e:
            self.set_state(AgentState.ERROR)
            return AgentResult(
                success=False,
                error=f"Synthesis failed: {str(e)}"
            )
            
    async def _generate_response(
        self,
        query: str,
        analysis: Dict[str, Any],
        feedback: List[str]
    ) -> str:
        """Generate the main response using LLM.
        
        Args:
            query: Original user query.
            analysis: Analysis results to synthesize.
            feedback: Feedback from critic for improvements.
            
        Returns:
            str: Generated response text.
        """
        if self.llm:
            prompt = self._build_synthesis_prompt(query, analysis, feedback)
            try:
                response = await self.llm.generate(prompt)
                return response if response else self._fallback_response(analysis)
            except Exception:
                return self._fallback_response(analysis)
        else:
            return self._fallback_response(analysis)
            
    def _build_synthesis_prompt(
        self,
        query: str,
        analysis: Dict[str, Any],
        feedback: List[str]
    ) -> str:
        """Build the synthesis prompt for the LLM.
        
        Args:
            query: Original query.
            analysis: Analysis results.
            feedback: Critic feedback.
            
        Returns:
            str: Synthesis prompt.
        """
        prompt_parts = [
            f"Generate a comprehensive response to: {query}",
            "",
            "Based on the following analysis:"
        ]
        
        # Add insights
        insights = analysis.get("key_insights", [])
        if insights:
            prompt_parts.append("\n## Key Insights:")
            for insight in insights[:5]:
                content = insight.get("content", str(insight)) if isinstance(insight, dict) else str(insight)
                prompt_parts.append(f"- {content}")
                
        # Add patterns
        patterns = analysis.get("patterns", [])
        if patterns:
            prompt_parts.append("\n## Patterns:")
            for pattern in patterns[:3]:
                desc = pattern.get("description", str(pattern)) if isinstance(pattern, dict) else str(pattern)
                prompt_parts.append(f"- {desc}")
                
        # Add summary
        summary = analysis.get("summary", "")
        if summary:
            prompt_parts.append(f"\n## Analysis Summary:\n{summary}")
            
        # Add feedback for improvement
        if feedback:
            prompt_parts.append("\n## Feedback to Address:")
            for fb in feedback:
                prompt_parts.append(f"- {fb}")
                
        prompt_parts.extend([
            "",
            "Generate a well-structured response in markdown format.",
            "Include an executive summary at the beginning.",
            "Organize the content with clear sections.",
            "Support claims with evidence from the analysis."
        ])
        
        return "\n".join(prompt_parts)
        
    def _fallback_response(self, analysis: Dict[str, Any]) -> str:
        """Generate a fallback response without LLM.
        
        Args:
            analysis: Analysis results.
            
        Returns:
            str: Fallback response.
        """
        parts = ["# Analysis Report", ""]
        
        # Executive Summary
        parts.append("## Executive Summary")
        summary = analysis.get("summary", "Analysis completed successfully.")
        parts.append(summary)
        parts.append("")
        
        # Key Findings
        insights = analysis.get("key_insights", [])
        if insights:
            parts.append("## Key Findings")
            for i, insight in enumerate(insights[:5], 1):
                content = insight.get("content", str(insight)) if isinstance(insight, dict) else str(insight)
                parts.append(f"{i}. {content}")
            parts.append("")
            
        # Patterns
        patterns = analysis.get("patterns", [])
        if patterns:
            parts.append("## Identified Patterns")
            for pattern in patterns[:3]:
                desc = pattern.get("description", str(pattern)) if isinstance(pattern, dict) else str(pattern)
                parts.append(f"- {desc}")
            parts.append("")
            
        # Facts
        facts = analysis.get("facts", [])
        if facts:
            parts.append("## Verified Facts")
            for fact in facts[:5]:
                claim = fact.get("claim", "") if isinstance(fact, dict) else str(fact)
                verified = fact.get("verified", False) if isinstance(fact, dict) else False
                status = "✓" if verified else "?"
                parts.append(f"- [{status}] {claim}")
            parts.append("")
            
        # Calculations
        calcs = analysis.get("calculations")
        if calcs:
            parts.append("## Numerical Analysis")
            if isinstance(calcs, dict):
                for key, value in calcs.items():
                    parts.append(f"- {key}: {value}")
            else:
                parts.append(str(calcs))
            parts.append("")
            
        return "\n".join(parts)
        
    def _generate_executive_summary(self, response: str) -> str:
        """Extract or generate an executive summary.
        
        Args:
            response: Full response text.
            
        Returns:
            str: Executive summary.
        """
        # Try to extract existing summary section
        lines = response.split("\n")
        in_summary = False
        summary_lines = []
        
        for line in lines:
            if "summary" in line.lower() and line.startswith("#"):
                in_summary = True
                continue
            elif in_summary and line.startswith("#"):
                break
            elif in_summary and line.strip():
                summary_lines.append(line)
                
        if summary_lines:
            return "\n".join(summary_lines[:5])
            
        # Fallback: First paragraph
        paragraphs = response.split("\n\n")
        for para in paragraphs:
            if para.strip() and not para.startswith("#"):
                return para.strip()[:500]
                
        return response[:300] if response else "No summary available."
        
    def _format_citations(
        self,
        intermediate_results: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Format citations from intermediate results.
        
        Args:
            intermediate_results: Results from other agents.
            
        Returns:
            List[Dict]: Formatted citations.
        """
        citations = []
        retrieved = intermediate_results.get("retrieved", {})
        
        if isinstance(retrieved, dict):
            raw_citations = retrieved.get("citations", [])
            for i, cit in enumerate(raw_citations, 1):
                if isinstance(cit, dict):
                    citations.append({
                        "number": i,
                        "source": cit.get("source", "Unknown"),
                        "title": cit.get("title", "Untitled"),
                        "type": cit.get("type", "document")
                    })
                    
        return citations
        
    def _extract_sections(self, response: str) -> List[Dict[str, str]]:
        """Extract sections from the response.
        
        Args:
            response: Full response text.
            
        Returns:
            List[Dict]: Extracted sections.
        """
        sections = []
        current_section = {"title": "Introduction", "content": []}
        
        for line in response.split("\n"):
            if line.startswith("##"):
                if current_section["content"]:
                    current_section["content"] = "\n".join(current_section["content"])
                    sections.append(current_section)
                current_section = {
                    "title": line.lstrip("#").strip(),
                    "content": []
                }
            elif line.startswith("#") and not line.startswith("##"):
                # Main title, skip
                continue
            else:
                current_section["content"].append(line)
                
        # Add last section
        if current_section["content"]:
            current_section["content"] = "\n".join(current_section["content"])
            sections.append(current_section)
            
        return sections
        
    def get_capabilities(self) -> List[str]:
        """Return the synthesizer's capabilities.
        
        Returns:
            List[str]: List of capabilities.
        """
        return [
            "report_generation",
            "executive_summary",
            "multi_source_fusion",
            "citation_formatting",
            "markdown_output",
            "section_organization"
        ]
