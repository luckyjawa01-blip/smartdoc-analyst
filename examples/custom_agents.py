#!/usr/bin/env python
"""Custom agents example for SmartDoc Analyst.

This script demonstrates how to create and integrate custom agents
with the SmartDoc Analyst system.
"""

import asyncio
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentState
from src.agents.orchestrator import OrchestratorAgent
from src.core.system import SmartDocAnalyst


class DomainExpertAgent(BaseAgent):
    """Custom domain expert agent for specialized analysis.
    
    This agent demonstrates how to create a custom specialized
    agent that can be integrated into the SmartDoc Analyst system.
    
    Attributes:
        domain: The domain of expertise (e.g., "healthcare", "finance").
        expertise_keywords: Keywords related to the domain.
    """
    
    def __init__(
        self,
        domain: str,
        expertise_keywords: List[str],
        llm: Any = None
    ):
        """Initialize the domain expert agent.
        
        Args:
            domain: Domain of expertise.
            expertise_keywords: Domain-specific keywords.
            llm: Language model interface.
        """
        super().__init__(
            name=f"DomainExpert_{domain.title()}",
            description=f"Specialized expert in {domain} domain analysis",
            llm=llm
        )
        self.domain = domain
        self.expertise_keywords = expertise_keywords
        
    async def process(
        self,
        context: AgentContext,
        input_data: Any
    ) -> AgentResult:
        """Process a query with domain expertise.
        
        Args:
            context: Task context.
            input_data: Input query and documents.
            
        Returns:
            AgentResult: Domain-specific analysis.
        """
        self.set_state(AgentState.RUNNING)
        
        try:
            query = input_data.get("query", "") if isinstance(input_data, dict) else str(input_data)
            documents = input_data.get("documents", []) if isinstance(input_data, dict) else []
            
            # Check if query is relevant to this domain
            relevance_score = self._calculate_domain_relevance(query)
            
            # Perform domain-specific analysis
            analysis = {
                "domain": self.domain,
                "relevance_score": relevance_score,
                "query": query,
                "domain_insights": self._extract_domain_insights(query, documents),
                "recommendations": self._generate_recommendations(query),
                "keywords_matched": [
                    kw for kw in self.expertise_keywords
                    if kw.lower() in query.lower()
                ]
            }
            
            self.set_state(AgentState.COMPLETED)
            
            return AgentResult(
                success=True,
                data=analysis,
                metrics={"relevance_score": relevance_score}
            )
            
        except Exception as e:
            self.set_state(AgentState.ERROR)
            return AgentResult(
                success=False,
                error=f"Domain analysis failed: {str(e)}"
            )
            
    def _calculate_domain_relevance(self, query: str) -> float:
        """Calculate how relevant the query is to this domain."""
        query_lower = query.lower()
        matches = sum(1 for kw in self.expertise_keywords if kw.lower() in query_lower)
        return min(1.0, matches / max(len(self.expertise_keywords) * 0.3, 1))
        
    def _extract_domain_insights(self, query: str, documents: List[Dict]) -> List[str]:
        """Extract domain-specific insights."""
        return [
            f"Analysis from {self.domain} perspective: Query relates to domain concepts",
            f"Domain relevance indicators: {', '.join(self.expertise_keywords[:3])}",
            f"Documents analyzed: {len(documents)} source(s)"
        ]
        
    def _generate_recommendations(self, query: str) -> List[str]:
        """Generate domain-specific recommendations."""
        return [
            f"Consider {self.domain}-specific regulations and standards",
            f"Review industry best practices in {self.domain}",
            f"Consult domain experts for validation"
        ]
        
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return [
            "domain_analysis",
            "relevance_scoring",
            "domain_insights",
            f"{self.domain}_expertise"
        ]


class SentimentAnalyzerAgent(BaseAgent):
    """Custom sentiment analysis agent.
    
    Demonstrates creating an agent for sentiment analysis
    of documents and queries.
    """
    
    def __init__(self, llm: Any = None):
        """Initialize sentiment analyzer."""
        super().__init__(
            name="SentimentAnalyzer",
            description="Analyzes sentiment and tone of content",
            llm=llm
        )
        
        # Simple sentiment indicators (in production, use ML model)
        self.positive_words = ["good", "great", "excellent", "positive", "growth", "success", "improvement"]
        self.negative_words = ["bad", "poor", "negative", "decline", "failure", "risk", "problem"]
        
    async def process(
        self,
        context: AgentContext,
        input_data: Any
    ) -> AgentResult:
        """Analyze sentiment of content.
        
        Args:
            context: Task context.
            input_data: Text content to analyze.
            
        Returns:
            AgentResult: Sentiment analysis results.
        """
        self.set_state(AgentState.RUNNING)
        
        try:
            text = input_data.get("text", "") if isinstance(input_data, dict) else str(input_data)
            
            # Simple sentiment calculation
            text_lower = text.lower()
            positive_count = sum(1 for word in self.positive_words if word in text_lower)
            negative_count = sum(1 for word in self.negative_words if word in text_lower)
            
            total = positive_count + negative_count
            if total == 0:
                sentiment_score = 0.0
                sentiment = "neutral"
            else:
                sentiment_score = (positive_count - negative_count) / total
                if sentiment_score > 0.2:
                    sentiment = "positive"
                elif sentiment_score < -0.2:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
            
            analysis = {
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "positive_indicators": positive_count,
                "negative_indicators": negative_count,
                "confidence": min(total / 10, 1.0)  # Simple confidence measure
            }
            
            self.set_state(AgentState.COMPLETED)
            
            return AgentResult(
                success=True,
                data=analysis
            )
            
        except Exception as e:
            self.set_state(AgentState.ERROR)
            return AgentResult(
                success=False,
                error=f"Sentiment analysis failed: {str(e)}"
            )
            
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities."""
        return ["sentiment_analysis", "tone_detection"]


async def main():
    """Run custom agents demonstration."""
    print("=" * 70)
    print("SmartDoc Analyst - Custom Agents Example")
    print("=" * 70)
    print()
    
    # Create custom agents
    print("Creating custom agents...")
    
    healthcare_expert = DomainExpertAgent(
        domain="healthcare",
        expertise_keywords=[
            "medical", "patient", "diagnosis", "treatment",
            "healthcare", "clinical", "disease", "therapy"
        ]
    )
    print(f"✓ Created: {healthcare_expert.name}")
    
    finance_expert = DomainExpertAgent(
        domain="finance",
        expertise_keywords=[
            "market", "investment", "trading", "portfolio",
            "financial", "stock", "bond", "asset"
        ]
    )
    print(f"✓ Created: {finance_expert.name}")
    
    sentiment_agent = SentimentAnalyzerAgent()
    print(f"✓ Created: {sentiment_agent.name}")
    print()
    
    # Test domain expert with healthcare query
    print("=" * 70)
    print("Testing Domain Expert Agent (Healthcare)")
    print("=" * 70)
    
    context = AgentContext(query="AI diagnosis in medical imaging")
    result = await healthcare_expert.process(
        context,
        {
            "query": "How is AI improving medical diagnosis accuracy?",
            "documents": [{"content": "AI in healthcare is growing rapidly."}]
        }
    )
    
    print(f"Query: How is AI improving medical diagnosis accuracy?")
    print(f"Relevance Score: {result.data['relevance_score']:.2f}")
    print(f"Domain: {result.data['domain']}")
    print(f"Keywords Matched: {result.data['keywords_matched']}")
    print("Insights:")
    for insight in result.data['domain_insights']:
        print(f"  - {insight}")
    print()
    
    # Test sentiment analysis
    print("=" * 70)
    print("Testing Sentiment Analyzer Agent")
    print("=" * 70)
    
    test_texts = [
        "The company reported excellent growth and great success this quarter.",
        "The project faced serious problems and risk of failure.",
        "The quarterly report contains financial data and market analysis."
    ]
    
    for text in test_texts:
        result = await sentiment_agent.process(
            AgentContext(),
            {"text": text}
        )
        
        print(f"\nText: {text[:60]}...")
        print(f"Sentiment: {result.data['sentiment']} (score: {result.data['sentiment_score']:.2f})")
        print(f"Confidence: {result.data['confidence']:.2f}")
    
    print()
    
    # Integrate custom agents with orchestrator
    print("=" * 70)
    print("Integrating Custom Agents with Orchestrator")
    print("=" * 70)
    print()
    
    # Create orchestrator and register custom agents
    orchestrator = OrchestratorAgent()
    orchestrator.register_agents(
        healthcare_expert=healthcare_expert,
        finance_expert=finance_expert,
        sentiment=sentiment_agent
    )
    
    print(f"Registered agents: {list(orchestrator.agents.keys())}")
    print()
    
    # Show agent capabilities
    print("Agent Capabilities:")
    for name, agent in orchestrator.agents.items():
        caps = agent.get_capabilities()
        print(f"  {name}: {', '.join(caps)}")
    
    print()
    print("=" * 70)
    print("Custom agents demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
