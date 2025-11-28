"""Tests for SmartDoc Analyst agents."""

import pytest
import asyncio
from src.agents import (
    BaseAgent,
    AgentState,
    OrchestratorAgent,
    RetrieverAgent,
    AnalyzerAgent,
    SynthesizerAgent,
    CriticAgent,
    PlannerAgent
)
from src.agents.base_agent import AgentContext, AgentResult


class TestAgentState:
    """Test AgentState enum."""
    
    def test_agent_states_exist(self):
        """Test all agent states are defined."""
        assert AgentState.IDLE.value == "idle"
        assert AgentState.RUNNING.value == "running"
        assert AgentState.COMPLETED.value == "completed"
        assert AgentState.ERROR.value == "error"


class TestAgentContext:
    """Test AgentContext class."""
    
    def test_context_creation(self):
        """Test context can be created with defaults."""
        context = AgentContext()
        assert context.task_id is not None
        assert context.trace_id is not None
        assert context.query == ""
        assert context.intermediate_results == {}
        
    def test_context_with_values(self):
        """Test context with custom values."""
        context = AgentContext(
            query="Test query",
            metadata={"key": "value"}
        )
        assert context.query == "Test query"
        assert context.metadata["key"] == "value"


class TestAgentResult:
    """Test AgentResult class."""
    
    def test_successful_result(self):
        """Test creating successful result."""
        result = AgentResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
        
    def test_failed_result(self):
        """Test creating failed result."""
        result = AgentResult(success=False, error="Test error")
        assert result.success is False
        assert result.error == "Test error"


class TestPlannerAgent:
    """Test PlannerAgent."""
    
    def test_planner_creation(self):
        """Test planner agent can be created."""
        planner = PlannerAgent()
        assert planner.name == "Planner"
        assert planner.state == AgentState.IDLE
        
    def test_planner_capabilities(self):
        """Test planner has expected capabilities."""
        planner = PlannerAgent()
        capabilities = planner.get_capabilities()
        assert "query_decomposition" in capabilities
        assert "complexity_analysis" in capabilities
        
    @pytest.mark.asyncio
    async def test_planner_process_simple_query(self):
        """Test planner processes simple query."""
        planner = PlannerAgent()
        context = AgentContext(query="What is AI?")
        
        result = await planner.process(context, {"query": "What is AI?"})
        
        assert result.success is True
        assert result.data is not None
        assert "complexity" in result.data
        assert "subtasks" in result.data
        
    @pytest.mark.asyncio
    async def test_planner_process_complex_query(self):
        """Test planner processes complex query."""
        planner = PlannerAgent()
        context = AgentContext()
        
        result = await planner.process(context, {
            "query": "Compare and analyze AI adoption trends in healthcare and finance sectors over the past decade"
        })
        
        assert result.success is True
        assert result.data["complexity"] in ["medium", "complex"]


class TestRetrieverAgent:
    """Test RetrieverAgent."""
    
    def test_retriever_creation(self):
        """Test retriever agent can be created."""
        retriever = RetrieverAgent()
        assert retriever.name == "Retriever"
        
    def test_retriever_capabilities(self):
        """Test retriever has expected capabilities."""
        retriever = RetrieverAgent()
        capabilities = retriever.get_capabilities()
        assert "semantic_document_search" in capabilities
        assert "web_search" in capabilities
        
    @pytest.mark.asyncio
    async def test_retriever_process(self):
        """Test retriever processes query."""
        retriever = RetrieverAgent()
        context = AgentContext()
        
        result = await retriever.process(context, {"query": "AI trends"})
        
        assert result.success is True
        assert result.data is not None


class TestAnalyzerAgent:
    """Test AnalyzerAgent."""
    
    def test_analyzer_creation(self):
        """Test analyzer agent can be created."""
        analyzer = AnalyzerAgent()
        assert analyzer.name == "Analyzer"
        
    def test_analyzer_capabilities(self):
        """Test analyzer has expected capabilities."""
        analyzer = AnalyzerAgent()
        capabilities = analyzer.get_capabilities()
        assert "insight_extraction" in capabilities
        assert "pattern_detection" in capabilities
        
    @pytest.mark.asyncio
    async def test_analyzer_process(self):
        """Test analyzer processes documents."""
        analyzer = AnalyzerAgent()
        context = AgentContext()
        
        result = await analyzer.process(context, {
            "query": "Analyze trends",
            "documents": {
                "documents": [
                    {"content": "AI is transforming healthcare."}
                ]
            }
        })
        
        assert result.success is True
        assert "key_insights" in result.data


class TestSynthesizerAgent:
    """Test SynthesizerAgent."""
    
    def test_synthesizer_creation(self):
        """Test synthesizer agent can be created."""
        synthesizer = SynthesizerAgent()
        assert synthesizer.name == "Synthesizer"
        
    def test_synthesizer_capabilities(self):
        """Test synthesizer has expected capabilities."""
        synthesizer = SynthesizerAgent()
        capabilities = synthesizer.get_capabilities()
        assert "report_generation" in capabilities
        assert "executive_summary" in capabilities
        
    @pytest.mark.asyncio
    async def test_synthesizer_process(self):
        """Test synthesizer generates response."""
        synthesizer = SynthesizerAgent()
        context = AgentContext()
        
        result = await synthesizer.process(context, {
            "query": "Summarize findings",
            "analysis": {
                "key_insights": [{"content": "Key insight 1"}],
                "summary": "Test summary"
            }
        })
        
        assert result.success is True
        assert "response" in result.data


class TestCriticAgent:
    """Test CriticAgent."""
    
    def test_critic_creation(self):
        """Test critic agent can be created."""
        critic = CriticAgent()
        assert critic.name == "Critic"
        
    def test_critic_capabilities(self):
        """Test critic has expected capabilities."""
        critic = CriticAgent()
        capabilities = critic.get_capabilities()
        assert "quality_scoring" in capabilities
        assert "hallucination_detection" in capabilities
        
    @pytest.mark.asyncio
    async def test_critic_process(self):
        """Test critic evaluates response."""
        critic = CriticAgent()
        context = AgentContext()
        
        result = await critic.process(context, {
            "query": "What is AI?",
            "response": "AI stands for Artificial Intelligence. It is a field of computer science."
        })
        
        assert result.success is True
        assert "score" in result.data


class TestOrchestratorAgent:
    """Test OrchestratorAgent."""
    
    def test_orchestrator_creation(self):
        """Test orchestrator agent can be created."""
        orchestrator = OrchestratorAgent()
        assert orchestrator.name == "Orchestrator"
        
    def test_orchestrator_register_agents(self):
        """Test orchestrator can register agents."""
        orchestrator = OrchestratorAgent()
        planner = PlannerAgent()
        
        orchestrator.register_agents(planner=planner)
        
        assert "planner" in orchestrator.agents
        
    def test_orchestrator_capabilities(self):
        """Test orchestrator has expected capabilities."""
        orchestrator = OrchestratorAgent()
        capabilities = orchestrator.get_capabilities()
        assert "query_coordination" in capabilities
        assert "agent_delegation" in capabilities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
