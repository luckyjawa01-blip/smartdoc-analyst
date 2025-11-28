"""Tests for SmartDoc Analyst tools."""

import pytest
from src.tools import (
    BaseTool,
    ToolResult,
    DocumentSearchTool,
    WebSearchTool,
    CodeExecutionTool,
    CitationTool,
    SummarizationTool,
    FactCheckerTool,
    VisualizationTool
)


class TestToolResult:
    """Test ToolResult class."""
    
    def test_successful_result(self):
        """Test creating successful result."""
        result = ToolResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        
    def test_failed_result(self):
        """Test creating failed result."""
        result = ToolResult(success=False, error="Test error")
        assert result.success is False
        assert result.error == "Test error"


class TestDocumentSearchTool:
    """Test DocumentSearchTool."""
    
    def test_creation(self):
        """Test tool can be created."""
        tool = DocumentSearchTool()
        assert tool.name == "document_search"
        
    def test_schema(self):
        """Test tool returns valid schema."""
        tool = DocumentSearchTool()
        schema = tool.get_schema()
        
        assert "type" in schema
        assert "properties" in schema
        assert "query" in schema["properties"]
        
    @pytest.mark.asyncio
    async def test_execute_demo_mode(self):
        """Test tool returns demo results without vector store."""
        tool = DocumentSearchTool()
        result = await tool.execute(query="test query")
        
        assert result.success is True
        assert "documents" in result.data
        assert result.data.get("demo_mode") is True
        
    @pytest.mark.asyncio
    async def test_execute_no_query(self):
        """Test tool fails without query."""
        tool = DocumentSearchTool()
        result = await tool.execute()
        
        assert result.success is False


class TestWebSearchTool:
    """Test WebSearchTool."""
    
    def test_creation(self):
        """Test tool can be created."""
        tool = WebSearchTool()
        assert tool.name == "web_search"
        
    def test_schema(self):
        """Test tool returns valid schema."""
        tool = WebSearchTool()
        schema = tool.get_schema()
        
        assert "query" in schema["properties"]
        
    @pytest.mark.asyncio
    async def test_execute_demo_mode(self):
        """Test tool returns results (demo or real)."""
        tool = WebSearchTool()
        result = await tool.execute(query="AI trends")
        
        assert result.success is True
        assert "results" in result.data


class TestCodeExecutionTool:
    """Test CodeExecutionTool."""
    
    def test_creation(self):
        """Test tool can be created."""
        tool = CodeExecutionTool()
        assert tool.name == "code_execution"
        
    @pytest.mark.asyncio
    async def test_execute_safe_code(self):
        """Test executing safe Python code."""
        tool = CodeExecutionTool()
        result = await tool.execute(code="result = 2 + 2")
        
        assert result.success is True
        assert result.data["result"] == 4
        
    @pytest.mark.asyncio
    async def test_execute_math_operations(self):
        """Test math operations with pre-loaded modules."""
        tool = CodeExecutionTool()
        # Math module is pre-loaded, use it directly
        result = await tool.execute(
            code="result = math.sqrt(16)"
        )
        
        assert result.success is True
        assert result.data["result"] == 4.0
        
    @pytest.mark.asyncio
    async def test_block_dangerous_code(self):
        """Test blocking dangerous code."""
        tool = CodeExecutionTool()
        
        # Try to import os
        result = await tool.execute(code="import os; os.system('ls')")
        assert result.success is False
        
    @pytest.mark.asyncio
    async def test_block_file_operations(self):
        """Test blocking file operations."""
        tool = CodeExecutionTool()
        
        result = await tool.execute(code="open('/etc/passwd').read()")
        assert result.success is False


class TestCitationTool:
    """Test CitationTool."""
    
    def test_creation(self):
        """Test tool can be created."""
        tool = CitationTool()
        assert tool.name == "citation"
        
    @pytest.mark.asyncio
    async def test_add_citation(self):
        """Test adding a citation."""
        tool = CitationTool()
        result = await tool.execute(
            action="add",
            source_type="document",
            title="Test Document",
            author="John Doe"
        )
        
        assert result.success is True
        assert "citation_id" in result.data
        
    @pytest.mark.asyncio
    async def test_list_citations(self):
        """Test listing citations."""
        tool = CitationTool()
        
        # Add a citation first
        await tool.execute(
            action="add",
            title="Test",
            author="Author"
        )
        
        result = await tool.execute(action="list")
        
        assert result.success is True
        assert "citations" in result.data
        
    @pytest.mark.asyncio
    async def test_format_citations(self):
        """Test formatting citations."""
        tool = CitationTool()
        
        # Add a citation
        await tool.execute(
            action="add",
            title="Test Paper",
            author="Smith, J.",
            date="2024"
        )
        
        result = await tool.execute(action="format", style="apa")
        
        assert result.success is True
        assert "formatted_citations" in result.data


class TestSummarizationTool:
    """Test SummarizationTool."""
    
    def test_creation(self):
        """Test tool can be created."""
        tool = SummarizationTool()
        assert tool.name == "summarization"
        
    @pytest.mark.asyncio
    async def test_extractive_summarize(self):
        """Test extractive summarization."""
        tool = SummarizationTool(method="extractive")
        
        text = """
        Artificial Intelligence is transforming many industries.
        Machine learning algorithms can process vast amounts of data.
        Deep learning has shown remarkable success in image recognition.
        Natural language processing enables computers to understand text.
        AI applications range from healthcare to autonomous vehicles.
        """
        
        result = await tool.execute(text=text, max_length=50)
        
        assert result.success is True
        assert "summary" in result.data
        assert len(result.data["summary"].split()) <= 60  # Allow some margin


class TestFactCheckerTool:
    """Test FactCheckerTool."""
    
    def test_creation(self):
        """Test tool can be created."""
        tool = FactCheckerTool()
        assert tool.name == "fact_checker"
        
    @pytest.mark.asyncio
    async def test_verify_claim_with_sources(self):
        """Test verifying a claim against sources."""
        tool = FactCheckerTool()
        
        result = await tool.execute(
            claim="AI is used in healthcare",
            sources=[
                {"content": "Artificial Intelligence is transforming healthcare delivery."}
            ]
        )
        
        assert result.success is True
        assert "verified" in result.data
        assert "confidence" in result.data
        
    @pytest.mark.asyncio
    async def test_verify_without_sources(self):
        """Test verification without sources."""
        tool = FactCheckerTool()
        
        result = await tool.execute(claim="Some claim")
        
        assert result.success is True
        assert result.data["verified"] is False


class TestVisualizationTool:
    """Test VisualizationTool."""
    
    def test_creation(self):
        """Test tool can be created."""
        tool = VisualizationTool()
        assert tool.name == "visualization"
        
    @pytest.mark.asyncio
    async def test_create_bar_chart(self):
        """Test creating bar chart."""
        tool = VisualizationTool()
        
        result = await tool.execute(
            data={"A": 10, "B": 20, "C": 30},
            chart_type="bar",
            title="Test Chart"
        )
        
        assert result.success is True
        assert "chart_type" in result.data
        
    @pytest.mark.asyncio
    async def test_create_pie_chart(self):
        """Test creating pie chart."""
        tool = VisualizationTool()
        
        result = await tool.execute(
            data={"Category A": 40, "Category B": 60},
            chart_type="pie"
        )
        
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
