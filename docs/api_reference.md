# SmartDoc Analyst API Reference

## Overview

This document provides the API reference for SmartDoc Analyst, including all public classes, methods, and their usage.

## Core System

### SmartDocAnalyst

Main entry point for the document analysis system.

```python
from src.core.system import SmartDocAnalyst

analyst = SmartDocAnalyst(
    api_key: Optional[str] = None,
    settings: Optional[Settings] = None,
    llm: Optional[LLMInterface] = None
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | str, optional | Gemini API key. If not provided, uses `SMARTDOC_GEMINI_API_KEY` env var |
| `settings` | Settings, optional | Custom settings object |
| `llm` | LLMInterface, optional | Custom LLM interface |

#### Methods

##### `ingest_documents(documents: List[Dict]) -> Dict`

Ingest documents into the system.

```python
result = analyst.ingest_documents([
    {
        "content": "Document text content...",
        "metadata": {
            "source": "document.pdf",
            "title": "Document Title",
            "author": "Author Name"
        }
    }
])
```

**Returns**:
```python
{
    "added": 1,           # Number of documents added
    "rejected": 0,        # Number rejected
    "document_ids": [...], # List of assigned IDs
    "rejected_details": [] # Details of rejections
}
```

##### `analyze(query: str, include_web_search: bool = True, user_id: str = None) -> Dict`

Analyze documents and answer a query.

```python
result = await analyst.analyze(
    query="What are the main findings?",
    include_web_search=True,
    user_id="user123"
)
```

**Returns**:
```python
{
    "success": True,
    "answer": "Analysis response...",
    "sources": [{"title": "...", "content": "..."}],
    "analysis": {...},
    "quality_score": 0.85,
    "processing_stages": ["planning", "retrieval", "analysis", "synthesis"],
    "execution_time_ms": 1500.0,
    "task_id": "uuid..."
}
```

##### `search(query: str, k: int = 5, include_web: bool = False) -> Dict`

Search documents without full analysis.

```python
results = await analyst.search(
    query="AI healthcare",
    k=5,
    include_web=False
)
```

**Returns**:
```python
{
    "documents": [
        {"content": "...", "metadata": {...}, "score": 0.95}
    ]
}
```

##### `summarize(text: str, max_length: int = 200) -> str`

Generate a summary of text content.

```python
summary = await analyst.summarize(
    text="Long text content...",
    max_length=100
)
```

##### `get_stats() -> Dict`

Get system statistics.

```python
stats = analyst.get_stats()
# Returns: {"memory": {...}, "metrics": {...}, "traces": {...}, "tools": {...}}
```

##### `clear_memory() -> Dict`

Clear all memory stores.

```python
result = analyst.clear_memory()
# Returns: {"working": 10, "episodic": 5, "semantic": 20}
```

---

## Agents

### BaseAgent

Abstract base class for all agents.

```python
from src.agents.base_agent import BaseAgent, AgentContext, AgentResult

class MyAgent(BaseAgent):
    async def process(self, context: AgentContext, input_data: Any) -> AgentResult:
        # Implementation
        return AgentResult(success=True, data=result)
    
    def get_capabilities(self) -> List[str]:
        return ["my_capability"]
```

### AgentContext

Context passed between agents.

```python
context = AgentContext(
    task_id="uuid",           # Unique task ID
    trace_id="uuid",          # Tracing ID
    query="user query",       # Original query
    intermediate_results={},  # Results from other agents
    metadata={}               # Additional metadata
)
```

### AgentResult

Result returned by agents.

```python
result = AgentResult(
    success=True,             # Success status
    data={"key": "value"},    # Output data
    error=None,               # Error message if failed
    metrics={"time_ms": 100}, # Performance metrics
    suggestions=[]            # Improvement suggestions
)
```

### Specialized Agents

#### OrchestratorAgent

```python
from src.agents.orchestrator import OrchestratorAgent

orchestrator = OrchestratorAgent(
    llm=llm,
    max_iterations=10,
    parallel_execution=True
)

# Register other agents
orchestrator.register_agents(
    planner=PlannerAgent(),
    retriever=RetrieverAgent()
)

# Process query
result = await orchestrator.process(context, "query")
```

#### PlannerAgent

```python
from src.agents.planner import PlannerAgent

planner = PlannerAgent(llm=llm)
result = await planner.process(context, {"query": "complex query"})

# Result data includes:
# - complexity: "simple" | "medium" | "complex"
# - subtasks: [{"task": "...", "description": "..."}]
# - strategy: "sequential" | "parallel"
```

#### RetrieverAgent

```python
from src.agents.retriever import RetrieverAgent

retriever = RetrieverAgent(
    vector_store=vector_store,
    web_search_tool=WebSearchTool(),
    llm=llm
)
result = await retriever.process(context, {"query": "search query"})
```

#### AnalyzerAgent

```python
from src.agents.analyzer import AnalyzerAgent

analyzer = AnalyzerAgent(
    llm=llm,
    code_executor=CodeExecutionTool(),
    fact_checker=FactCheckerTool()
)
result = await analyzer.process(context, {
    "query": "analyze this",
    "documents": [...]
})
```

#### SynthesizerAgent

```python
from src.agents.synthesizer import SynthesizerAgent

synthesizer = SynthesizerAgent(llm=llm)
result = await synthesizer.process(context, {
    "query": "original query",
    "analysis": {...}
})
```

#### CriticAgent

```python
from src.agents.critic import CriticAgent

critic = CriticAgent(llm=llm)
result = await critic.process(context, {
    "query": "original query",
    "response": "generated response"
})

# Result data includes:
# - score: 0.0-1.0
# - needs_improvement: bool
# - issues: [...]
# - suggestions: [...]
```

---

## Tools

### BaseTool

Abstract base class for all tools.

```python
from src.tools.base_tool import BaseTool, ToolResult

class MyTool(BaseTool):
    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, data=result)
    
    def get_schema(self) -> Dict:
        return {"param": {"type": "string", "required": True}}
```

### ToolResult

Result returned by tools.

```python
result = ToolResult(
    success=True,
    data={"output": "..."},
    error=None,
    execution_time_ms=50.0,
    metadata={}
)
```

### Tool Classes

#### DocumentSearchTool

```python
from src.tools.document_search import DocumentSearchTool

tool = DocumentSearchTool(vector_store=vector_store)
result = await tool.execute(query="search query", k=5)
```

#### WebSearchTool

```python
from src.tools.web_search import WebSearchTool

tool = WebSearchTool()
result = await tool.execute(query="search query", max_results=10)
```

#### CodeExecutionTool

```python
from src.tools.code_execution import CodeExecutionTool

tool = CodeExecutionTool(timeout=30)
result = await tool.execute(code="result = 2 + 2")
```

#### CitationTool

```python
from src.tools.citation import CitationTool

tool = CitationTool()
await tool.execute(action="add", source={...})
await tool.execute(action="list")
await tool.execute(action="format", style="apa")
```

#### SummarizationTool

```python
from src.tools.summarization import SummarizationTool

tool = SummarizationTool(llm=llm)
result = await tool.execute(text="long text", max_length=100)
```

#### FactCheckerTool

```python
from src.tools.fact_checker import FactCheckerTool

tool = FactCheckerTool(llm=llm)
result = await tool.execute(
    claim="claim to verify",
    sources=[{"content": "..."}]
)
```

#### VisualizationTool

```python
from src.tools.visualization import VisualizationTool

tool = VisualizationTool()
result = await tool.execute(
    chart_type="bar",
    data={"labels": [...], "values": [...]},
    title="Chart Title"
)
```

---

## Memory

### MemoryManager

Unified memory management.

```python
from src.memory import MemoryManager

memory = MemoryManager(
    working_memory_size=100,
    vector_store_path="./data/vectors"
)

# Working memory
memory.add_to_context("item", metadata={}, importance=0.8)
context = memory.get_recent_context(n=10)

# Episodic memory
memory.store_episode("episode_id", data)
episode = memory.recall_episode("episode_id")

# Semantic memory
memory.store_fact("fact_key", "fact value")
fact = memory.recall_fact("fact_key")

# Document search
memory.add_documents([{"content": "...", "metadata": {...}}])
results = memory.search_documents("query", k=5)
```

---

## Observability

### Logging

```python
from src.observability import get_logger

logger = get_logger("component.name", level="INFO")

logger.info("Message", extra={
    "metric": 100,
    "trace_id": "abc123"
})
```

### Metrics

```python
from src.observability import metrics

metrics.increment("counter_name")
metrics.gauge("gauge_name", value)
metrics.timing("timer_name", milliseconds)

all_metrics = metrics.get_all_metrics()
```

### Tracing

```python
from src.observability import get_tracer

tracer = get_tracer("service_name")

with tracer.span("operation_name", {"key": "value"}) as span:
    span.set_attribute("attr", "value")
    # Do work
```

---

## Configuration

### Settings

```python
from src.config import Settings, get_settings

settings = get_settings()

# Available settings:
# - gemini_api_key: str
# - model_name: str (default: "gemini-pro")
# - temperature: float (default: 0.7)
# - max_tokens: int (default: 2048)
# - log_level: str (default: "INFO")
# - vector_store_path: str
# - max_agent_iterations: int
# - parallel_agents: bool
# - max_input_length: int
# - rate_limit_rpm: int
```

---

## Safety

### SafetyGuard

```python
from src.core.safety import SafetyGuard

guard = SafetyGuard(
    max_input_length=10000,
    rate_limit_rpm=60
)

# Validate input
result = guard.validate_input(text)
# Returns: ValidationResult(valid=bool, sanitized=str, issues=[], risk_score=float)

# Sanitize output
clean = guard.sanitize_output(response)

# Rate limiting
allowed = guard.rate_limit(user_id)
```

---

## Error Handling

All methods may raise:

- `ValueError`: Invalid parameters
- `RuntimeError`: Execution errors
- `TimeoutError`: Operation timeouts

Use try/except for robust error handling:

```python
try:
    result = await analyst.analyze(query)
    if not result["success"]:
        print(f"Analysis failed: {result['error']}")
except Exception as e:
    print(f"Error: {e}")
```
