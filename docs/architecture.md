# SmartDoc Analyst Architecture

## Overview

SmartDoc Analyst is a production-ready multi-agent document analysis system built for the Kaggle Agents Intensive Capstone Project 2025. This document describes the system architecture, component interactions, and design decisions.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SmartDoc Analyst                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Orchestrator Agent                         │    │
│  │   ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │    │
│  │   │ Planner │→│ Retriever│→│ Analyzer │→│Synthesizer│        │    │
│  │   └─────────┘ └──────────┘ └──────────┘ └──────────┘        │    │
│  │                      ↓ ↑                    ↑                 │    │
│  │                    ┌──────────┐             │                 │    │
│  │                    │  Critic  │─────────────┘                 │    │
│  │                    └──────────┘                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                 ↓ ↑                                  │
│  ┌──────────────────────┐  ┌────────────────────┐                   │
│  │        Tools         │  │      Memory        │                   │
│  │ ┌─────────────────┐  │  │ ┌────────────────┐ │                   │
│  │ │ Document Search │  │  │ │ Working Memory │ │                   │
│  │ │ Web Search      │  │  │ │ Episodic Memory│ │                   │
│  │ │ Code Execution  │  │  │ │ Semantic Memory│ │                   │
│  │ │ Citation        │  │  │ │ Vector Store   │ │                   │
│  │ │ Summarization   │  │  │ └────────────────┘ │                   │
│  │ │ Fact Checker    │  │  └────────────────────┘                   │
│  │ │ Visualization   │  │                                           │
│  │ └─────────────────┘  │                                           │
│  └──────────────────────┘                                           │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      Observability Layer                       │   │
│  │    ┌──────────┐    ┌──────────┐    ┌──────────┐              │   │
│  │    │  Logger  │    │ Metrics  │    │  Tracer  │              │   │
│  │    └──────────┘    └──────────┘    └──────────┘              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                       Safety Guards                            │   │
│  │    Input Validation │ Rate Limiting │ Output Sanitization     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent System

The system uses six specialized agents working together:

#### OrchestratorAgent (Master Coordinator)
- **Role**: Central controller for all agent interactions
- **Responsibilities**:
  - Task planning and delegation
  - Parallel/sequential agent coordination
  - Quality control and retry logic
  - Result aggregation

#### PlannerAgent (Query Decomposition)
- **Role**: Analyze and break down complex queries
- **Responsibilities**:
  - Query complexity analysis
  - Subtask identification
  - Execution strategy planning

#### RetrieverAgent (Information Retrieval)
- **Role**: Find relevant information
- **Responsibilities**:
  - Semantic document search
  - Web search integration
  - Citation tracking
  - Result ranking

#### AnalyzerAgent (Deep Analysis)
- **Role**: Analyze retrieved information
- **Responsibilities**:
  - Pattern detection
  - Insight extraction
  - Fact verification
  - Code execution for calculations

#### SynthesizerAgent (Output Generation)
- **Role**: Generate final responses
- **Responsibilities**:
  - Multi-source fusion
  - Report generation
  - Executive summaries
  - Citation formatting

#### CriticAgent (Quality Assurance)
- **Role**: Validate response quality
- **Responsibilities**:
  - Quality scoring
  - Hallucination detection
  - Consistency checking
  - Improvement suggestions

### 2. Tools

Seven tools provide specific capabilities:

| Tool | Purpose | Key Features |
|------|---------|--------------|
| DocumentSearchTool | Semantic search | Vector similarity, ranking |
| WebSearchTool | External search | Real-time web results |
| CodeExecutionTool | Safe code execution | Sandboxed Python |
| CitationTool | Citation management | Tracking, formatting |
| SummarizationTool | Text summarization | Extractive, abstractive |
| FactCheckerTool | Claim verification | Source-based checking |
| VisualizationTool | Data visualization | Charts, graphs |

### 3. Memory System

Three-tier memory architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Manager                            │
├───────────────┬───────────────────┬────────────────────────┤
│ Working Memory│  Episodic Memory  │   Semantic Memory      │
│   (Current)   │    (Session)      │    (Persistent)        │
├───────────────┼───────────────────┼────────────────────────┤
│ - Task context│ - Conversation    │ - Facts & knowledge    │
│ - Recent items│   history         │ - Preferences          │
│ - Priorities  │ - Episodes        │ - Learning             │
└───────────────┴───────────────────┴────────────────────────┘
                         ↓
              ┌─────────────────────┐
              │    Vector Store     │
              │  (Document Index)   │
              └─────────────────────┘
```

### 4. Observability

Full observability stack:

- **Structured Logging**: JSON-formatted logs with trace correlation
- **Metrics Collection**: Performance and usage metrics
- **Distributed Tracing**: Request tracing across agents

### 5. A2A Protocol

Agent-to-Agent communication protocol:

```python
@dataclass
class AgentMessage:
    id: str                    # Unique message ID
    from_agent: str            # Sender agent
    to_agent: str              # Recipient agent
    message_type: MessageType  # TASK, RESULT, ERROR, FEEDBACK
    content: Any               # Message content
    metadata: Dict             # Additional metadata
    timestamp: datetime        # Message timestamp
    correlation_id: str        # Request-response correlation
```

### 6. Safety Guards

Input/output protection:

- **Input Validation**: Injection prevention, length limits
- **Rate Limiting**: Per-user request limits
- **Output Sanitization**: PII/sensitive data removal

## Data Flow

### Query Processing Flow

```
1. User Query
      ↓
2. Safety Validation
      ↓
3. Orchestrator receives query
      ↓
4. Planner decomposes query
      ↓
5. Retriever fetches documents
      ↓
6. Analyzer processes content
      ↓
7. Synthesizer generates response
      ↓
8. Critic validates quality
      ↓ (if needs improvement)
      ↓
9. Re-synthesis with feedback
      ↓
10. Final response
```

### Document Ingestion Flow

```
1. Document Input
      ↓
2. Safety Validation
      ↓
3. Text Extraction
      ↓
4. Chunking
      ↓
5. Embedding Generation
      ↓
6. Vector Store Indexing
      ↓
7. Metadata Storage
```

## Design Decisions

### Why Multi-Agent?
- **Separation of concerns**: Each agent has a specific responsibility
- **Scalability**: Agents can be scaled independently
- **Flexibility**: Easy to add/replace agents
- **Maintainability**: Isolated code changes

### Why Three-Tier Memory?
- **Working Memory**: Fast access to current context
- **Episodic Memory**: Conversation continuity
- **Semantic Memory**: Long-term knowledge retention

### Why Full Observability?
- **Debugging**: Trace issues across agents
- **Performance**: Identify bottlenecks
- **Monitoring**: Production health tracking

## Extension Points

### Adding Custom Agents

```python
class MyCustomAgent(BaseAgent):
    async def process(self, context, input_data):
        # Your logic here
        return AgentResult(success=True, data=result)
    
    def get_capabilities(self):
        return ["my_capability"]

# Register with orchestrator
orchestrator.register_agents(my_agent=MyCustomAgent())
```

### Adding Custom Tools

```python
class MyCustomTool(BaseTool):
    async def execute(self, **kwargs):
        # Your logic here
        return ToolResult(success=True, data=result)
    
    def get_schema(self):
        return {"param": {"type": "string", "required": True}}
```

## Performance Considerations

- **Parallel Execution**: Agents run in parallel when possible
- **Caching**: Memory system caches frequent queries
- **Lazy Loading**: Components load on demand
- **Connection Pooling**: Reused API connections

## Security

- **Input Validation**: All inputs validated before processing
- **Code Sandboxing**: Python execution in restricted environment
- **Rate Limiting**: Prevents abuse
- **Output Sanitization**: Removes sensitive data

## Future Improvements

1. **Streaming Responses**: Real-time output streaming
2. **Multi-Modal Support**: Image and audio processing
3. **Advanced RAG**: Hybrid search strategies
4. **Agent Learning**: Continuous improvement from feedback
