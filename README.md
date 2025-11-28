# 🔍 SmartDoc Analyst

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Kaggle](https://img.shields.io/badge/Kaggle-Agents%20Intensive-20BEFF.svg)](https://www.kaggle.com/)

> **Intelligent Document Research & Analysis Multi-Agent System**
>
> A production-ready multi-agent system for comprehensive document analysis, built for the Kaggle Agents Intensive Capstone Project 2025.

```
   _____ __  __   ___    ____ _____   ____   ____   ____ 
  / ___//  |/  | /   |  / __ \_   _| |  _ \ / __ \ / ___|
  \___ \|  .  . |/ /| | / /_/ / | |   | | | | |  | | |    
  ___) |  |\/|  / ___ |/  _  /  | |   | |_| | |__| | |___ 
 |____/|__|  |_/_/  |_/_/ |_|  |___|  |____/ \____/ \____|
                                                          
  Intelligent Document Research & Analysis Multi-Agent System
```

## ✨ Features

### 🤖 Six Specialized Agents
- **Orchestrator**: Master coordinator for task management
- **Planner**: Query decomposition and strategy planning
- **Retriever**: Semantic document and web search
- **Analyzer**: Deep analysis and insight generation
- **Synthesizer**: Report generation and summarization
- **Critic**: Quality assurance and validation

### 🛠️ Seven Powerful Tools
- Document Search (vector similarity)
- Web Search (real-time results)
- Code Execution (sandboxed Python)
- Citation Management
- Text Summarization
- Fact Checking
- Data Visualization

### 🧠 Three-Tier Memory System
- **Working Memory**: Current task context
- **Episodic Memory**: Conversation history
- **Semantic Memory**: Persistent knowledge

### 📊 Full Observability Stack
- Structured logging with trace correlation
- Metrics collection and monitoring
- Distributed tracing across agents

### 🔒 Safety & Security
- Input validation and sanitization
- Rate limiting
- Safe code execution sandbox
- Output sanitization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SmartDoc Analyst                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 Orchestrator Agent                        │    │
│  │   Planner → Retriever → Analyzer → Synthesizer ← Critic │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓ ↑                                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐    │
│  │    7 Tools    │    │  3-Tier       │    │ Observability │    │
│  │               │    │  Memory       │    │ Stack         │    │
│  └───────────────┘    └───────────────┘    └───────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Safety Guards & A2A Protocol                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/smartdoc-analyst/smartdoc-analyst.git
cd smartdoc-analyst

# Install dependencies
pip install -r requirements.txt

# Set your API key
export SMARTDOC_GEMINI_API_KEY="your-api-key"
```

### Basic Usage

```python
import asyncio
from src.core.system import SmartDocAnalyst

async def main():
    # Initialize the system
    analyst = SmartDocAnalyst()
    
    # Ingest documents
    analyst.ingest_documents([
        {
            "content": "AI is transforming healthcare...",
            "metadata": {"source": "report.pdf", "title": "AI Report"}
        }
    ])
    
    # Analyze and answer questions
    result = await analyst.analyze(
        "What are the main trends in AI healthcare?"
    )
    
    print(result["answer"])
    print(f"Sources: {result['sources']}")
    print(f"Quality Score: {result['quality_score']}")

asyncio.run(main())
```

### Run Examples

```bash
# Basic usage
python examples/basic_usage.py

# Multi-document analysis
python examples/multi_document.py

# Custom agents
python examples/custom_agents.py
```

## 📈 Evaluation Results

| Metric | Score | Target |
|--------|-------|--------|
| Task Success Rate | 95.5% | ≥90% ✅ |
| Completeness | 88.2% | ≥85% ✅ |
| Relevance | 92.1% | ≥90% ✅ |
| Citation Accuracy | 87.3% | ≥80% ✅ |
| Hallucination Rate | 3.2% | ≤5% ✅ |
| Avg. Latency | 1.2s | ≤3s ✅ |

> See [evaluation_report.md](docs/evaluation_report.md) for detailed results.

## 📚 Course Concepts Demonstrated

This project demonstrates all seven core concepts from the Kaggle Agents Intensive course:

| Concept | Implementation |
|---------|---------------|
| 🤖 Multi-Agent System | 6 specialized agents with clear roles |
| 🛠️ Tool Integration | 7 tools for various capabilities |
| 🧠 Memory Management | 3-tier memory (Working/Episodic/Semantic) |
| 📝 Context Handling | Agent context passing and accumulation |
| 📊 Observability | Logging, metrics, and tracing |
| ✅ Evaluation | 22 test cases across 7 categories |
| 🚀 Production Ready | Safety guards, configuration, deployment |

## 📁 Project Structure

```
smartdoc-analyst/
├── src/
│   ├── agents/           # 6 specialized agents
│   ├── tools/            # 7 analysis tools
│   ├── memory/           # 3-tier memory system
│   ├── observability/    # Logging, metrics, tracing
│   ├── protocols/        # A2A communication
│   ├── core/             # Main system, LLM, safety
│   └── config.py         # Configuration
├── evaluation/           # Evaluation framework
├── tests/               # Unit tests
├── examples/            # Usage examples
├── docs/                # Documentation
└── notebooks/           # Kaggle submission notebook
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_agents.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📖 Documentation

- [Architecture](docs/architecture.md) - System design and components
- [API Reference](docs/api_reference.md) - Complete API documentation
- [Evaluation Report](docs/evaluation_report.md) - Detailed evaluation results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle Agents Intensive Course Team
- Google Gemini API
- LangChain Community

---

<p align="center">
  Built with ❤️ for the Kaggle Agents Intensive Capstone Project 2025
</p>
