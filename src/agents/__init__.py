"""Agents module for SmartDoc Analyst.

This module provides six specialized agents for document analysis:
- OrchestratorAgent: Master coordinator for task management
- RetrieverAgent: Information retrieval from documents and web
- AnalyzerAgent: Deep analysis and insight generation
- SynthesizerAgent: Report synthesis and output generation
- CriticAgent: Quality assurance and validation
- PlannerAgent: Query decomposition and task planning
"""

from .base_agent import BaseAgent, AgentState, AgentContext, AgentResult
from .orchestrator import OrchestratorAgent
from .retriever import RetrieverAgent
from .analyzer import AnalyzerAgent
from .synthesizer import SynthesizerAgent
from .critic import CriticAgent
from .planner import PlannerAgent

__all__ = [
    "BaseAgent",
    "AgentState",
    "AgentContext",
    "AgentResult",
    "OrchestratorAgent",
    "RetrieverAgent",
    "AnalyzerAgent",
    "SynthesizerAgent",
    "CriticAgent",
    "PlannerAgent",
]
