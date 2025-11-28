"""SmartDoc Analyst - Intelligent Document Research & Analysis Multi-Agent System.

This package provides a production-ready multi-agent document analysis system
featuring six specialized agents, seven tools, three-tier memory, and full observability.
"""

__version__ = "1.0.0"
__author__ = "SmartDoc Analyst Team"

from .core.system import SmartDocAnalyst
from .config import Settings, get_settings

__all__ = [
    "SmartDocAnalyst",
    "Settings",
    "get_settings",
    "__version__",
]
