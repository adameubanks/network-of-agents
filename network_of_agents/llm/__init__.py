"""
Simplified LLM integration framework for the network of agents simulation.
"""

from .agent import LLMAgent
from .litellm_client import LiteLLMClient

__all__ = [
    "LLMAgent",
    "LiteLLMClient",
] 