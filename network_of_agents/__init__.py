"""
Network of Agents: A social network simulation with LLM agents.

This package provides a framework for studying opinion convergence, biases, and fairness
in social networks where users are represented by LLM agents.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .simulation.controller import SimulationController
from .network.graph_model import NetworkModel
from .llm.agent import LLMAgent
from .visualization.dashboard import Dashboard

__all__ = [
    "SimulationController",
    "NetworkModel", 
    "LLMAgent",
    "Dashboard",
] 