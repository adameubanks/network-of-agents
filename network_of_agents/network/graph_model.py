"""
Network graph model for managing the social network structure.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional

class NetworkModel:
    """Simple network model for adjacency matrix management."""
    def __init__(self, n_agents: int, random_seed: Optional[int] = None):
        self.n_agents = n_agents
        self.random_seed = random_seed
        self.adjacency_matrix = np.zeros((self.n_agents, self.n_agents))
        self.network_history = []

    def get_adjacency_matrix(self) -> np.ndarray:
        return self.adjacency_matrix.copy()
    
    def update_adjacency_matrix(self, new_adjacency: np.ndarray):
        self.network_history.append(self.adjacency_matrix.copy())
        self.adjacency_matrix = new_adjacency.copy()
    
    
    