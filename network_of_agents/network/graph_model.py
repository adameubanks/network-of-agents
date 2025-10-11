"""
Network graph model for managing the social network structure.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional

class NetworkModel:
    """
    Manages the social network graph structure and provides analysis methods.
    """
    def __init__(self, n_agents: int, random_seed: Optional[int] = None):
        """
        Initialize the network model.
        
        Args:
            n_agents: Number of agents in the network
            random_seed: Random seed for reproducible results
        """
        self.n_agents = n_agents
        self.random_seed = random_seed
        
        # Paper-consistent: start with empty adjacency; controller sets A[0] via Eq.(4)
        self.adjacency_matrix = np.zeros((self.n_agents, self.n_agents))
        self.network_history = []

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the current adjacency matrix.
        
        Returns:
            Current adjacency matrix
        """
        return self.adjacency_matrix.copy()
    
    def update_adjacency_matrix(self, new_adjacency: np.ndarray):
        """
        Update the adjacency matrix.
        
        Args:
            new_adjacency: New adjacency matrix
        """
        # Store current matrix in history
        self.network_history.append(self.adjacency_matrix.copy())
        
        # Update current matrix
        self.adjacency_matrix = new_adjacency.copy()
    
    
    