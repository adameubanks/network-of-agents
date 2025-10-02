"""
Erdős-Rényi random network topology implementation.
"""

import numpy as np
from typing import Dict, Any, Optional
from ..core.base_models import NetworkTopology

class Random(NetworkTopology):
    """
    Erdős-Rényi random network topology.
    
    Each possible edge exists with probability p, independently of other edges.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.n_agents = parameters["n_agents"]
        self.p = parameters["p"]
    
    def generate_network(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate an Erdős-Rényi random network.
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            Adjacency matrix
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n = self.n_agents
        p = self.p
        
        # Generate random adjacency matrix
        random_matrix = np.random.random((n, n))
        adjacency = (random_matrix < p).astype(int)
        
        # Make symmetric (undirected graph)
        adjacency = np.triu(adjacency) + np.triu(adjacency, 1).T
        
        # Remove self-loops
        np.fill_diagonal(adjacency, 0)
        
        return adjacency
    
    def get_n_agents(self) -> int:
        """Get number of agents in the network."""
        return self.n_agents

