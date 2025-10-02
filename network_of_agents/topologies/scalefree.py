"""
Barabási-Albert scale-free network topology implementation.
"""

import numpy as np
from typing import Dict, Any, Optional
from ..core.base_models import NetworkTopology

class ScaleFree(NetworkTopology):
    """
    Barabási-Albert scale-free network topology.
    
    Grows a network by adding nodes one at a time, connecting each new node
    to m existing nodes with probability proportional to their degree.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.n_agents = parameters["n_agents"]
        self.m = parameters["m"]
    
    def generate_network(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a Barabási-Albert scale-free network.
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            Adjacency matrix
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n = self.n_agents
        m = self.m
        
        if n < m + 1:
            raise ValueError(f"n_agents ({n}) must be >= m + 1 ({m + 1})")
        
        # Start with m+1 nodes fully connected
        adjacency = np.zeros((n, n), dtype=int)
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                adjacency[i, j] = 1
                adjacency[j, i] = 1
        
        # Add remaining nodes
        for new_node in range(m + 1, n):
            # Calculate degree distribution for existing nodes
            degrees = np.sum(adjacency[:new_node, :new_node], axis=1)
            total_degree = np.sum(degrees)
            
            if total_degree == 0:
                # Fallback: connect to first m nodes
                for i in range(min(m, new_node)):
                    adjacency[new_node, i] = 1
                    adjacency[i, new_node] = 1
            else:
                # Connect to m existing nodes with probability proportional to degree
                probabilities = degrees / total_degree
                chosen_nodes = np.random.choice(
                    new_node, 
                    size=min(m, new_node), 
                    replace=False, 
                    p=probabilities
                )
                
                for chosen_node in chosen_nodes:
                    adjacency[new_node, chosen_node] = 1
                    adjacency[chosen_node, new_node] = 1
        
        return adjacency
    
    def get_n_agents(self) -> int:
        """Get number of agents in the network."""
        return self.n_agents

