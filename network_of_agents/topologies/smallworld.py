"""
Watts-Strogatz small-world network topology implementation.
"""

import numpy as np
from typing import Dict, Any, Optional
from ..core.base_models import NetworkTopology

class SmallWorld(NetworkTopology):
    """
    Watts-Strogatz small-world network topology.
    
    Creates a ring lattice where each node is connected to k nearest neighbors,
    then rewires each edge with probability beta.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.n_agents = parameters["n_agents"]
        self.k = parameters["k"]
        self.beta = parameters["beta"]
    
    def generate_network(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a Watts-Strogatz small-world network.
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            Adjacency matrix
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n = self.n_agents
        k = self.k
        beta = self.beta
        
        # Create ring lattice
        adjacency = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            for j in range(1, k // 2 + 1):
                # Connect to k/2 neighbors on each side
                adjacency[i, (i + j) % n] = 1
                adjacency[i, (i - j) % n] = 1
        
        # Rewire edges with probability beta
        for i in range(n):
            for j in range(1, k // 2 + 1):
                if np.random.random() < beta:
                    # Remove existing edge
                    right_neighbor = (i + j) % n
                    adjacency[i, right_neighbor] = 0
                    adjacency[right_neighbor, i] = 0
                    
                    # Add new random edge
                    # Find a random node that's not already connected
                    possible_targets = [x for x in range(n) if x != i and adjacency[i, x] == 0]
                    if possible_targets:
                        target = np.random.choice(possible_targets)
                        adjacency[i, target] = 1
                        adjacency[target, i] = 1
        
        return adjacency
    
    def get_n_agents(self) -> int:
        """Get number of agents in the network."""
        return self.n_agents

