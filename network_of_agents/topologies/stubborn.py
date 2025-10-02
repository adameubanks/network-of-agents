"""
Stubborn agents network topology implementation.

This implements a small-world network with designated stubborn agents
for the Friedkin-Johnsen model.
"""

import numpy as np
from typing import Dict, Any, Optional
from ..core.base_models import NetworkTopology

class Stubborn(NetworkTopology):
    """
    Small-world network with stubborn agents for Friedkin-Johnsen model.
    
    Creates a Watts-Strogatz small-world network and designates a fraction
    of agents as stubborn (λ=0) while others are flexible (λ=0.8).
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.n_agents = parameters["n_agents"]
        self.k = parameters["k"]
        self.beta = parameters["beta"]
        self.stubborn_fraction = parameters.get("stubborn_fraction", 0.1)
        self.lambda_flexible = parameters.get("lambda_flexible", 0.8)
        
        # Generate the base small-world network
        self.base_network = self._create_small_world_network()
        
        # Designate stubborn agents
        self.stubborn_agents = self._select_stubborn_agents()
        self.lambda_values = self._create_lambda_values()
    
    def _create_small_world_network(self) -> np.ndarray:
        """Create the base Watts-Strogatz small-world network."""
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
                    possible_targets = [x for x in range(n) if x != i and adjacency[i, x] == 0]
                    if possible_targets:
                        target = np.random.choice(possible_targets)
                        adjacency[i, target] = 1
                        adjacency[target, i] = 1
        
        return adjacency
    
    def _select_stubborn_agents(self) -> np.ndarray:
        """Select which agents are stubborn."""
        n_stubborn = int(self.n_agents * self.stubborn_fraction)
        stubborn_indices = np.random.choice(
            self.n_agents, 
            size=n_stubborn, 
            replace=False
        )
        stubborn_mask = np.zeros(self.n_agents, dtype=bool)
        stubborn_mask[stubborn_indices] = True
        return stubborn_mask
    
    def _create_lambda_values(self) -> np.ndarray:
        """Create lambda values for each agent."""
        lambda_values = np.full(self.n_agents, self.lambda_flexible)
        lambda_values[self.stubborn_agents] = 0.0  # Stubborn agents have λ=0
        return lambda_values
    
    def generate_network(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate the small-world network with stubborn agents.
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            Adjacency matrix
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            # Regenerate network and stubborn assignments
            self.base_network = self._create_small_world_network()
            self.stubborn_agents = self._select_stubborn_agents()
            self.lambda_values = self._create_lambda_values()
        
        return self.base_network.copy()
    
    def get_n_agents(self) -> int:
        """Get number of agents in the network."""
        return self.n_agents
    
    def get_lambda_values(self) -> np.ndarray:
        """Get lambda values for each agent."""
        return self.lambda_values.copy()
    
    def get_stubborn_agents(self) -> np.ndarray:
        """Get boolean mask of stubborn agents."""
        return self.stubborn_agents.copy()
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get information about the network.
        
        Returns:
            Dictionary with network statistics
        """
        A = self.base_network
        n_agents = self.n_agents
        n_edges = np.sum(A) // 2
        n_stubborn = np.sum(self.stubborn_agents)
        
        return {
            "n_agents": n_agents,
            "n_edges": n_edges,
            "n_stubborn": n_stubborn,
            "stubborn_fraction": self.stubborn_fraction,
            "lambda_flexible": self.lambda_flexible,
            "avg_degree": np.mean(np.sum(A, axis=1)),
            "topology": "Small-World with Stubborn Agents"
        }
