"""
Stochastic Block Model network topology implementation.
"""

import numpy as np
from typing import Dict, Any, Optional
from ..core.base_models import NetworkTopology

class Echo(NetworkTopology):
    """
    Stochastic Block Model network topology.
    
    Creates communities with high internal connectivity and low inter-community connectivity.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.n_agents = parameters["n_agents"]
        self.n_communities = parameters["n_communities"]
        self.p_intra = parameters["p_intra"]
        self.p_inter = parameters["p_inter"]
    
    def generate_network(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a Stochastic Block Model network.
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            Adjacency matrix
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n = self.n_agents
        n_communities = self.n_communities
        p_intra = self.p_intra
        p_inter = self.p_inter
        
        # Assign nodes to communities
        community_size = n // n_communities
        communities = []
        for i in range(n_communities):
            start_idx = i * community_size
            if i == n_communities - 1:
                # Last community gets remaining nodes
                end_idx = n
            else:
                end_idx = (i + 1) * community_size
            communities.append(list(range(start_idx, end_idx)))
        
        # Initialize adjacency matrix
        adjacency = np.zeros((n, n), dtype=int)
        
        # Add intra-community edges
        for community in communities:
            for i in community:
                for j in community:
                    if i != j and np.random.random() < p_intra:
                        adjacency[i, j] = 1
        
        # Add inter-community edges
        for i in range(n):
            for j in range(i + 1, n):
                # Check if nodes are in different communities
                i_community = None
                j_community = None
                for comm_idx, community in enumerate(communities):
                    if i in community:
                        i_community = comm_idx
                    if j in community:
                        j_community = comm_idx
                
                if i_community != j_community and np.random.random() < p_inter:
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
        
        return adjacency
    
    def get_n_agents(self) -> int:
        """Get number of agents in the network."""
        return self.n_agents

