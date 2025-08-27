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
    
    def get_network_density(self) -> float:
        """
        Calculate the current network density.
        
        Returns:
            Network density (number of edges / maximum possible edges)
        """
        total_edges = np.sum(self.adjacency_matrix) / 2  # Divide by 2 for undirected graph
        max_possible_edges = self.n_agents * (self.n_agents - 1) / 2
        return total_edges / max_possible_edges
    
    def get_average_degree(self) -> float:
        """
        Calculate the average degree of nodes in the network.
        
        Returns:
            Average degree
        """
        degrees = np.sum(self.adjacency_matrix, axis=1)
        return np.mean(degrees)
    

    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert network model to dictionary for serialization.
        
        Returns:
            Dictionary representation of the network model
        """
        return {
            'n_agents': self.n_agents,
            'adjacency_matrix': self.adjacency_matrix.tolist(),
            'network_history': [adj.tolist() for adj in self.network_history]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkModel':
        """
        Create network model from dictionary.
        
        Args:
            data: Dictionary representation of the network model
            
        Returns:
            Network model instance
        """
        network = cls(
            n_agents=data['n_agents']
        )
        network.adjacency_matrix = np.array(data['adjacency_matrix'])
        network.network_history = [np.array(adj) for adj in data['network_history']]
        return network 