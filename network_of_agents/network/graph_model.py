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
    
    def __init__(self, n_agents: int, initial_connection_probability: float = 0.2, random_seed: Optional[int] = None):
        """
        Initialize the network model.
        
        Args:
            n_agents: Number of agents in the network
            initial_connection_probability: Probability of initial connections
            random_seed: Random seed for reproducible results
        """
        self.n_agents = n_agents
        self.initial_connection_probability = initial_connection_probability
        self.random_seed = random_seed
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.adjacency_matrix = self._initialize_adjacency_matrix()
        self.network_history = []
    
    def _initialize_adjacency_matrix(self) -> np.ndarray:
        """
        Initialize the adjacency matrix with random connections.
        
        Returns:
            Initial adjacency matrix
        """
        A = np.zeros((self.n_agents, self.n_agents))
        
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.random.rand() < self.initial_connection_probability:
                    A[i, j] = 1
                    A[j, i] = 1
        
        return A
    

    
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
    
    def get_clustering_coefficient(self) -> float:
        """
        Calculate the global clustering coefficient.
        
        Returns:
            Global clustering coefficient
        """
        G = nx.from_numpy_array(self.adjacency_matrix)
        return nx.average_clustering(G)
    
    def get_connected_components(self) -> List[List[int]]:
        """
        Find connected components in the network.
        
        Returns:
            List of connected components (each component is a list of agent IDs)
        """
        G = nx.from_numpy_array(self.adjacency_matrix)
        components = list(nx.connected_components(G))
        return [list(comp) for comp in components]
    

    
    def get_agent_degrees(self) -> Dict[int, int]:
        """
        Get the degree of each agent.
        
        Returns:
            Dictionary mapping agent ID to degree
        """
        degrees = np.sum(self.adjacency_matrix, axis=1)
        return {i: int(degrees[i]) for i in range(self.n_agents)}
    

    
    def to_networkx(self) -> nx.Graph:
        """
        Convert to NetworkX graph for advanced analysis.
        
        Returns:
            NetworkX graph object
        """
        G = nx.from_numpy_array(self.adjacency_matrix)
        return G
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert network model to dictionary for serialization.
        
        Returns:
            Dictionary representation of the network model
        """
        return {
            'n_agents': self.n_agents,
            'initial_connection_probability': self.initial_connection_probability,
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
            n_agents=data['n_agents'],
            initial_connection_probability=data['initial_connection_probability']
        )
        network.adjacency_matrix = np.array(data['adjacency_matrix'])
        network.network_history = [np.array(adj) for adj in data['network_history']]
        return network 