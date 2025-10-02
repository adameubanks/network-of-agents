"""
Zachary's Karate Club network topology implementation.

This implements the classic Zachary's Karate Club network, a real social network
from a karate club that split into two factions.
"""

import numpy as np
from typing import Dict, Any, Optional
from ..core.base_models import NetworkTopology

class Karate(NetworkTopology):
    """
    Zachary's Karate Club network topology.
    
    This is the classic empirical network from Zachary's 1977 study of a karate club
    that split into two factions. The network has 34 nodes and 78 edges.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(parameters)
        self.n_agents = 34  # Fixed by the dataset
        
        # Zachary's Karate Club adjacency matrix (34x34)
        # This is the actual network from the 1977 study
        self.adjacency_matrix = self._create_karate_club_network()
    
    def _create_karate_club_network(self) -> np.ndarray:
        """
        Create the Zachary's Karate Club network adjacency matrix.
        
        Returns:
            34x34 adjacency matrix
        """
        # Zachary's Karate Club edge list (undirected)
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
            (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
            (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
            (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
            (3, 7), (3, 12), (3, 13),
            (4, 6), (4, 10),
            (5, 6), (5, 10), (5, 16),
            (6, 16),
            (8, 30), (8, 32), (8, 33),
            (9, 33),
            (11, 30),
            (12, 33),
            (13, 33),
            (14, 32), (14, 33),
            (15, 32), (15, 33),
            (16, 32), (16, 33),
            (17, 30), (17, 32), (17, 33),
            (18, 32), (18, 33),
            (19, 33),
            (20, 32), (20, 33),
            (21, 30), (21, 32), (21, 33),
            (22, 32), (22, 33),
            (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
            (24, 25), (24, 27), (24, 31),
            (25, 31),
            (26, 29), (26, 33),
            (27, 33),
            (28, 31), (28, 33),
            (29, 32), (29, 33),
            (30, 32), (30, 33),
            (31, 32), (31, 33),
            (32, 33)
        ]
        
        # Create adjacency matrix
        A = np.zeros((34, 34), dtype=int)
        for i, j in edges:
            A[i, j] = 1
            A[j, i] = 1  # Make symmetric
        
        return A
    
    def generate_network(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate the Zachary's Karate Club network.
        
        Args:
            random_seed: Random seed (ignored for this fixed network)
            
        Returns:
            Adjacency matrix
        """
        return self.adjacency_matrix.copy()
    
    def get_n_agents(self) -> int:
        """Get number of agents in the network."""
        return self.n_agents
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get information about the network.
        
        Returns:
            Dictionary with network statistics
        """
        A = self.adjacency_matrix
        n_agents = self.n_agents
        n_edges = np.sum(A) // 2  # Undirected graph
        
        # Calculate degree statistics
        degrees = np.sum(A, axis=1)
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        min_degree = np.min(degrees)
        
        # Calculate clustering coefficient
        clustering = self._calculate_clustering_coefficient(A)
        
        return {
            "n_agents": n_agents,
            "n_edges": n_edges,
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "min_degree": min_degree,
            "clustering_coefficient": clustering,
            "density": n_edges / (n_agents * (n_agents - 1) / 2),
            "network_type": "Zachary's Karate Club (Empirical)"
        }
    
    def _calculate_clustering_coefficient(self, A: np.ndarray) -> float:
        """Calculate the clustering coefficient of the network."""
        n = A.shape[0]
        total_clustering = 0.0
        
        for i in range(n):
            neighbors = np.where(A[i] == 1)[0]
            k = len(neighbors)
            
            if k < 2:
                continue
                
            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for k_idx in range(j + 1, len(neighbors)):
                    if A[neighbors[j], neighbors[k_idx]] == 1:
                        triangles += 1
            
            # Clustering coefficient for node i
            max_possible = k * (k - 1) / 2
            if max_possible > 0:
                total_clustering += triangles / max_possible
        
        return total_clustering / n
