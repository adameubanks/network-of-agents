"""
Simplified data storage for simulation results.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import json
import os


class DataStorage:
    """
    Simplified storage for simulation data.
    """
    
    def __init__(self):
        """Initialize data storage."""
        self.opinion_history = []
        self.adjacency_history = []
        self.initialized = False
        self.n_agents = 0
        self.num_timesteps = 0
    
    def initialize(self, n_agents: int, num_timesteps: int):
        """
        Initialize storage for simulation.
        
        Args:
            n_agents: Number of agents
            num_timesteps: Number of timesteps
        """
        self.n_agents = n_agents
        self.num_timesteps = num_timesteps
        
        # Initialize history arrays
        self.opinion_history = []
        self.adjacency_history = []
        
        # Pre-allocate arrays
        for _ in range(num_timesteps + 1):
            self.opinion_history.append(None)
            self.adjacency_history.append(None)
        
        self.initialized = True
    
    def store_timestep(self, timestep: int, opinions: np.ndarray, adjacency: np.ndarray):
        """
        Store timestep data.
        
        Args:
            timestep: Current timestep
            opinions: Opinion vector
            adjacency: Adjacency matrix
        """
        if not self.initialized:
            return
        
        # Ensure we have enough space
        while len(self.opinion_history) <= timestep:
            self.opinion_history.append(None)
        while len(self.adjacency_history) <= timestep:
            self.adjacency_history.append(None)
        
        self.opinion_history[timestep] = opinions.copy()
        self.adjacency_history[timestep] = adjacency.copy()
    
    def get_opinion_history(self) -> List[np.ndarray]:
        """
        Get opinion history.
        
        Returns:
            List of opinion vectors over time
        """
        return [op for op in self.opinion_history if op is not None]
    
    def get_adjacency_history(self) -> List[np.ndarray]:
        """
        Get adjacency history.
        
        Returns:
            List of adjacency matrices over time
        """
        return [adj for adj in self.adjacency_history if adj is not None]
    
    def get_simulation_results(self) -> Dict[str, Any]:
        """
        Get complete simulation results.
        
        Returns:
            Dictionary containing all simulation data
        """
        return {
            'opinion_history': [op.tolist() if op is not None else [] for op in self.opinion_history],
            'adjacency_history': [adj.tolist() if adj is not None else [] for adj in self.adjacency_history],
            'n_agents': self.n_agents,
            'num_timesteps': self.num_timesteps
        }
    
    def save_to_file(self, filename: str):
        """
        Save simulation data to file.
        
        Args:
            filename: Output filename
        """
        results = self.get_simulation_results()
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2) 