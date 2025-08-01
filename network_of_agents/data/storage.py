"""
Data storage and management for simulation results.
"""

import numpy as np
import json
import pickle
from typing import Dict, List, Any, Optional
import os
from datetime import datetime


class SimulationDataStorage:
    """
    Manages storage and retrieval of simulation data.
    """
    
    def __init__(self):
        """Initialize the data storage."""
        self.opinion_history = []
        self.adjacency_history = []
        self.metrics_history = []
        self.agent_data_history = []
        self.parameters = {}
        self.initialized = False
    
    def initialize(self, n_agents: int, n_topics: int, num_timesteps: int):
        """
        Initialize storage for a new simulation.
        
        Args:
            n_agents: Number of agents
            n_topics: Number of topics
            num_timesteps: Number of timesteps
        """
        self.n_agents = n_agents
        self.n_topics = n_topics
        self.num_timesteps = num_timesteps
        
        # Pre-allocate storage
        self.opinion_history = []
        self.adjacency_history = []
        self.metrics_history = []
        self.agent_data_history = []
        
        self.initialized = True
    
    def store_opinions(self, timestep: int, opinion_matrix: np.ndarray):
        """
        Store opinion matrix for a timestep.
        
        Args:
            timestep: Current timestep
            opinion_matrix: Opinion matrix to store
        """
        if not self.initialized:
            raise RuntimeError("Storage not initialized. Call initialize() first.")
        
        # Ensure we have space for this timestep
        while len(self.opinion_history) <= timestep:
            self.opinion_history.append(None)
        
        self.opinion_history[timestep] = opinion_matrix.copy()
    
    def store_adjacency(self, timestep: int, adjacency_matrix: np.ndarray):
        """
        Store adjacency matrix for a timestep.
        
        Args:
            timestep: Current timestep
            adjacency_matrix: Adjacency matrix to store
        """
        if not self.initialized:
            raise RuntimeError("Storage not initialized. Call initialize() first.")
        
        # Ensure we have space for this timestep
        while len(self.adjacency_history) <= timestep:
            self.adjacency_history.append(None)
        
        self.adjacency_history[timestep] = adjacency_matrix.copy()
    
    def store_metrics(self, timestep: int, metrics: Dict[str, float]):
        """
        Store network metrics for a timestep.
        
        Args:
            timestep: Current timestep
            metrics: Dictionary of metrics to store
        """
        if not self.initialized:
            raise RuntimeError("Storage not initialized. Call initialize() first.")
        
        # Ensure we have space for this timestep
        while len(self.metrics_history) <= timestep:
            self.metrics_history.append(None)
        
        self.metrics_history[timestep] = metrics.copy()
    
    def store_agent_data(self, timestep: int, agent_data: List[Dict[str, Any]]):
        """
        Store agent-specific data for a timestep.
        
        Args:
            timestep: Current timestep
            agent_data: List of agent data dictionaries
        """
        if not self.initialized:
            raise RuntimeError("Storage not initialized. Call initialize() first.")
        
        # Ensure we have space for this timestep
        while len(self.agent_data_history) <= timestep:
            self.agent_data_history.append(None)
        
        self.agent_data_history[timestep] = agent_data.copy()
    
    def get_opinion_history(self) -> List[np.ndarray]:
        """
        Get the complete opinion history.
        
        Returns:
            List of opinion matrices over time
        """
        return [op.copy() if op is not None else None for op in self.opinion_history]
    
    def get_adjacency_history(self) -> List[np.ndarray]:
        """
        Get the complete adjacency matrix history.
        
        Returns:
            List of adjacency matrices over time
        """
        return [adj.copy() if adj is not None else None for adj in self.adjacency_history]
    
    def get_metrics_history(self) -> List[Dict[str, float]]:
        """
        Get the complete metrics history.
        
        Returns:
            List of metrics dictionaries over time
        """
        return [metrics.copy() if metrics is not None else None for metrics in self.metrics_history]
    
    def get_agent_data_history(self) -> List[List[Dict[str, Any]]]:
        """
        Get the complete agent data history.
        
        Returns:
            List of agent data lists over time
        """
        return [data.copy() if data is not None else None for data in self.agent_data_history]
    
    def get_simulation_results(self) -> Dict[str, Any]:
        """
        Get complete simulation results.
        
        Returns:
            Dictionary containing all simulation data
        """
        return {
            'parameters': {
                'n_agents': self.n_agents,
                'n_topics': self.n_topics,
                'num_timesteps': self.num_timesteps
            },
            'opinion_history': [op.tolist() if op is not None else None for op in self.opinion_history],
            'adjacency_history': [adj.tolist() if adj is not None else None for adj in self.adjacency_history],
            'metrics_history': self.metrics_history,
            'agent_data_history': self.agent_data_history
        }
    
    def save_to_file(self, filename: str, data: Dict[str, Any]):
        """
        Save simulation data to a file.
        
        Args:
            filename: Name of the file to save to
            data: Data to save
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Determine file format based on extension
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif filename.endswith('.pkl'):
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        else:
            # Default to JSON
            filename = filename + '.json' if not filename.endswith('.json') else filename
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
    
    def load_from_file(self, filename: str) -> Dict[str, Any]:
        """
        Load simulation data from a file.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            Loaded simulation data
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        
        # Determine file format based on extension
        if filename.endswith('.json'):
            with open(filename, 'r') as f:
                data = json.load(f)
        elif filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        return data
    
    def load_simulation_data(self, simulation_data: Dict[str, Any]):
        """
        Load simulation data into storage.
        
        Args:
            simulation_data: Simulation data dictionary
        """
        # Load parameters
        params = simulation_data['parameters']
        self.n_agents = params['n_agents']
        self.n_topics = params['n_topics']
        self.num_timesteps = params['num_timesteps']
        
        # Load history data
        self.opinion_history = [np.array(op) if op is not None else None 
                               for op in simulation_data['opinion_history']]
        self.adjacency_history = [np.array(adj) if adj is not None else None 
                                 for adj in simulation_data['adjacency_history']]
        self.metrics_history = simulation_data['metrics_history']
        self.agent_data_history = simulation_data['agent_data_history']
        
        self.initialized = True
    
    def export_to_csv(self, output_dir: str, prefix: str = "simulation"):
        """
        Export simulation data to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
            prefix: Prefix for file names
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export opinion history
        opinion_data = []
        for timestep, opinions in enumerate(self.opinion_history):
            if opinions is not None:
                for agent_id in range(self.n_agents):
                    for topic_id in range(self.n_topics):
                        opinion_data.append({
                            'timestep': timestep,
                            'agent_id': agent_id,
                            'topic_id': topic_id,
                            'opinion': opinions[agent_id, topic_id]
                        })
        
        import pandas as pd
        opinion_df = pd.DataFrame(opinion_data)
        opinion_df.to_csv(os.path.join(output_dir, f"{prefix}_opinions.csv"), index=False)
        
        # Export metrics history
        metrics_data = []
        for timestep, metrics in enumerate(self.metrics_history):
            if metrics is not None:
                row = {'timestep': timestep}
                row.update(metrics)
                metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(output_dir, f"{prefix}_metrics.csv"), index=False)
        
        # Export network density over time
        density_data = []
        for timestep, adj_matrix in enumerate(self.adjacency_history):
            if adj_matrix is not None:
                total_edges = np.sum(adj_matrix) / 2
                max_possible_edges = self.n_agents * (self.n_agents - 1) / 2
                density = total_edges / max_possible_edges
                density_data.append({
                    'timestep': timestep,
                    'density': density,
                    'total_edges': total_edges
                })
        
        density_df = pd.DataFrame(density_data)
        density_df.to_csv(os.path.join(output_dir, f"{prefix}_network_density.csv"), index=False)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the simulation.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.initialized or not self.opinion_history:
            return {}
        
        # Calculate opinion convergence
        initial_opinions = self.opinion_history[0]
        final_opinions = self.opinion_history[-1]
        
        if initial_opinions is not None and final_opinions is not None:
            opinion_change = np.mean(np.abs(final_opinions - initial_opinions))
            opinion_variance = np.var(final_opinions, axis=0)
        else:
            opinion_change = 0.0
            opinion_variance = np.zeros(self.n_topics)
        
        # Calculate network evolution
        if self.adjacency_history:
            initial_density = self.metrics_history[0]['density'] if self.metrics_history[0] else 0.0
            final_density = self.metrics_history[-1]['density'] if self.metrics_history[-1] else 0.0
            density_change = final_density - initial_density
        else:
            density_change = 0.0
        
        return {
            'total_timesteps': len(self.opinion_history),
            'n_agents': self.n_agents,
            'n_topics': self.n_topics,
            'average_opinion_change': opinion_change,
            'final_opinion_variance': opinion_variance.tolist(),
            'network_density_change': density_change,
            'final_network_density': self.metrics_history[-1]['density'] if self.metrics_history[-1] else 0.0,
            'final_clustering_coefficient': self.metrics_history[-1]['clustering_coefficient'] if self.metrics_history[-1] else 0.0,
            'final_num_components': self.metrics_history[-1]['num_components'] if self.metrics_history[-1] else 0,
            'final_echo_chambers': self.metrics_history[-1]['echo_chambers'] if self.metrics_history[-1] else 0
        } 