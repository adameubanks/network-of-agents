"""
Network graph model for managing the social network structure.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from ..llm.agent import LLMAgent


class NetworkModel:
    """
    Manages the social network graph structure and provides analysis methods.
    """
    
    def __init__(self, n_agents: int, initial_connection_probability: float = 0.2):
        """
        Initialize the network model.
        
        Args:
            n_agents: Number of agents in the network
            initial_connection_probability: Probability of initial connections
        """
        self.n_agents = n_agents
        self.initial_connection_probability = initial_connection_probability
        self.adjacency_matrix = self._initialize_adjacency_matrix()
        self.agents: List[LLMAgent] = []
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
    
    def add_agents(self, agents: List[LLMAgent]):
        """
        Add agents to the network.
        
        Args:
            agents: List of LLM agents
        """
        if len(agents) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agents, got {len(agents)}")
        
        self.agents = agents
    
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
    
    def get_echo_chambers(self, similarity_threshold: float = 0.8) -> List[List[int]]:
        """
        Detect echo chambers based on opinion similarity.
        
        Args:
            similarity_threshold: Minimum similarity for echo chamber detection
            
        Returns:
            List of echo chambers (each chamber is a list of agent IDs)
        """
        if not self.agents:
            return []
        
        # Create similarity graph
        similarity_matrix = np.zeros((self.n_agents, self.n_agents))
        
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                similarity = self.agents[i].get_similarity_to(self.agents[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Create graph where edges exist if both connected and similar
        echo_chamber_graph = (self.adjacency_matrix > 0) & (similarity_matrix > similarity_threshold)
        
        # Find connected components in this graph
        G = nx.from_numpy_array(echo_chamber_graph.astype(int))
        components = list(nx.connected_components(G))
        
        return [list(comp) for comp in components if len(comp) > 1]
    
    def get_agent_degrees(self) -> Dict[int, int]:
        """
        Get the degree of each agent.
        
        Returns:
            Dictionary mapping agent ID to degree
        """
        degrees = np.sum(self.adjacency_matrix, axis=1)
        return {i: int(degrees[i]) for i in range(self.n_agents)}
    
    def get_network_evolution_metrics(self) -> Dict[str, List[float]]:
        """
        Get network evolution metrics over time.
        
        Returns:
            Dictionary containing lists of metrics over time
        """
        metrics = {
            'density': [],
            'average_degree': [],
            'clustering_coefficient': [],
            'num_components': []
        }
        
        for adj_matrix in self.network_history + [self.adjacency_matrix]:
            # Calculate density
            total_edges = np.sum(adj_matrix) / 2
            max_possible_edges = self.n_agents * (self.n_agents - 1) / 2
            density = total_edges / max_possible_edges
            metrics['density'].append(density)
            
            # Calculate average degree
            degrees = np.sum(adj_matrix, axis=1)
            avg_degree = np.mean(degrees)
            metrics['average_degree'].append(avg_degree)
            
            # Calculate clustering coefficient
            try:
                G = nx.from_numpy_array(adj_matrix)
                clustering = nx.average_clustering(G)
                metrics['clustering_coefficient'].append(clustering)
            except:
                metrics['clustering_coefficient'].append(0.0)
            
            # Calculate number of components
            G = nx.from_numpy_array(adj_matrix)
            num_components = nx.number_connected_components(G)
            metrics['num_components'].append(num_components)
        
        return metrics
    
    def to_networkx(self) -> nx.Graph:
        """
        Convert to NetworkX graph for advanced analysis.
        
        Returns:
            NetworkX graph object
        """
        G = nx.from_numpy_array(self.adjacency_matrix)
        
        # Add agent attributes
        for i, agent in enumerate(self.agents):
            G.nodes[i]['agent_id'] = agent.agent_id
            G.nodes[i]['persona'] = agent.persona
            G.nodes[i]['opinions'] = agent.get_opinions().tolist()
            G.nodes[i]['degree'] = agent.get_degree(self.adjacency_matrix)
        
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
            'network_history': [adj.tolist() for adj in self.network_history],
            'agents': [agent.to_dict() for agent in self.agents]
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
        network.agents = [LLMAgent.from_dict(agent_data) for agent_data in data['agents']]
        return network 