"""
Base classes and interfaces for opinion dynamics models.

This module provides abstract base classes that define the interfaces
for different components of the opinion dynamics simulation system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# OpinionDynamicsModel removed - using math functions directly

class NetworkTopology(ABC):
    """
    Abstract base class for network topologies.
    
    This defines the interface for generating different types of networks.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the network topology.
        
        Args:
            parameters: Topology-specific parameters
        """
        self.parameters = parameters
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_network(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a network adjacency matrix.
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            Adjacency matrix (shape: n_agents x n_agents)
        """
        pass
    
    @abstractmethod
    def get_n_agents(self) -> int:
        """
        Get the number of agents in the network.
        
        Returns:
            Number of agents
        """
        pass
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get information about the generated network.
        
        Returns:
            Dictionary with network statistics
        """
        adjacency = self.generate_network()
        n_agents = self.get_n_agents()
        n_edges = int(np.sum(adjacency) / 2)
        density = n_edges / (n_agents * (n_agents - 1)) if n_agents > 1 else 0
        avg_degree = np.mean(np.sum(adjacency, axis=1))
        
        return {
            "n_agents": n_agents,
            "n_edges": n_edges,
            "density": density,
            "avg_degree": avg_degree,
            "topology": self.name
        }

class OpinionGenerator(ABC):
    """
    Abstract base class for opinion generation methods.
    
    This defines the interface for generating agent opinions, whether through
    pure mathematical computation or LLM-based generation.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the opinion generator.
        
        Args:
            parameters: Generator-specific parameters
        """
        self.parameters = parameters
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_opinion(self, 
                        agent_id: int,
                        current_opinion: float,
                        topic: Dict[str, Any],
                        neighbor_posts: List[str] = None) -> float:
        """
        Generate an opinion for an agent.
        
        Args:
            agent_id: ID of the agent
            current_opinion: Current opinion value
            topic: Topic information (statement_a, statement_b, etc.)
            neighbor_posts: Posts from neighboring agents (for LLM methods)
            
        Returns:
            New opinion value in [-1, 1] range
        """
        pass
    
    @abstractmethod
    def generate_post(self, 
                     agent_id: int,
                     current_opinion: float,
                     topic: Dict[str, Any],
                     neighbor_posts: List[str] = None) -> str:
        """
        Generate a post expressing the agent's opinion.
        
        Args:
            agent_id: ID of the agent
            current_opinion: Current opinion value
            topic: Topic information
            neighbor_posts: Posts from neighboring agents
            
        Returns:
            Generated post text
        """
        pass

class OpinionRater(ABC):
    """
    Abstract base class for opinion rating methods.
    
    This defines the interface for rating posts, whether through
    pure mathematical computation or LLM-based rating.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the opinion rater.
        
        Args:
            parameters: Rater-specific parameters
        """
        self.parameters = parameters
        self.name = self.__class__.__name__
    
    @abstractmethod
    def rate_post(self, 
                 rater_agent_id: int,
                 rater_opinion: float,
                 poster_agent_id: int,
                 post: str,
                 topic: Dict[str, Any]) -> float:
        """
        Rate a post from another agent.
        
        Args:
            rater_agent_id: ID of the agent doing the rating
            rater_opinion: Current opinion of the rater
            poster_agent_id: ID of the agent who wrote the post
            post: Post text to rate
            topic: Topic information
            
        Returns:
            Rating value in [-1, 1] range
        """
        pass

class ExperimentRunner(ABC):
    """
    Abstract base class for experiment runners.
    
    This defines the interface for running different types of experiments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def run_experiment(self, 
                      model: str,
                      topology: NetworkTopology,
                      opinion_generator: OpinionGenerator,
                      opinion_rater: OpinionRater,
                      topic: Dict[str, Any],
                      **kwargs) -> Dict[str, Any]:
        """
        Run a single experiment.
        
        Args:
            model: Opinion dynamics model to use
            topology: Network topology to use
            opinion_generator: Opinion generation method
            opinion_rater: Opinion rating method
            topic: Topic for the experiment
            **kwargs: Additional experiment parameters
            
        Returns:
            Experiment results
        """
        pass
    
    @abstractmethod
    def run_batch(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a batch of experiments.
        
        Args:
            experiments: List of experiment configurations
            
        Returns:
            Batch results
        """
        pass

