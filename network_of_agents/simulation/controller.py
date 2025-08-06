"""
Streamlined simulation controller for the network of agents.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time

from ..agent import Agent
from ..llm_client import LLMClient
from ..network.graph_model import NetworkModel
from ..core.mathematics import update_opinions, update_edges


class Controller:
    """
    Streamlined controller for the network of agents simulation.
    """
    
    def __init__(self, 
                 llm_client: LLMClient,
                 n_agents: int = 10,
                 epsilon: float = 0.001,
                 theta: int = 7,
                 num_timesteps: int = 50,
                 initial_connection_probability: float = 0.2,
                 topics: Optional[List[str]] = None,
                 random_seed: Optional[int] = None,
                 generation_temperature: float = 0.9,
                 rating_temperature: float = 0.1):
        """
        Initialize simulation controller.
        
        Args:
            n_agents: Number of agents in the network
            epsilon: Small positive parameter for numerical stability
            theta: Positive integer parameter for edge formation
            num_timesteps: Number of simulation timesteps
            initial_connection_probability: Initial probability of connections
            llm_client: LLM client for post generation and interpretation
            topics: List of topics for opinions
            random_seed: Random seed for reproducible results
            generation_temperature: Temperature for post generation (default: 0.9)
            rating_temperature: Temperature for opinion rating (default: 0.1)
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.n_agents = n_agents
        self.epsilon = epsilon
        self.theta = theta
        self.num_timesteps = num_timesteps
        self.initial_connection_probability = initial_connection_probability
        self.llm_client = llm_client
        self.topics = topics
        self.random_seed = random_seed
        self.generation_temperature = generation_temperature
        self.rating_temperature = rating_temperature
        
        # Update LLM client temperatures from config
        self.llm_client.generation_temperature = generation_temperature
        self.llm_client.rating_temperature = rating_temperature
        
        # Initialize components
        self.network = NetworkModel(n_agents, initial_connection_probability, random_seed)
        self.agents = self._initialize_agents()
        
        # Simulation state
        self.is_running = False
        self.current_timestep = 0
        
        # Data storage
        self.opinion_history = []
        self.mean_opinions = []
        self.std_opinions = []
        self.posts_history = []
        self.interpretations_history = []
    
    def _initialize_agents(self) -> List[Agent]:
        """
        Initialize agents with random opinions.
        
        Returns:
            List of initialized agents
        """
        agents = []
        
        # Create agents with random opinions
        for i in range(self.n_agents):
            # Create agent with random opinion
            agent = Agent(agent_id=i)
            agents.append(agent)
        
        return agents
    
    def run_simulation(self, progress_bar: bool = True) -> Dict[str, Any]:
        """
        Run the simulation with LLM post generation and interpretation.
        
        Args:
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary containing simulation results
        """
        self.is_running = True
        self.current_timestep = 0
        
        # Store initial state
        self._store_current_state([], [])
        
        # Main simulation loop
        iterator = tqdm(range(self.num_timesteps), desc="Running simulation") if progress_bar else range(self.num_timesteps)
        
        for timestep in iterator:
            self.current_timestep = timestep
            
            # Step 1: Generate all posts using agent-specific prompting
            current_topic = self.topics[0]  # Use single topic for entire simulation
            posts = self.llm_client.generate_posts_for_agents(current_topic, self.agents)
            
            # Step 2: Interpret own posts only (optimized)
            self_interpretations = self.llm_client.interpret_posts_for_agents(posts, current_topic, self.agents)
            
            # Debug: Print first few timesteps
            if timestep < 3:
                print(f"Timestep {timestep}:")
                print(f"  Posts: {posts[:3]}...")
                print(f"  Self-interpretations: {[f'{x:.3f}' for x in self_interpretations[:3]]}...")
                print(f"  Number of agents: {len(self.agents)}")
                print(f"  Number of posts: {len(posts)}")
            
            # Step 3: Update opinions using mathematical framework
            X_current = self._get_opinion_matrix()
            A_current = self.network.get_adjacency_matrix()
            
            # Use self-interpretations directly (no diagonal extraction needed)
            X_interpreted = np.array(self_interpretations)
            X_next = update_opinions(X_interpreted, A_current, self.epsilon)
            
            # Debug: Print first few timesteps
            if timestep < 3:
                print(f"  Current opinions: {[f'{x:.3f}' for x in X_current[:3]]}...")
                print(f"  Self-interpreted opinions: {[f'{x:.3f}' for x in X_interpreted[:3]]}...")
                print(f"  Next opinions: {[f'{x:.3f}' for x in X_next[:3]]}...")
                print(f"  Network density: {self.network.get_network_density():.3f}")
                print()
            
            # Step 4: Update agent opinions
            self._update_agent_opinions(X_next)
            
            # Step 5: Update network topology based on opinion similarity
            A_next = update_edges(A_current, X_next, self.theta, self.epsilon)
            self.network.update_adjacency_matrix(A_next)
            
            # Step 6: Store current state
            self._store_current_state(posts, self_interpretations)
        
        self.is_running = False
        
        return self._get_simulation_results()
    
    def _get_opinion_matrix(self) -> np.ndarray:
        """
        Get current opinion matrix.
        
        Returns:
            Current opinion vector
        """
        opinions = []
        for agent in self.agents:
            opinions.append(agent.get_opinion())
        return np.array(opinions)
    
    def _update_agent_opinions(self, new_opinions: np.ndarray):
        """
        Update all agent opinions.
        
        Args:
            new_opinions: New opinion vector (-1 to 1)
        """
        for i, agent in enumerate(self.agents):
            agent.update_opinion(new_opinions[i])
    
    def _store_current_state(self, posts: List[str], interpretations: List[float]):
        """Store current simulation state."""
        opinions = self._get_opinion_matrix()
        self.opinion_history.append(opinions.copy())
        
        # Calculate mean and standard deviation
        mean_opinion = np.mean(opinions)
        std_opinion = np.std(opinions)
        
        self.mean_opinions.append(mean_opinion)
        self.std_opinions.append(std_opinion)
        
        # Only store posts and interpretations if they exist (not for initial state)
        if posts:
            self.posts_history.append(posts)
        else:
            self.posts_history.append([])
            
        if interpretations:
            self.interpretations_history.append([interpretations])  # Wrap in list for compatibility
        else:
            self.interpretations_history.append([])
    
    def _get_simulation_results(self) -> Dict[str, Any]:
        """
        Get simulation results.
        
        Returns:
            Dictionary containing simulation results
        """
        return {
            'opinion_history': [op.tolist() for op in self.opinion_history],
            'mean_opinions': self.mean_opinions,
            'std_opinions': self.std_opinions,
            'posts_history': self.posts_history,
            'interpretations_history': self.interpretations_history,
            'final_opinions': self._get_opinion_matrix().tolist(),
            'final_adjacency': self.network.get_adjacency_matrix().tolist(),
            'simulation_params': {
                'n_agents': self.n_agents,
                'num_timesteps': self.num_timesteps,
                'epsilon': self.epsilon,
                'theta': self.theta,
                'initial_connection_probability': self.initial_connection_probability
            },
            'random_seed': self.random_seed,
            'topics': self.topics
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current simulation state.
        
        Returns:
            Dictionary containing current state
        """
        return {
            'timestep': self.current_timestep,
            'opinions': self._get_opinion_matrix().tolist(),
            'adjacency': self.network.get_adjacency_matrix().tolist(),
            'is_running': self.is_running,
            'mean_opinion': self.mean_opinions[-1] if self.mean_opinions else 0.0,
            'std_opinion': self.std_opinions[-1] if self.std_opinions else 0.0
        } 