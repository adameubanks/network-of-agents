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
                 random_seed: Optional[int] = None):
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
        self.topics = topics or ["Climate Change"]
        self.random_seed = random_seed
        
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
    
    def _initialize_agents(self) -> List[Agent]:
        """
        Initialize agents with diverse initial opinions.
        
        Returns:
            List of initialized agents
        """
        agents = []
        
        # Create diverse initial opinions
        for i in range(self.n_agents):
            # Create clusters of opinions for diversity
            cluster_id = i % 3  # 3 clusters
            base_opinion = cluster_id / 2.0  # Spread across [0, 1]
            noise = (np.random.random() - 0.5) * 0.3
            initial_opinion = np.clip(base_opinion + noise, 0.0, 1.0)
            
            agent = Agent(agent_id=i, initial_opinion=initial_opinion)
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
        self._store_current_state()
        
        # Main simulation loop
        iterator = tqdm(range(self.num_timesteps), desc="Running simulation") if progress_bar else range(self.num_timesteps)
        
        for timestep in iterator:
            self.current_timestep = timestep
            
            # Step 1: Generate all posts in batch
            current_topic = self.topics[0]  # Use single topic for entire simulation
            agent_ids = [agent.agent_id for agent in self.agents]
            posts = self.llm_client.generate_posts_batch(current_topic, agent_ids)
            
            # Step 2: Interpret all posts in batch
            if self.llm_client is not None:
                # Create a list of all posts that need to be interpreted by each agent
                all_interpretations = []
                for agent in self.agents:
                    # Each agent interprets all posts
                    interpretations = self.llm_client.interpret_posts_batch(posts, current_topic)
                    all_interpretations.append(interpretations)
            else:
                # Fallback: simple interpretation based on keywords
                all_interpretations = []
                for agent in self.agents:
                    opinions = []
                    for post in posts:
                        if "strongly support" in post.lower():
                            opinions.append(0.9)
                        elif "support" in post.lower():
                            opinions.append(0.7)
                        elif "somewhat oppose" in post.lower():
                            opinions.append(0.3)
                        elif "strongly oppose" in post.lower():
                            opinions.append(0.1)
                        else:
                            opinions.append(0.5)
                    all_interpretations.append(opinions)
            
            # Step 3: Update opinions using mathematical framework
            X_current = self._get_opinion_matrix()
            A_current = self.network.get_adjacency_matrix()
            
            # Use interpreted opinions as input to mathematical dynamics
            # Take the diagonal of interpreted opinions (each agent's interpretation of their own post)
            X_interpreted = np.array([all_interpretations[i][i] for i in range(len(all_interpretations))])
            X_next = update_opinions(X_interpreted, A_current, self.epsilon)
            
            # Step 4: Update agent opinions
            self._update_agent_opinions(X_next)
            
            # Step 5: Update network topology based on opinion similarity
            A_next = update_edges(A_current, X_next, self.theta, self.epsilon)
            self.network.update_adjacency_matrix(A_next)
            
            # Step 6: Store current state
            self._store_current_state()
        
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
            new_opinions: New opinion vector
        """
        for i, agent in enumerate(self.agents):
            agent.update_opinion(new_opinions[i])
    
    def _store_current_state(self):
        """Store current simulation state."""
        opinions = self._get_opinion_matrix()
        self.opinion_history.append(opinions.copy())
        
        # Calculate mean and standard deviation
        mean_opinion = np.mean(opinions)
        std_opinion = np.std(opinions)
        
        self.mean_opinions.append(mean_opinion)
        self.std_opinions.append(std_opinion)
    
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