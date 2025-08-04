"""
Simplified simulation controller for the network of agents.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time

from ..llm.agent import LLMAgent
from ..llm.litellm_client import LiteLLMClient
from ..network.graph_model import NetworkModel
from ..data.storage import DataStorage
from ..core.mathematics import update_opinions, update_edges


class SimulationController:
    """
    Simplified controller for the network of agents simulation.
    """
    
    def __init__(self, 
                 n_agents: int = 10,
                 epsilon: float = 0.001,
                 theta: int = 7,
                 num_timesteps: int = 50,
                 initial_connection_probability: float = 0.2,
                 llm_client: Optional[LiteLLMClient] = None,
                 topics: Optional[List[str]] = None,
                 initial_opinions: Optional[List[float]] = None,
                 random_seed: Optional[int] = None,
                 initial_opinion_diversity: Optional[float] = None):
        """
        Initialize simulation controller.
        
        Args:
            n_agents: Number of agents in the network
            epsilon: Small positive parameter for numerical stability
            theta: Positive integer parameter for edge formation
            num_timesteps: Number of simulation timesteps
            initial_connection_probability: Initial probability of connections
            llm_client: LLM client for opinion generation
            topics: List of topics for opinions
            initial_opinions: Initial opinion values for agents
            random_seed: Random seed for reproducible results
            initial_opinion_diversity: Factor controlling initial opinion diversity (0-1)
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
        self.topics = topics or self._generate_default_topics()
        self.initial_opinions = initial_opinions
        self.random_seed = random_seed
        self.initial_opinion_diversity = initial_opinion_diversity
        
        # Initialize components
        self.network = NetworkModel(n_agents, initial_connection_probability, random_seed)
        self.data_storage = DataStorage()
        self.agents = self._initialize_agents()
        
        # Simulation state
        self.is_running = False
        self.current_timestep = 0
    
    def _generate_default_topics(self) -> List[str]:
        """Generate default topics for the simulation."""
        return ["Topic 1"]
    
    def _generate_diverse_initial_opinions(self, n_agents: int, diversity_factor: float = 0.8) -> List[float]:
        """
        Generate diverse initial opinions for agents.
        
        Args:
            n_agents: Number of agents
            diversity_factor: Factor controlling opinion diversity (0-1)
            
        Returns:
            List of initial opinion values for each agent
        """
        opinions = []
        
        # Create opinion clusters to ensure diversity
        n_clusters = max(2, int(n_agents * 0.3))  # At least 2 clusters
        
        for i in range(n_agents):
            # Assign agents to clusters
            cluster_id = i % n_clusters
            
            # Generate opinion within cluster
            base_opinion = cluster_id / (n_clusters - 1)  # Spread clusters across [0, 1]
            noise = (np.random.random() - 0.5) * diversity_factor
            opinion = np.clip(base_opinion + noise, 0.0, 1.0)
            
            opinions.append(opinion)
        
        return opinions
    
    def _initialize_agents(self) -> List[LLMAgent]:
        """
        Initialize all agents.
        
        Returns:
            List of initialized agents
        """
        agents = []
        
        # Generate initial opinions if not provided
        if self.initial_opinions is None:
            if self.initial_opinion_diversity is not None:
                self.initial_opinions = self._generate_diverse_initial_opinions(
                    self.n_agents, self.initial_opinion_diversity
                )
            else:
                self.initial_opinions = np.random.random(self.n_agents).tolist()
        
        # Create agents
        for i in range(self.n_agents):
            agent = LLMAgent(
                agent_id=i,
                initial_opinions=[self.initial_opinions[i]],
                topics=self.topics,
                llm_client=self.llm_client
            )
            agents.append(agent)
        
        return agents
    
    def run_simulation(self, progress_bar: bool = True) -> Dict[str, Any]:
        """
        Run the simplified simulation with LLM encoding/decoding.
        
        Args:
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary containing simulation results
        """
        self.is_running = True
        self.current_timestep = 0
        
        # Initialize data storage
        self.data_storage.initialize(self.n_agents, self.num_timesteps)
        
        # Store initial state
        self._store_current_state()
        
        # Main simulation loop
        iterator = tqdm(range(self.num_timesteps), desc="Running simulation") if progress_bar else range(self.num_timesteps)
        
        for timestep in iterator:
            self.current_timestep = timestep
            
            # Step 1: Update opinions using mathematical framework
            A_current = self.network.get_adjacency_matrix()
            X_current = self._get_opinion_matrix()
            X_next = update_opinions(X_current, A_current, self.epsilon)
            
            # Step 2: Apply LLM encoding/decoding if available
            if self.llm_client is not None:
                X_next = self._apply_llm_encoding_decoding(X_next)
            
            self._update_agent_opinions(X_next)
            
            # Step 3: Update network topology based on opinion similarity
            A_next = update_edges(A_current, X_next, self.theta, self.epsilon)
            self.network.update_adjacency_matrix(A_next)
            
            # Step 4: Store current state
            self._store_current_state()
        
        self.is_running = False
        
        return self.data_storage.get_simulation_results()
    
    def _get_opinion_matrix(self) -> np.ndarray:
        """
        Get current opinion matrix.
        
        Returns:
            Current opinion vector
        """
        opinions = []
        for agent in self.agents:
            opinions.append(agent.get_opinions()[0])  # Single topic
        return np.array(opinions)
    
    def _update_agent_opinions(self, new_opinions: np.ndarray):
        """
        Update all agent opinions.
        
        Args:
            new_opinions: New opinion vector
        """
        for i, agent in enumerate(self.agents):
            agent.update_opinions(np.array([new_opinions[i]]))
    
    def _store_current_state(self):
        """Store current simulation state."""
        opinions = self._get_opinion_matrix()
        adjacency = self.network.get_adjacency_matrix()
        self.data_storage.store_timestep(self.current_timestep, opinions, adjacency)
    
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
            'is_running': self.is_running
        } 

    def _apply_llm_encoding_decoding(self, opinions: np.ndarray) -> np.ndarray:
        """
        Apply LLM encoding and decoding to opinions.
        
        Args:
            opinions: Current opinion vector
            
        Returns:
            Updated opinion vector after LLM processing
        """
        if self.llm_client is None:
            return opinions
        
        try:
            # Encode opinions to text
            encoded_texts = []
            for i, opinion in enumerate(opinions):
                if opinion > 0.7:
                    stance = "strongly support"
                elif opinion > 0.5:
                    stance = "support"
                elif opinion > 0.3:
                    stance = "somewhat oppose"
                else:
                    stance = "strongly oppose"
                
                text = f"Agent {i} {stance} {self.topics[0]} (opinion: {opinion:.2f})"
                encoded_texts.append(text)
            
            # Decode back to opinions using LLM
            decoded_opinions = []
            for i, text in enumerate(encoded_texts):
                try:
                    opinion = self.llm_client.interpret_text_to_opinions(text, self.topics)[0]
                    decoded_opinions.append(opinion)
                except Exception as e:
                    # If LLM fails, keep original opinion
                    decoded_opinions.append(opinions[i])
            
            return np.array(decoded_opinions)
            
        except Exception as e:
            # If any error occurs, return original opinions
            return opinions 