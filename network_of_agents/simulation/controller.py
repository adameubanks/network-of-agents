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
                 llm_client: Optional[LLMClient] = None,
                 n_agents: int = 10,
                 epsilon: float = 0.001,
                 theta: int = 3,
                 num_timesteps: int = 50,
                 initial_connection_probability: float = 0.3,
                 topics: Optional[List[str]] = None,
                 random_seed: Optional[int] = None,
                 generation_temperature: float = 0.9,
                 rating_temperature: float = 0.1,
                 llm_enabled: bool = True,
                 noise_enabled: bool = False,
                 noise_mean: float = 0.0,
                 noise_std: float = 0.0,
                 noise_clip: bool = True):
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
        self.llm_enabled = llm_enabled
        self.noise_enabled = noise_enabled
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        
        # Update LLM client temperatures from config
        if self.llm_enabled and self.llm_client is not None:
            self.llm_client.generation_temperature = generation_temperature
            self.llm_client.rating_temperature = rating_temperature
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
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
            agent = Agent(agent_id=i, random_seed=self.random_seed)
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
        if self.llm_enabled:
            self._store_current_state([], [])
        else:
            self._store_current_state(None, None)
        
        # Main simulation loop
        iterator = tqdm(range(self.num_timesteps), desc="Running simulation") if progress_bar else range(self.num_timesteps)
        
        try:
            for timestep in iterator:
                self.current_timestep = timestep
                
                # Step 1/2: Generate posts and interpretations (LLM) or skip (no-LLM)
                current_topic = self.topics[0]  # Use single topic for entire simulation
                if self.llm_enabled:
                    posts = self.llm_client.generate_posts_for_agents(current_topic, self.agents)
                    self_interpretations = self.llm_client.interpret_posts_for_agents(posts, current_topic, self.agents)
                else:
                    posts = None
                    self_interpretations = None

                # Step 3: Update opinions using mathematical framework
                X_current = self._get_opinion_matrix()
                A_current = self.network.get_adjacency_matrix()

                if self.llm_enabled:
                    X_interpreted = np.array(self_interpretations)
                else:
                    # No-LLM: self-perception equals current opinions, with optional Gaussian noise
                    X_interpreted = X_current.copy()
                    if self.noise_enabled and self.noise_std > 0.0:
                        noise = np.random.normal(self.noise_mean, self.noise_std, size=self.n_agents)
                        X_interpreted = X_interpreted + noise
                        if self.noise_clip:
                            X_interpreted = np.clip(X_interpreted, -1.0, 1.0)

                # Convert to math domain [0, 1] just-in-time for opinion dynamics
                X_for_math = self._to_math_domain(X_interpreted)

                # Compute next opinions in math domain
                X_next_math = update_opinions(X_for_math, A_current, self.epsilon)

                # Step 4: Update network topology A[k+1] based on X[k] (math-domain opinions)
                A_next = update_edges(A_current, X_for_math, self.theta, self.epsilon)
                self.network.update_adjacency_matrix(A_next)

                # Step 5: Convert back to agent domain [-1, 1] and update agent opinions
                X_next_agent = self._to_agent_domain(X_next_math)
                self._update_agent_opinions(X_next_agent)

                # Step 6: Store current state
                if self.llm_enabled:
                    self._store_current_state(posts, self_interpretations)
                else:
                    self._store_current_state(None, None)
            
            self.is_running = False
            return self._get_simulation_results()
            
        except Exception as e:
            # Save partial results when an error occurs
            self.is_running = False
            print(f"\nSimulation interrupted at timestep {self.current_timestep + 1}/{self.num_timesteps}")
            print(f"Error: {e}")
            print("Saving partial results...")
            
            partial_results = self._get_partial_simulation_results()
            partial_results['error'] = str(e)
            partial_results['interrupted_at_timestep'] = self.current_timestep + 1
            partial_results['completed_timesteps'] = len(self.mean_opinions)
            partial_results['total_timesteps'] = self.num_timesteps
            
            return partial_results
    
    def _get_partial_simulation_results(self) -> Dict[str, Any]:
        """
        Get partial simulation results when simulation is interrupted.
        
        Returns:
            Dictionary containing partial simulation results
        """
        results: Dict[str, Any] = {
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
                'initial_connection_probability': self.initial_connection_probability,
                'llm_enabled': self.llm_enabled,
                'noise_enabled': self.noise_enabled,
                'noise_mean': self.noise_mean,
                'noise_std': self.noise_std,
                'noise_clip': self.noise_clip,
            },
            'random_seed': self.random_seed,
            'topics': self.topics,
            'is_partial': True
        }
        if self.llm_enabled:
            results['posts_history'] = self.posts_history
            results['interpretations_history'] = self.interpretations_history
        return results
    
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

    def _to_math_domain(self, x: np.ndarray) -> np.ndarray:
        """Map agent-domain opinions x ∈ [-1, 1] to math-domain s ∈ [0, 1]."""
        return (x + 1.0) / 2.0

    def _to_agent_domain(self, s: np.ndarray) -> np.ndarray:
        """Map math-domain opinions s ∈ [0, 1] back to agent-domain x ∈ [-1, 1]."""
        return (2.0 * s) - 1.0
    
    def _update_agent_opinions(self, new_opinions: np.ndarray):
        """
        Update all agent opinions.
        
        Args:
            new_opinions: New opinion vector in agent domain [-1, 1]
        """
        for i, agent in enumerate(self.agents):
            agent.update_opinion(new_opinions[i])
    
    def _store_current_state(self, posts: Optional[List[str]], interpretations: Optional[List[float]]):
        """Store current simulation state."""
        opinions = self._get_opinion_matrix()
        self.opinion_history.append(opinions.copy())
        
        # Calculate mean and standard deviation
        mean_opinion = np.mean(opinions)
        std_opinion = np.std(opinions)
        
        self.mean_opinions.append(mean_opinion)
        self.std_opinions.append(std_opinion)
        
        # Only store posts and interpretations when provided (LLM mode only)
        if posts is not None:
            self.posts_history.append(posts)
        if interpretations is not None:
            # Wrap in list for compatibility with existing structure
            self.interpretations_history.append([interpretations])
    
    def _get_simulation_results(self) -> Dict[str, Any]:
        """
        Get simulation results.
        
        Returns:
            Dictionary containing simulation results
        """
        results: Dict[str, Any] = {
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
                'initial_connection_probability': self.initial_connection_probability,
                'llm_enabled': self.llm_enabled,
                'noise_enabled': self.noise_enabled,
                'noise_mean': self.noise_mean,
                'noise_std': self.noise_std,
                'noise_clip': self.noise_clip,
            },
            'random_seed': self.random_seed,
            'topics': self.topics
        }
        if self.llm_enabled:
            results['posts_history'] = self.posts_history
            results['interpretations_history'] = self.interpretations_history
        return results
    
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