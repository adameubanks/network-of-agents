"""
Streamlined simulation controller for the network of agents.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from tqdm import tqdm
import time
import logging
import concurrent.futures as cf
import os
import json
import tempfile

from ..agent import Agent
from ..llm_client import LLMClient
from ..network.graph_model import NetworkModel
from ..core.mathematics import update_opinions_pure_degroot, create_connected_degroot_network

logger = logging.getLogger(__name__)

class Controller:
    """
    Streamlined controller for the network of agents simulation.
    """
    
    def __init__(self, 
                 llm_client: Optional[LLMClient] = None,
                 n_agents: int = 10,
                 epsilon: float = None,
                 connectivity: float = None,
                 num_timesteps: int = 50,
                 topics: Optional[List[str]] = None,
                 random_seed: Optional[int] = None,
                 llm_enabled: bool = True,
                 model: str = "degroot",
                 reversed: bool = False,
                 output_file: Optional[str] = None):
        """Initialize simulation controller."""
        if epsilon is None:
            raise ValueError("epsilon must be provided")
        if connectivity is None:
            raise ValueError("connectivity must be provided")
        self.n_agents = n_agents
        self.num_timesteps = num_timesteps
        self.llm_client = llm_client
        self.topics = topics
        self.random_seed = random_seed
        self.model = model
        self.llm_enabled = llm_enabled
        self.epsilon = float(epsilon)
        self.connectivity = float(connectivity)
        self.reversed = reversed
        self.output_file = output_file
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.network = NetworkModel(n_agents, random_seed)
        self.agents = self._initialize_agents()
        self.initial_opinions = np.array([agent.get_opinion() for agent in self.agents])
        
        A0 = create_connected_degroot_network(self.n_agents, connectivity=self.connectivity)
        self.network.adjacency_matrix = A0
        
        # Streaming data storage
        self.opinion_history = []
        self.mean_opinions = []
        self.std_opinions = []
        self.posts_history = []
        self.ratings_history = []
        self.final_opinions = None
    
    def _initialize_agents(self) -> List[Agent]:
        """
        Initialize agents with random opinions.
        
        Returns:
            List of initialized agents
        """
        # Create agents with random opinions
        return [Agent(agent_id=i, random_seed=self.random_seed) for i in range(self.n_agents)]
    
    
    def run_simulation(self, progress_bar: bool = True) -> Dict[str, Any]:
        """Run the simulation with streaming data storage."""
        # Initialize output file if provided
        if self.output_file:
            self._initialize_output_file()
        
        # Store initial state
        self._store_current_state()
        
        # Main simulation loop
        if progress_bar:
            topic_name = self.topics[0] if self.topics else "mathematical"
            iterator = tqdm(range(self.num_timesteps), 
                          desc=f"🔄 {topic_name} ({self.n_agents} agents)", 
                          unit="step")
        else:
            iterator = range(self.num_timesteps)
        
        for timestep in iterator:
            # Get current state
            X_current = self._get_opinion_matrix()
            A_current = self.network.get_adjacency_matrix()
            
            # Generate posts and interpretations (LLM) or skip (no-LLM)
            if self.llm_enabled and self.topics:
                topic_key = self.topics[0]
                from ..runner import get_topic_framing
                current_topic = get_topic_framing(topic_key, self.reversed)
                
                # Get previous posts from last stored state
                prev_posts = []
                if self.opinion_history:
                    # Get posts from last timestep (stored in output file or memory)
                    prev_posts = [f"Agent {i}: (no post)" for i in range(self.n_agents)]
                
                # Prepare neighbor posts
                neighbor_posts_per_agent = []
                for i in range(self.n_agents):
                    connections = A_current[i, :]
                    neighbor_indices = np.where(connections == 1)[0]
                    neighbor_texts = [prev_posts[j] for j in neighbor_indices if prev_posts[j] and prev_posts[j].strip()]
                    neighbor_posts_per_agent.append(neighbor_texts)

                # Generate posts and get opinions
                posts = self.llm_client.generate_posts(current_topic, self.agents, neighbor_posts_per_agent)
                R_pairwise, individual_ratings = self.llm_client.rate_posts_pairwise(posts, current_topic, self.agents, A_current)
                neighbor_opinions = [float(np.nanmean(R_pairwise[:, j])) if not np.all(np.isnan(R_pairwise[:, j])) 
                                   else float(self.agents[j].get_opinion()) for j in range(self.n_agents)]
            else:
                posts = None
                neighbor_opinions = None
                individual_ratings = None
                X_interpreted = X_current.copy()

            # Update opinions
            if self.llm_enabled:
                X_interpreted = np.array(neighbor_opinions)

            X_for_math = self._convert_domain(X_interpreted, to_math=True)
            
            if self.model == "degroot":
                X_next_math = update_opinions_pure_degroot(X_for_math, A_current, self.epsilon)
            else:
                raise ValueError(f"Unknown model: {self.model}")
                
            X_next_agent = self._convert_domain(X_next_math, to_math=False)
            self._update_agent_opinions(X_next_agent)
            
            # Store current state (streaming)
            self._store_current_state(posts, neighbor_opinions, individual_ratings)
            
            # Stream data to file if output file is provided
            if self.output_file:
                self._stream_to_file(timestep)
        
        # Finalize output file
        if self.output_file:
            self._finalize_output_file()
        
        return self._get_simulation_results()
    
    
    def _get_opinion_matrix(self) -> np.ndarray:
        """
        Get current opinion vector.
        
        Returns:
            Current opinion vector (n_agents,)
        """
        return np.array([agent.get_opinion() for agent in self.agents])

    def _convert_domain(self, x: np.ndarray, to_math: bool = True) -> np.ndarray:
        """Convert between agent domain [-1, 1] and math domain [0, 1]."""
        return (x + 1.0) / 2.0 if to_math else (2.0 * x) - 1.0
    
    def _update_agent_opinions(self, new_opinions: np.ndarray):
        """
        Update all agent opinions.
        
        Args:
            new_opinions: New opinion vector in agent domain [-1, 1]
        """
        for i, agent in enumerate(self.agents):
            agent.update_opinion(new_opinions[i])
    
    def _store_current_state(self, posts: Optional[List[str]] = None, neighbor_opinions: Optional[List[float]] = None, 
                           individual_ratings: Optional[List[List[tuple[int, float]]]] = None):
        """Store current simulation state (streaming approach)."""
        opinions = self._get_opinion_matrix()
        
        # Store only essential data for streaming
        self.opinion_history.append(opinions.tolist())
        self.mean_opinions.append(float(np.mean(opinions)))
        self.std_opinions.append(float(np.std(opinions)))
        self.final_opinions = opinions.tolist()
        
        # Store posts and ratings for analysis
        if posts:
            self.posts_history.append(posts)
            self.last_posts = posts
        else:
            self.posts_history.append([])
            
        if individual_ratings:
            # Convert ratings to serializable format
            serializable_ratings = []
            for agent_ratings in individual_ratings:
                agent_ratings_list = [(int(agent_id), float(rating)) for agent_id, rating in agent_ratings]
                serializable_ratings.append(agent_ratings_list)
            self.ratings_history.append(serializable_ratings)
        else:
            self.ratings_history.append([])
    
    def _initialize_output_file(self):
        """Initialize the output file for streaming data."""
        import json
        import os
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(self.output_file)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize with metadata
        initial_data = {
            'experiment_metadata': {
                'model': self.model,
                'topology': 'unknown',
                'n_agents': self.n_agents,
                'num_timesteps': self.num_timesteps,
                'llm_enabled': self.llm_enabled,
                'random_seed': self.random_seed,
                'topics': self.topics
            },
            'opinion_history': [],
            'mean_opinions': [],
            'std_opinions': [],
            'final_opinions': None,
            'posts_history': [],
            'ratings_history': [],
            'network_info': {
                'adjacency_matrix': self.network.get_adjacency_matrix().tolist(),
                'n_agents': self.n_agents
            }
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
    
    def _stream_to_file(self, timestep: int):
        """Stream current timestep data to file."""
        import json
        
        # Read current file
        with open(self.output_file, 'r') as f:
            data = json.load(f)
        
        # Update with current data
        data['opinion_history'] = self.opinion_history
        data['mean_opinions'] = self.mean_opinions
        data['std_opinions'] = self.std_opinions
        data['final_opinions'] = self.final_opinions
        data['posts_history'] = self.posts_history
        data['ratings_history'] = self.ratings_history
        
        # Write back to file
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _finalize_output_file(self):
        """Finalize the output file with complete results."""
        import json
        
        # Read current file
        with open(self.output_file, 'r') as f:
            data = json.load(f)
        
        # Add final metadata
        data['experiment_metadata']['completed'] = True
        data['experiment_metadata']['final_timestep'] = len(self.opinion_history) - 1
        
        # Write final version
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_simulation_results(self) -> Dict[str, Any]:
        """Get simulation results (simplified streaming approach)."""
        current_topic = self.topics[0] if self.topics else ("pure_math_model" if not self.llm_enabled else "unknown")
        
        # Use streaming data directly - flatten structure for compatibility
        results = {
            'experiment_metadata': {
                'model': self.model,
                'topology': 'unknown',
                'n_agents': self.n_agents,
                'num_timesteps': self.num_timesteps,
                'llm_enabled': self.llm_enabled,
                'random_seed': self.random_seed,
                'topics': self.topics,
                'reversed': self.reversed
            },
            'opinion_history': self.opinion_history,
            'mean_opinions': self.mean_opinions,
            'std_opinions': self.std_opinions,
            'final_opinions': self.final_opinions,
            'posts_history': self.posts_history,
            'ratings_history': self.ratings_history,
            'network_info': {
                'adjacency_matrix': self.network.get_adjacency_matrix().tolist(),
                'n_agents': self.n_agents
            }
        }
        
        return results
    
