"""
Canonical simulation controller for opinion dynamics experiments.

This controller implements the 6 canonical experimental configurations
with support for both DeGroot and Friedkin-Johnsen models.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import logging

from ..agent import Agent
from ..llm_client import LLMClient
from ..network.graph_model import NetworkModel
from ..network.graph_generator import create_network_model, get_network_info
from ..core.mathematics import update_opinions_pure_degroot, update_opinions_friedkin_johnsen

logger = logging.getLogger(__name__)

class CanonicalController:
    """
    Controller for canonical opinion dynamics experiments.
    """
    
    def __init__(self, 
                 experiment_config: Dict[str, Any],
                 llm_client: Optional[LLMClient] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize canonical simulation controller.
        
        Args:
            experiment_config: Complete experiment configuration
            llm_client: LLM client for post generation and interpretation
            random_seed: Random seed for reproducibility
        """
        self.experiment_config = experiment_config
        self.llm_client = llm_client
        self.random_seed = random_seed or experiment_config.get("random_seed", 42)
        
        # Set random seed
        np.random.seed(self.random_seed)
        
        # Extract configuration
        self.model = experiment_config["model"]
        self.model_params = experiment_config["model_params"]
        self.topology = experiment_config["topology"]
        self.topology_params = experiment_config["topology_params"]
        self.opinion_distribution = experiment_config["opinion_distribution"]
        self.opinion_params = experiment_config["opinion_params"]
        self.topic = experiment_config["topic"]
        self.max_timesteps = experiment_config.get("max_timesteps", 200)
        self.convergence_threshold = experiment_config.get("convergence_threshold", 1e-6)
        
        # Initialize network
        self.network = create_network_model(
            self.topology, 
            self.topology_params, 
            self.random_seed
        )
        
        # Get number of agents from network
        self.n_agents = self.network.n_agents
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize simulation state
        self.current_timestep = 0
        self.is_running = False
        
        # Data storage
        self.opinion_history = []
        self.pure_math_history = []
        self.posts_history = []
        self.ratings_history = []
        self.convergence_info = {}
        
        # Metrics
        self.api_call_count = 0
        self.simulation_start_time = None
        
        logger.info(f"Initialized {experiment_config['experiment_name']} with {self.n_agents} agents")
        logger.info(f"Network info: {get_network_info(self.network.get_adjacency_matrix())}")
    
    def _initialize_agents(self) -> List[Agent]:
        """Initialize agents with opinions based on the specified distribution."""
        agents = []
        
        # Generate initial opinions
        if self.opinion_distribution == "normal":
            opinions = np.random.normal(
                self.opinion_params["mu"], 
                self.opinion_params["sigma"], 
                self.n_agents
            )
        else:
            raise ValueError(f"Unsupported opinion distribution: {self.opinion_distribution}")
        
        # Clamp opinions to [-1, 1] range
        opinions = np.clip(opinions, -1.0, 1.0)
        
        # Create agents
        for i in range(self.n_agents):
            agent = Agent(
                agent_id=i,
                initial_opinion=opinions[i],
                topic_a=self.topic["a"],
                topic_b=self.topic["b"]
            )
            agents.append(agent)
        
        return agents
    
    def _generate_posts(self) -> List[str]:
        """Generate posts from all agents."""
        posts = []
        
        for agent in self.agents:
            if self.llm_client is not None:
                # Get context from neighbors
                neighbor_posts = self._get_neighbor_posts(agent.agent_id)
                
                # Generate post using LLM
                post = self.llm_client.generate_post(
                    agent=agent,
                    neighbor_posts=neighbor_posts
                )
                self.api_call_count += 1
            else:
                # Fallback: simple text generation
                post = f"Agent {agent.agent_id}: My opinion is {agent.current_opinion:.3f}"
            
            posts.append(post)
        
        return posts
    
    def _get_neighbor_posts(self, agent_id: int, max_posts: int = 6) -> List[str]:
        """Get recent posts from neighbors."""
        adjacency = self.network.get_adjacency_matrix()
        neighbors = np.where(adjacency[agent_id] > 0)[0]
        
        neighbor_posts = []
        for neighbor_id in neighbors[:max_posts]:
            if (neighbor_id < len(self.posts_history) and 
                len(self.posts_history[neighbor_id]) > 0):
                # Get the most recent post from this neighbor
                recent_post = self.posts_history[neighbor_id][-1]
                neighbor_posts.append(recent_post)
        
        return neighbor_posts
    
    def _rate_posts(self, posts: List[str]) -> np.ndarray:
        """Rate all posts and return rating matrix."""
        n_agents = len(self.agents)
        ratings = np.zeros((n_agents, n_agents))
        
        adjacency = self.network.get_adjacency_matrix()
        
        for i in range(n_agents):
            for j in range(n_agents):
                if adjacency[i, j] > 0:  # Agent i can see agent j's post
                    if self.llm_client is not None:
                        # Use LLM to rate the post
                        rating = self.llm_client.rate_post(
                            agent=self.agents[i],
                            post=posts[j]
                        )
                        self.api_call_count += 1
                    else:
                        # Fallback: simple rating based on opinion similarity
                        rating = self._simple_rating(self.agents[i], self.agents[j])
                    
                    ratings[i, j] = rating
        
        return ratings
    
    def _simple_rating(self, rater: Agent, poster: Agent) -> float:
        """Simple rating based on opinion similarity."""
        # Convert opinions to [0, 1] range
        rater_opinion = (rater.current_opinion + 1) / 2
        poster_opinion = (poster.current_opinion + 1) / 2
        
        # Simple similarity-based rating
        similarity = 1 - abs(rater_opinion - poster_opinion)
        return 2 * similarity - 1  # Convert back to [-1, 1]
    
    def _update_opinions(self, ratings: np.ndarray):
        """Update agent opinions based on ratings."""
        adjacency = self.network.get_adjacency_matrix()
        
        for i, agent in enumerate(self.agents):
            # Get ratings from neighbors
            neighbor_ratings = []
            for j in range(len(self.agents)):
                if adjacency[i, j] > 0:
                    neighbor_ratings.append(ratings[i, j])
            
            if neighbor_ratings:
                # Calculate mean rating
                mean_rating = np.mean(neighbor_ratings)
                
                # Convert to [0, 1] range for mathematical models
                math_opinion = (mean_rating + 1) / 2
                
                # Update agent opinion
                agent.current_opinion = mean_rating
        
        # Store current opinions
        current_opinions = np.array([agent.current_opinion for agent in self.agents])
        self.opinion_history.append(current_opinions.copy())
    
    def _update_pure_math(self):
        """Update pure mathematical model for comparison."""
        if not self.opinion_history:
            return
        
        current_opinions = self.opinion_history[-1]
        
        # Convert to [0, 1] range
        math_opinions = (current_opinions + 1) / 2
        
        # Apply mathematical update
        adjacency = self.network.get_adjacency_matrix()
        
        if self.model == "degroot":
            new_math_opinions = update_opinions_pure_degroot(
                math_opinions, 
                adjacency, 
                self.model_params.get("epsilon", 1e-6)
            )
        elif self.model == "friedkin_johnsen":
            # For Friedkin-Johnsen, we need to handle stubborn agents
            lambda_val = self.model_params.get("lambda", 0.8)
            stubborn_fraction = self.model_params.get("stubborn_fraction", 0.1)
            
            # Create susceptibility matrix
            n_stubborn = int(self.n_agents * stubborn_fraction)
            susceptibility = np.ones(self.n_agents) * lambda_val
            if n_stubborn > 0:
                # Make first n_stubborn agents stubborn
                susceptibility[:n_stubborn] = 0.0
            
            new_math_opinions = update_opinions_friedkin_johnsen(
                math_opinions,
                adjacency,
                susceptibility
            )
        else:
            raise ValueError(f"Unknown model: {self.model}")
        
        # Convert back to [-1, 1] range
        new_opinions = 2 * new_math_opinions - 1
        self.pure_math_history.append(new_opinions)
    
    def _check_convergence(self) -> bool:
        """Check if the simulation has converged."""
        if len(self.opinion_history) < 2:
            return False
        
        current_opinions = self.opinion_history[-1]
        previous_opinions = self.opinion_history[-2]
        
        # Check if opinions have changed significantly
        opinion_change = np.mean(np.abs(current_opinions - previous_opinions))
        
        return opinion_change < self.convergence_threshold
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run the complete simulation."""
        logger.info(f"Starting simulation: {self.experiment_config['experiment_name']}")
        self.is_running = True
        self.simulation_start_time = time.time()
        
        try:
            for timestep in tqdm(range(self.max_timesteps), desc="Simulation"):
                self.current_timestep = timestep
                
                # Generate posts
                posts = self._generate_posts()
                self.posts_history.append(posts)
                
                # Rate posts
                ratings = self._rate_posts(posts)
                self.ratings_history.append(ratings)
                
                # Update opinions
                self._update_opinions(ratings)
                
                # Update pure mathematical model
                self._update_pure_math()
                
                # Check convergence
                if self._check_convergence():
                    logger.info(f"Converged at timestep {timestep}")
                    break
            
            # Finalize simulation
            self._finalize_simulation()
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
        finally:
            self.is_running = False
        
        return self.get_results()
    
    def _finalize_simulation(self):
        """Finalize simulation and calculate metrics."""
        if not self.opinion_history:
            return
        
        # Calculate convergence info
        final_opinions = self.opinion_history[-1]
        self.convergence_info = {
            "final_mean": np.mean(final_opinions),
            "final_std": np.std(final_opinions),
            "converged": self._check_convergence(),
            "timesteps": len(self.opinion_history),
            "api_calls": self.api_call_count
        }
        
        # Calculate divergence from pure math
        if self.pure_math_history:
            final_pure_math = self.pure_math_history[-1]
            divergence = np.mean(np.abs(final_opinions - final_pure_math))
            self.convergence_info["divergence"] = divergence
    
    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        return {
            "experiment_config": self.experiment_config,
            "convergence_info": self.convergence_info,
            "opinion_history": [opinions.tolist() for opinions in self.opinion_history],
            "pure_math_history": [opinions.tolist() for opinions in self.pure_math_history],
            "posts_history": self.posts_history,
            "ratings_history": [ratings.tolist() for ratings in self.ratings_history],
            "network_info": get_network_info(self.network.get_adjacency_matrix()),
            "simulation_time": time.time() - self.simulation_start_time if self.simulation_start_time else 0
        }
