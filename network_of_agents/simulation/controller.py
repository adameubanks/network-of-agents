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
from ..core.mathematics import update_opinions_pure_degroot, update_opinions_friedkin_johnsen, create_connected_degroot_network

logger = logging.getLogger(__name__)

class Controller:
    """
    Streamlined controller for the network of agents simulation.
    """
    
    def __init__(self, 
                 llm_client: Optional[LLMClient] = None,
                 n_agents: int = 10,
                 epsilon: float = 0.001,
                 num_timesteps: int = 50,
                 topics: Optional[List[str]] = None,
                 random_seed: Optional[int] = None,
                 llm_enabled: bool = True,
                 model: str = "degroot",
                 on_timestep: Optional[Callable[[Dict[str, Any], int], None]] = None,
                 progress_callback: Optional[Callable[[int, int], None]] = None,
                 resume_data: Optional[Dict[str, Any]] = None,
                 checkpoint_interval: int = 10,
                 checkpoint_dir: Optional[str] = None):
        """
        Initialize simulation controller.
        
        Args:
            n_agents: Number of agents in the network
            epsilon: Small positive parameter for numerical stability
            num_timesteps: Number of simulation timesteps
            llm_client: LLM client for post generation and interpretation
            topics: List of topics for opinions
            random_seed: Random seed for reproducible results
            checkpoint_interval: Save checkpoint every N timesteps (0 to disable)
            checkpoint_dir: Directory to save checkpoints (None for temp dir)

        """
        self.n_agents = n_agents
        self.num_timesteps = num_timesteps
        # Paper-consistent: no ER init, no lazy update probability
        self.llm_client = llm_client
        self.topics = topics
        self.random_seed = random_seed
        self.model = model

        self.llm_enabled = llm_enabled
        self.on_timestep = on_timestep
        self.progress_callback = progress_callback

        self.epsilon = float(epsilon)

        self.llm_opinion_history = []
        self.pure_degroot_opinion_history = []
        
        # Checkpoint configuration
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = None
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.network = NetworkModel(n_agents, random_seed)
        self.agents = self._initialize_agents()
        self.initial_opinions = np.array([agent.get_opinion() for agent in self.agents])
        self.susceptibility_matrix = None

        if resume_data is None:
            A0 = create_connected_degroot_network(self.n_agents, connectivity=0.05)
            self.network.adjacency_matrix = A0
        
        self.is_running = False
        self.current_timestep = 0
        self.simulation_start_time = None
        self.timesteps = []

        if resume_data is not None:
            timesteps_data = resume_data.get('timesteps', [])
            if isinstance(timesteps_data, dict):
                self.timesteps = [timesteps_data[str(i)] for i in sorted(int(k) for k in timesteps_data.keys())]
            else:
                self.timesteps = timesteps_data
            
            if self.timesteps:
                last_ts = self.timesteps[-1]
                for agent_id, agent_data in last_ts.get('agents', {}).items():
                    i = int(agent_id)
                    if i < self.n_agents:
                        self.agents[i].update_opinion(float(agent_data.get('opinion', 0)))
                
                A = np.zeros((self.n_agents, self.n_agents))
                for agent_id, agent_data in last_ts.get('agents', {}).items():
                    i = int(agent_id)
                    for j in agent_data.get('connected_to', []):
                        j = int(j)
                        if j < self.n_agents:
                            A[i, j] = A[j, i] = 1.0
                self.network.adjacency_matrix = A
                
                self.current_timestep = last_ts.get('timestep', 0) + 1
    
    def _initialize_agents(self) -> List[Agent]:
        """
        Initialize agents with random opinions.
        
        Returns:
            List of initialized agents
        """
        # Create agents with random opinions
        return [Agent(agent_id=i, random_seed=self.random_seed) for i in range(self.n_agents)]
    
    def _calculate_pure_degroot_opinions(self, X_current: np.ndarray, A_current: np.ndarray) -> np.ndarray:
        """Calculate what pure DeGroot would produce for comparison with LLM results."""
        X_for_math = self._convert_domain(X_current, to_math=True)
        X_next_math = update_opinions_pure_degroot(X_for_math, A_current, self.epsilon)
        return self._convert_domain(X_next_math, to_math=False)
    
    def _calculate_llm_degroot_divergence(self, llm_opinions: np.ndarray, pure_opinions: np.ndarray) -> Dict[str, float]:
        """Calculate divergence metrics between LLM and pure DeGroot results."""
        if len(llm_opinions) != len(pure_opinions):
            return {"error": "Opinion vector length mismatch"}
        
        # Mean Absolute Error
        mae = np.mean(np.abs(llm_opinions - pure_opinions))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((llm_opinions - pure_opinions) ** 2))
        
        # Maximum Absolute Error
        max_error = np.max(np.abs(llm_opinions - pure_opinions))
        
        # Correlation coefficient - handle NaN cases
        try:
            if len(llm_opinions) > 1 and np.std(llm_opinions) > 0 and np.std(pure_opinions) > 0:
                correlation = np.corrcoef(llm_opinions, pure_opinions)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
        except:
            correlation = 0.0
        
        return {
            "mae": float(mae),
            "rmse": float(rmse), 
            "max_error": float(max_error),
            "correlation": float(correlation)
        }
    
    def _calculate_timestep_divergence(self) -> List[float]:
        """Calculate average distance between LLM and pure DeGroot at each timestep."""
        if not self.llm_enabled or len(self.llm_opinion_history) == 0:
            return []
        
        divergences = []
        for t in range(len(self.llm_opinion_history)):
            llm_opinions = np.array(self.llm_opinion_history[t])
            pure_opinions = np.array(self.pure_degroot_opinion_history[t])
            
            # Calculate average absolute distance between the two opinion vectors
            avg_distance = np.mean(np.abs(llm_opinions - pure_opinions))
            divergences.append(float(avg_distance))
        
        return divergences
    
    def run_simulation(self, progress_bar: bool = True) -> Dict[str, Any]:
        """
        Run the simulation with LLM post generation and interpretation.
        
        Args:
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary containing simulation results
        """
        self.is_running = True
        # Preserve current_timestep if resuming
        start_timestep = self.current_timestep
        self.simulation_start_time = time.time()
        
        # Setup checkpoint file
        self._setup_checkpoint_file()
        
        # Store initial state only on fresh runs
        if start_timestep == 0:
            self._store_current_state()
        
        # Main simulation loop with enhanced progress tracking
        if progress_bar:
            topic_name = self.topics[0] if self.topics else "mathematical"
            iterator = tqdm(range(start_timestep, self.num_timesteps), 
                          desc=f"🔄 {topic_name} ({self.n_agents} agents)", 
                          unit="step", 
                          position=0,  # Position 0 for main progress
                          leave=True)  # Keep visible to show timestep progress
        else:
            iterator = range(start_timestep, self.num_timesteps)
        
        for timestep in iterator:
            timestep_start_time = time.time()
            self.current_timestep = timestep
            
            
            # Step 1: Get current network state
            X_current = self._get_opinion_matrix()
            A_current = self.network.get_adjacency_matrix()
            A_prev = self.network.network_history[-1] if self.network.network_history else A_current
            
            # Step 2: Generate posts and interpretations (LLM) or skip (no-LLM)
            # Get the topic key and convert to proper framing for LLM calls
            topic_key = self.topics[0] if self.topics else None
            if topic_key:
                # Import the topic mapping to get proper framing
                from ..runner import get_topic_framing
                current_topic = get_topic_framing(topic_key)
            else:
                current_topic = None
            
            # Capture opinions before any updates (used for post generation)
            pre_update_opinions = self._get_opinion_matrix()
            
            if self.llm_enabled:
                # Get previous posts from last timestep
                prev_posts = []
                if self.timesteps:
                    last_ts = self.timesteps[-1]
                    prev_posts = [last_ts['agents'].get(str(i), {}).get('post', f"Agent {i}: (no post)") 
                                for i in range(self.n_agents)]
                
                # Prepare neighbor posts for connected agents
                neighbor_posts_per_agent = []
                for i in range(self.n_agents):
                    connections = A_prev[i, :]
                    neighbor_indices = np.where(connections == 1)[0]
                    neighbor_texts = [prev_posts[j] for j in neighbor_indices if prev_posts[j] and prev_posts[j].strip()]
                    neighbor_posts_per_agent.append(neighbor_texts)

                # Generate posts for all agents
                posts = self.llm_client.generate_posts(current_topic, self.agents, neighbor_posts_per_agent)
                
                # Rate posts and get neighbor opinions
                R_pairwise, individual_ratings = self.llm_client.rate_posts_pairwise(posts, current_topic, self.agents, A_current)
                neighbor_opinions = [float(np.nanmean(R_pairwise[:, j])) if not np.all(np.isnan(R_pairwise[:, j])) 
                                   else float(self.agents[j].get_opinion()) for j in range(self.n_agents)]
            else:
                posts = None
                neighbor_opinions = None
                individual_ratings = None

            # Step 3: Update opinions using pure DeGroot (fixed network)
            if self.llm_enabled:
                X_interpreted = np.array(neighbor_opinions)

                # Calculate what pure DeGroot would produce for comparison
                pure_degroot_opinions = self._calculate_pure_degroot_opinions(X_current, A_current)
                self.pure_degroot_opinion_history.append(pure_degroot_opinions.copy())

                # Calculate divergence metrics
                divergence_metrics = self._calculate_llm_degroot_divergence(X_interpreted, pure_degroot_opinions)

                logger.info(f"Timestep {timestep}: MAE: {divergence_metrics['mae']:.4f}, Correlation: {divergence_metrics['correlation']:.4f}")
            else:
                X_interpreted = X_current.copy()

            X_for_math = self._convert_domain(X_interpreted, to_math=True)
            
            # Use the specified model for opinion updates
            if self.model == "degroot":
                X_next_math = update_opinions_pure_degroot(X_for_math, A_current, self.epsilon)
            elif self.model == "friedkin_johnsen":
                lambda_values = np.full(self.n_agents, 0.1)
                X_0_math = self._convert_domain(self.initial_opinions, to_math=True)
                X_next_math = update_opinions_friedkin_johnsen(X_for_math, A_current, lambda_values, X_0_math, self.epsilon)
            else:
                raise ValueError(f"Unknown model: {self.model}")
                
            X_next_agent = self._convert_domain(X_next_math, to_math=False)
            self._update_agent_opinions(X_next_agent)
            
            # Store LLM opinions for comparison
            if self.llm_enabled:
                self.llm_opinion_history.append(X_interpreted.copy())
            
            # No early convergence check - run for full specified timesteps

            # Step 6: Store current state
            self._store_current_state(posts, neighbor_opinions, individual_ratings, pre_update_opinions)
            
            # Record timing
            timestep_time = time.time() - timestep_start_time
            
            # Call progress callback if provided (this updates the higher-level progress bars)
            if self.progress_callback:
                self.progress_callback(timestep + 1, self.num_timesteps)

            # Save checkpoint if interval reached
            if (self.checkpoint_interval > 0 and 
                (timestep + 1) % self.checkpoint_interval == 0):
                self._save_checkpoint()
            
            # Invoke per-timestep callback if provided
            if self.on_timestep is not None:
                current_state = self.get_current_state()
                current_state['completed_timesteps'] = timestep + 1
                current_state['total_timesteps'] = self.num_timesteps
                self.on_timestep(current_state, timestep)
        
        self.is_running = False
        
        # Cleanup checkpoint file after successful completion
        self._cleanup_checkpoint()
        
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
                           individual_ratings: Optional[List[List[tuple[int, float]]]] = None, 
                           pre_update_opinions: Optional[np.ndarray] = None):
        """Store current simulation state in consolidated format."""
        opinions = self._get_opinion_matrix()
        current_adjacency = self.network.get_adjacency_matrix()
        
        # Build ratings matrix for LLM data
        ratings_matrix = {}
        if individual_ratings:
            for i, out_list in enumerate(individual_ratings):
                for (j, r) in out_list:
                    if j not in ratings_matrix:
                        ratings_matrix[j] = {}
                    ratings_matrix[int(j)][int(i)] = float(r)
        
        # Store agent data
        agent_data = {}
        for i, agent in enumerate(self.agents):
            ratings_received = []
            if i in ratings_matrix:
                for rater_id, rating in ratings_matrix[i].items():
                    ratings_received.append({
                        'rater_agent': int(rater_id), 
                        'rating': float(rating),
                        'rater_opinion': float(opinions[rater_id])
                    })
            
            agent_data[str(i)] = {
                'opinion': float(opinions[i]),
                'post': posts[i] if posts and i < len(posts) else "No post available",
                'connected_to': [str(j) for j in range(self.n_agents) if current_adjacency[i, j] == 1],
                'interpretations_received': {str(r['rater_agent']): r['rating'] for r in ratings_received}
            }
        
        # Store timestep data
        timestep_data = {
            'timestep': self.current_timestep,
            'agents': agent_data,
            'mean_opinion': float(np.mean(opinions)),
            'std_opinion': float(np.std(opinions))
        }
        
        self.timesteps.append(timestep_data)
    
        
    
    def _get_simulation_results(self) -> Dict[str, Any]:
        """Get simulation results from consolidated timesteps data."""
        current_topic = self.topics[0] if self.topics else ("pure_math_model" if not self.llm_enabled else "unknown")
        
        # Extract data from timesteps
        opinion_history = []
        mean_opinions = []
        std_opinions = []
        
        for timestep_data in self.timesteps:
            opinions = [agent_data['opinion'] for agent_data in timestep_data['agents'].values()]
            opinion_history.append(opinions)
            mean_opinions.append(timestep_data['mean_opinion'])
            std_opinions.append(timestep_data['std_opinion'])
        
        # Convert timesteps to nested format
        timesteps_nested = {str(ts['timestep']): {'agents': ts['agents']} for ts in self.timesteps}
        
        results = {
            'experiment_metadata': {
                'model': self.model,
                'topology': 'unknown',
                'n_agents': self.n_agents,
                'num_timesteps': self.num_timesteps,
                'llm_enabled': self.llm_enabled,
                'random_seed': self.random_seed,
                'topics': self.topics,
                'convergence_metrics': {
                    'final_mean_opinion': mean_opinions[-1] if mean_opinions else 0.0,
                    'final_std_opinion': std_opinions[-1] if std_opinions else 0.0,
                    'converged': len(mean_opinions) < self.num_timesteps
                }
            },
            'results': {
                current_topic: {
                    'timesteps': timesteps_nested,
                    'summary_metrics': {
                        'opinion_history': opinion_history,
                        'mean_opinions': mean_opinions,
                        'std_opinions': std_opinions,
                        'final_opinions': self._get_opinion_matrix().tolist(),
                        'network_info': {
                            'adjacency_matrix': self.network.get_adjacency_matrix().tolist(),
                            'n_agents': self.n_agents
                        }
                    }
                }
            }
        }
        
        # Add LLM comparison data if available
        if self.llm_enabled and len(self.llm_opinion_history) > 0:
            results['results'][current_topic]['llm_vs_pure_degroot'] = {
                'llm_opinion_history': [op.tolist() for op in self.llm_opinion_history],
                'pure_degroot_opinion_history': [op.tolist() for op in self.pure_degroot_opinion_history],
                'final_divergence': self._calculate_llm_degroot_divergence(
                    self.llm_opinion_history[-1], 
                    self.pure_degroot_opinion_history[-1]
                ),
                'timestep_divergence': self._calculate_timestep_divergence()
            }
        
        return results
    
    def _setup_checkpoint_file(self):
        """Setup checkpoint file path"""
        if self.checkpoint_interval <= 0:
            return
            
        if self.checkpoint_dir is None:
            self.checkpoint_dir = tempfile.mkdtemp(prefix="simulation_checkpoint_")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        timestamp = int(time.time())
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_{timestamp}.json")
        logger.info(f"Checkpoint file: {self.checkpoint_file}")
    
    def _save_checkpoint(self):
        """Save current simulation state to checkpoint file"""
        if self.checkpoint_interval <= 0 or self.checkpoint_file is None:
            return
            
        try:
            checkpoint_data = {
                'current_timestep': self.current_timestep,
                'num_timesteps': self.num_timesteps,
                'n_agents': self.n_agents,
                'model': self.model,
                'topics': self.topics,
                'random_seed': self.random_seed,
                'llm_enabled': self.llm_enabled,
                'timesteps': self.timesteps,
                'llm_opinion_history': [op.tolist() for op in self.llm_opinion_history],
                'pure_degroot_opinion_history': [op.tolist() for op in self.pure_degroot_opinion_history],
                'network_adjacency': self.network.get_adjacency_matrix().tolist(),
                'initial_opinions': self.initial_opinions.tolist(),
                'checkpoint_timestamp': time.time()
            }
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
            logger.info(f"Checkpoint saved at timestep {self.current_timestep}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _cleanup_checkpoint(self):
        """Remove checkpoint file after successful completion"""
        if self.checkpoint_file and os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                logger.info("Checkpoint file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoint file: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        opinions = self._get_opinion_matrix()
        return {
            'timestep': self.current_timestep,
            'opinions': opinions.tolist(),
            'is_running': self.is_running,
            'mean_opinion': float(np.mean(opinions)),
            'std_opinion': float(np.std(opinions))
        } 
