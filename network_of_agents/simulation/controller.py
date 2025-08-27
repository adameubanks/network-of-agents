"""
Streamlined simulation controller for the network of agents.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import time
import logging

from ..agent import Agent
from ..llm_client import LLMClient
from ..network.graph_model import NetworkModel
from ..core.mathematics import update_opinions, update_edges

logger = logging.getLogger(__name__)


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
                 topics: Optional[List[str]] = None,
                 random_seed: Optional[int] = None,

                 llm_enabled: bool = True,
                 noise_enabled: bool = False,
                 noise_mean: float = 0.0,
                 noise_std: float = 0.0,
                 noise_clip: bool = True,
                 on_timestep: Optional[Callable[[Dict[str, Any], int], None]] = None):
        """
        Initialize simulation controller.
        
        Args:
            n_agents: Number of agents in the network
            epsilon: Small positive parameter for numerical stability
            theta: Positive integer parameter for edge formation
            num_timesteps: Number of simulation timesteps
            llm_client: LLM client for post generation and interpretation
            topics: List of topics for opinions
            random_seed: Random seed for reproducible results

        """
        self.n_agents = n_agents
        self.epsilon = epsilon
        self.theta = theta
        self.num_timesteps = num_timesteps
        # Paper-consistent: no ER init, no lazy update probability
        self.llm_client = llm_client
        self.topics = topics
        self.random_seed = random_seed

        self.llm_enabled = llm_enabled
        self.noise_enabled = noise_enabled
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.on_timestep = on_timestep
        

        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize components
        self.network = NetworkModel(n_agents, random_seed)
        self.agents = self._initialize_agents()

        # Reinitialize A[0] using Eq.(4) based on initial opinions to avoid discontinuity
        try:
            X0_agent = self._get_opinion_matrix()
            X0_math = self._to_math_domain(X0_agent)
            # Sample all pairs per Eq.(4)
            A0 = update_edges(self.network.get_adjacency_matrix(), X0_math, self.theta, self.epsilon, update_probability=1.0)
            # Set directly without adding ER graph to history
            self.network.adjacency_matrix = A0
        except Exception:
            # If anything fails, keep the initialized ER graph
            pass
        
        # Simulation state
        self.is_running = False
        self.current_timestep = 0
        
        # Circuit breaker for LLM failures
        self.llm_failure_count = 0
        self.llm_circuit_open = False
        self.llm_circuit_open_time = 0
        self.llm_circuit_timeout = 60  # 60 seconds timeout
        

        
        # Health monitoring
        self.simulation_start_time = None
        self.timestep_times = []
        self.llm_success_count = 0
        self.llm_failure_count = 0
        self.llm_total_calls = 0
        
        # Data storage
        self.opinion_history = []
        self.mean_opinions = []
        self.std_opinions = []
        self.posts_history = []
        self.interpretations_history = []
        self.timesteps = []
    
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
        self.simulation_start_time = time.time()
        
        # Store initial state
        if self.llm_enabled:
            # Don't store detailed agent state for initial empty state
            self._store_current_state([], [], [], skip_detailed=True)
        else:
            self._store_current_state(None, None, None)
        
        # Main simulation loop
        iterator = tqdm(range(self.num_timesteps), desc="Running simulation") if progress_bar else range(self.num_timesteps)
        
        try:
            for timestep in iterator:
                timestep_start_time = time.time()
                self.current_timestep = timestep
                
                # Step 1: Get current network state
                X_current = self._get_opinion_matrix()
                A_current = self.network.get_adjacency_matrix()
                
                # Step 2: Generate posts and interpretations (LLM) or skip (no-LLM)
                current_topic = self.topics[0]  # Use single topic for entire simulation
                
                # Check circuit breaker
                if self.llm_circuit_open:
                    if time.time() - self.llm_circuit_open_time > self.llm_circuit_timeout:
                        logger.info("Circuit breaker timeout expired, re-enabling LLM calls")
                        self.llm_circuit_open = False
                        self.llm_failure_count = 0
                    else:
                        logger.warning(f"Circuit breaker is open, skipping LLM calls. Time remaining: {self.llm_circuit_timeout - (time.time() - self.llm_circuit_open_time):.1f}s")
                        posts = None
                        neighbor_opinions = None
                        individual_ratings = None
                
                if self.llm_enabled and not self.llm_circuit_open:
                    self.llm_total_calls += 1
                    try:
                        # Prepare prior-timestep neighbor posts (raw text only)
                        if self.posts_history and len(self.posts_history[-1]) == len(self.agents):
                            prev_posts = self.posts_history[-1]
                            neighbor_posts_per_agent = []
                            for i in range(self.n_agents):
                                connections = A_current[i, :]
                                neighbor_indices = np.where(connections == 1)[0]
                                neighbor_texts = [prev_posts[j] for j in neighbor_indices if prev_posts[j] and prev_posts[j].strip()]
                                neighbor_posts_per_agent.append(neighbor_texts)
                        else:
                            neighbor_posts_per_agent = None

                        posts = self.llm_client.generate_posts_for_agents_with_retry(current_topic, self.agents, neighbor_posts_per_agent)
                        logger.info(f"Posts generated successfully. Count: {len(posts)}, Expected: {len(self.agents)}")
                        
                        # Validate posts data
                        if len(posts) != len(self.agents):
                            logger.error(f"POSTS VALIDATION FAILED: Got {len(posts)} posts, expected {len(self.agents)}")
                            raise ValueError(f"Posts count mismatch: got {len(posts)}, expected {len(self.agents)}")
                        
                        neighbor_opinions, individual_ratings = self.llm_client.interpret_neighbor_posts_with_retry(posts, current_topic, self.agents, A_current)
                        
                        # Validate interpretation data
                        if len(neighbor_opinions) != len(self.agents):
                            logger.error(f"OPINIONS VALIDATION FAILED: Got {len(neighbor_opinions)} opinions, expected {len(self.agents)}")
                            raise ValueError(f"Opinions count mismatch: got {len(neighbor_opinions)}, expected {len(self.agents)}")
                        
                        if len(individual_ratings) != len(self.agents):
                            logger.error(f"RATINGS VALIDATION FAILED: Got {len(individual_ratings)} ratings, expected {len(self.agents)}")
                            raise ValueError(f"Ratings count mismatch: got {len(individual_ratings)}, expected {len(self.agents)}")
                                                
                        # Track success
                        self.llm_success_count += 1
                        
                        # Reset failure count on success
                        self.llm_failure_count = 0
                        
                    except Exception as e:
                        logger.error(f"LLM call failed at timestep {timestep}: {str(e)}")
                        self.llm_failure_count += 1
                        
                        # Check if circuit breaker should open
                        if self.llm_failure_count >= 3:  # Open circuit after 3 consecutive failures
                            logger.error(f"Opening circuit breaker after {self.llm_failure_count} consecutive failures")
                            self.llm_circuit_open = True
                            self.llm_circuit_open_time = time.time()
                        
                        # Fallback to no-LLM mode for this timestep
                        posts = None
                        neighbor_opinions = None
                        individual_ratings = None
                        logger.info("Falling back to no-LLM mode for this timestep")
                else:
                    posts = None
                    neighbor_opinions = None
                    individual_ratings = None

                # Step 3: Update opinions using mathematical framework

                if self.llm_enabled:
                    X_interpreted = np.array(neighbor_opinions)
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
                A_next = update_edges(A_current, X_for_math, self.theta, self.epsilon, update_probability=1.0)
                self.network.update_adjacency_matrix(A_next)

                # Step 5: Convert back to agent domain [-1, 1] and update agent opinions
                X_next_agent = self._to_agent_domain(X_next_math)
                self._update_agent_opinions(X_next_agent)

                # Step 6: Store current state
                if self.llm_enabled:
                    # Additional validation before storing state
                    if posts is not None and neighbor_opinions is not None and individual_ratings is not None:
                        self._store_current_state(posts, neighbor_opinions, individual_ratings)
                    else:
                        self._store_current_state(None, None, None, skip_detailed=True)
                else:
                    self._store_current_state(None, None, None)
                

                
                # Record timestep performance
                timestep_duration = time.time() - timestep_start_time
                self.timestep_times.append(timestep_duration)

                # Invoke per-timestep callback with partial results snapshot
                if self.on_timestep is not None:
                    try:
                        partial_results = self._get_partial_simulation_results()
                        partial_results['completed_timesteps'] = len(self.mean_opinions)
                        partial_results['total_timesteps'] = self.num_timesteps
                        self.on_timestep(partial_results, timestep)
                    except Exception as e:
                        logger.warning(f"on_timestep callback failed at timestep {timestep}: {e}")
                
                # Report health metrics every 10 timesteps
                if timestep % 10 == 0:
                    self._report_health_metrics(timestep)
            
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
            'simulation_params': {
                'n_agents': self.n_agents,
                'num_timesteps': self.num_timesteps,
                'epsilon': self.epsilon,
                'theta': self.theta,
                # No initial ER probability; full resample each step per Eq.(4)
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
        # Always include timesteps for both LLM and non-LLM simulations
        results['timesteps'] = self.timesteps
        
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
    
    def _store_current_state(self, posts: Optional[List[str]], interpretations: Optional[List[float]], individual_ratings: Optional[List[List[tuple[int, float]]]], skip_detailed: bool = False):
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
        
        # Always store a per-timestep network snapshot
        if self.llm_enabled and (not skip_detailed) and posts is not None and interpretations is not None and individual_ratings is not None:
            self._store_detailed_agent_state(posts, interpretations, individual_ratings)
        else:
            # Basic snapshot (works for both LLM-skipped and non-LLM modes)
            self._store_basic_agent_state()
    
    def _store_detailed_agent_state(self, posts: List[str], interpretations: List[float], individual_ratings: List[List[tuple[int, float]]]):
        """Store detailed information about each agent at current timestep."""
        current_adjacency = self.network.get_adjacency_matrix()
        current_opinions = self._get_opinion_matrix()
        
        timestep_data = {
            'timestep': self.current_timestep,
            'agents': []
        }
                
        for i, agent in enumerate(self.agents):
            # Find connected neighbors for this agent
            connections = current_adjacency[i, :]
            neighbor_indices = np.where(connections == 1)[0]
            
            # Use safe access methods to prevent index errors
            agent_post = self._safe_get_posts(posts, i)
            agent_interpretation = self._safe_get_interpretation(interpretations, i)
            agent_ratings = self._safe_get_individual_ratings(individual_ratings, i)
            
            # Get just the agent IDs that are connected
            connected_agent_ids = neighbor_indices.tolist()
            
            agent_data = {
                'agent_id': i,
                'opinion': float(current_opinions[i]),
                'post': agent_post,
                'connected_agents': connected_agent_ids
            }
            
            timestep_data['agents'].append(agent_data)
        
        self.timesteps.append(timestep_data)
    
    def _store_basic_agent_state(self):
        """Store basic agent information for non-LLM simulations."""
        current_adjacency = self.network.get_adjacency_matrix()
        current_opinions = self._get_opinion_matrix()
        
        timestep_data = {
            'timestep': self.current_timestep,
            'agents': []
        }
                
        for i, agent in enumerate(self.agents):
            # Find connected neighbors for this agent
            connections = current_adjacency[i, :]
            neighbor_indices = np.where(connections == 1)[0]
            
            # Get just the agent IDs that are connected
            connected_agent_ids = neighbor_indices.tolist()
            
            agent_data = {
                'agent_id': i,
                'opinion': float(current_opinions[i]),
                'post': "No post available (LLM disabled)",
                'connected_agents': connected_agent_ids
            }
            
            timestep_data['agents'].append(agent_data)
        
        self.timesteps.append(timestep_data)
        logger.info(f"Basic agent state stored successfully for timestep {self.current_timestep}")
    

    
    def _report_health_metrics(self, timestep: int):
        """
        Report health metrics for the simulation.
        
        Args:
            timestep: Current timestep number
        """
        if not self.timestep_times:
            return
        
        # Calculate performance metrics
        avg_timestep_time = sum(self.timestep_times) / len(self.timestep_times)
        recent_timestep_time = self.timestep_times[-1] if self.timestep_times else 0
        
        # Calculate LLM metrics
        llm_success_rate = (self.llm_success_count / max(self.llm_total_calls, 1)) * 100 if self.llm_total_calls > 0 else 0
        
        # Calculate estimated completion time
        remaining_timesteps = self.num_timesteps - timestep - 1
        estimated_completion_time = remaining_timesteps * avg_timestep_time
        
        # Report metrics
        logger.info(f"=== HEALTH METRICS (Timestep {timestep + 1}/{self.num_timesteps}) ===")
        logger.info(f"Average timestep time: {avg_timestep_time:.2f}s")
        logger.info(f"Last timestep time: {recent_timestep_time:.2f}s")
        logger.info(f"LLM success rate: {llm_success_rate:.1f}% ({self.llm_success_count}/{self.llm_total_calls})")
        logger.info(f"Circuit breaker status: {'OPEN' if self.llm_circuit_open else 'CLOSED'}")
        logger.info(f"Estimated completion time: {estimated_completion_time:.1f}s")
        
        if self.simulation_start_time:
            elapsed_time = time.time() - self.simulation_start_time
            logger.info(f"Total elapsed time: {elapsed_time:.1f}s")
        
        logger.info("=" * 50)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of simulation health metrics.
        
        Returns:
            Dictionary containing health metrics
        """
        if not self.timestep_times:
            return {}
        
        avg_timestep_time = sum(self.timestep_times) / len(self.timestep_times)
        llm_success_rate = (self.llm_success_count / max(self.llm_total_calls, 1)) * 100 if self.llm_total_calls > 0 else 0
        
        return {
            'total_timesteps': len(self.timestep_times),
            'average_timestep_time': avg_timestep_time,
            'llm_total_calls': self.llm_total_calls,
            'llm_success_count': self.llm_success_count,
            'llm_failure_count': self.llm_failure_count,
            'llm_success_rate': llm_success_rate,
            'circuit_breaker_open': self.llm_circuit_open,
            'circuit_breaker_failure_count': self.llm_failure_count
        }
    
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
            'simulation_params': {
                'n_agents': self.n_agents,
                'num_timesteps': self.num_timesteps,
                'epsilon': self.epsilon,
                'theta': self.theta,
                'llm_enabled': self.llm_enabled,
                'noise_enabled': self.noise_enabled,
                'noise_mean': self.noise_mean,
                'noise_std': self.noise_std,
                'noise_clip': self.noise_clip,
            },
            'random_seed': self.random_seed,
            'topics': self.topics
        }
        # Always include timesteps for both LLM and non-LLM simulations
        results['timesteps'] = self.timesteps
        
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
            'is_running': self.is_running,
            'mean_opinion': self.mean_opinions[-1] if self.mean_opinions else 0.0,
            'std_opinion': self.std_opinions[-1] if self.std_opinions else 0.0
        } 

    def _safe_get_list_element(self, data_list: Optional[List], index: int, default_value: Any = None, list_name: str = "list") -> Any:
        """
        Safely get an element from a list, returning default if index is out of range.
        
        Args:
            data_list: List to access
            index: Index to access
            default_value: Default value if index is out of range
            list_name: Name of the list for logging purposes
            
        Returns:
            Element at index or default value
        """
        if data_list is None:
            logger.warning(f"Attempted to access {list_name}[{index}] but list is None, using default: {default_value}")
            return default_value
        
        if index >= len(data_list):
            logger.warning(f"Index {index} out of range for {list_name} (length: {len(data_list)}), using default: {default_value}")
            return default_value
        
        return data_list[index]
    
    def _safe_get_posts(self, posts: Optional[List[str]], index: int) -> str:
        """Safely get a post at the specified index."""
        return self._safe_get_list_element(posts, index, "No post available", "posts")
    
    def _safe_get_interpretation(self, interpretations: Optional[List[float]], index: int) -> float:
        """Safely get an interpretation at the specified index."""
        return self._safe_get_list_element(interpretations, index, 0.0, "interpretations")
    
    def _safe_get_individual_ratings(self, ratings: Optional[List[List[tuple[int, float]]]], index: int) -> List[tuple[int, float]]:
        """Safely get individual ratings at the specified index."""
        return self._safe_get_list_element(ratings, index, [], "individual_ratings") 