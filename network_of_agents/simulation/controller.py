"""
Streamlined simulation controller for the network of agents.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from tqdm import tqdm
import time
import logging

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
                 epsilon: float = 0.001,
                 num_timesteps: int = 50,
                 topics: Optional[List[str]] = None,
                 random_seed: Optional[int] = None,
                 llm_enabled: bool = True,
                 on_timestep: Optional[Callable[[Dict[str, Any], int], None]] = None,
                 resume_data: Optional[Dict[str, Any]] = None):
        """
        Initialize simulation controller.
        
        Args:
            n_agents: Number of agents in the network
            epsilon: Small positive parameter for numerical stability
            num_timesteps: Number of simulation timesteps
            llm_client: LLM client for post generation and interpretation
            topics: List of topics for opinions
            random_seed: Random seed for reproducible results

        """
        self.n_agents = n_agents
        self.num_timesteps = num_timesteps
        # Paper-consistent: no ER init, no lazy update probability
        self.llm_client = llm_client
        self.topics = topics
        self.random_seed = random_seed

        self.llm_enabled = llm_enabled
        self.on_timestep = on_timestep

        # DeGroot-only configuration
        self.epsilon = float(epsilon)

        # Metrics for comparing LLM vs pure DeGroot
        self.llm_opinion_history = []
        self.pure_degroot_opinion_history = []
        self.api_call_count = 0
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize components
        self.network = NetworkModel(n_agents, random_seed)
        self.agents = self._initialize_agents()
        
        # No susceptibility matrix in DeGroot-only mode
        self.susceptibility_matrix = None

        # Initialize network topology based on initial opinions (unless resuming)
        if resume_data is None:
            # Create a connected network for DeGroot model
            A0 = create_connected_degroot_network(self.n_agents, connectivity=0.05)
            self.network.adjacency_matrix = A0
        
        # Simulation state
        self.is_running = False
        self.current_timestep = 0
        
        # Minimal state (no circuit breaker/health)
        self.simulation_start_time = None
        
        # Data storage
        self.opinion_history = []
        self.mean_opinions = []
        self.std_opinions = []
        self.posts_history = []
        self.interpretations_history = []
        self.timesteps = []

        # If resuming, restore state
        if resume_data is not None:
            self.opinion_history = [np.array(x) for x in resume_data.get('opinion_history', [])]
            self.mean_opinions = resume_data.get('mean_opinions', [])
            self.std_opinions = resume_data.get('std_opinions', [])
            self.posts_history = resume_data.get('posts_history', [])
            self.interpretations_history = resume_data.get('interpretations_history', [])
            timesteps_data = resume_data.get('timesteps', [])
            # Handle both list and dict formats for timesteps
            if isinstance(timesteps_data, dict):
                # Convert dict to list, sorted by timestep number
                self.timesteps = [timesteps_data[str(i)] for i in sorted(int(k) for k in timesteps_data.keys())]
            else:
                self.timesteps = timesteps_data
            last_ops = None
            if self.opinion_history:
                last_ops = self.opinion_history[-1]
            elif resume_data.get('final_opinions') is not None:
                last_ops = np.array(resume_data['final_opinions'])
            if last_ops is not None and len(last_ops) == self.n_agents:
                for i in range(self.n_agents):
                    self.agents[i].update_opinion(float(last_ops[i]))
            # Rebuild adjacency from last timestep if available
            if self.timesteps:
                last_ts = self.timesteps[-1]
                A = np.zeros((self.n_agents, self.n_agents))
                for agent_info in last_ts.get('agents', []):
                    i = int(agent_info.get('agent_id'))
                    for j in agent_info.get('connected_agents', []):
                        A[i, j] = 1.0; A[j, i] = 1.0
                self.network.adjacency_matrix = A
            # Continue from next timestep
            self.current_timestep = max(0, len(self.mean_opinions))
    
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
        X_for_math = self._to_math_domain(X_current)
        X_next_math = update_opinions_pure_degroot(X_for_math, A_current, self.epsilon)
        return self._to_agent_domain(X_next_math)
    
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
        
        # Correlation coefficient
        correlation = np.corrcoef(llm_opinions, pure_opinions)[0, 1] if len(llm_opinions) > 1 else 0.0
        
        return {
            "mae": float(mae),
            "rmse": float(rmse), 
            "max_error": float(max_error),
            "correlation": float(correlation)
        }
    
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
        
        # Store initial state only on fresh runs
        if start_timestep == 0:
            if self.llm_enabled:
                self._store_current_state([], [], [])
            else:
                self._store_current_state(None, None, None)
        
        # Main simulation loop
        iterator = tqdm(range(start_timestep, self.num_timesteps), desc="Running simulation") if progress_bar else range(start_timestep, self.num_timesteps)
        
        for timestep in iterator:
            timestep_start_time = time.time()
            self.current_timestep = timestep
            
            # Step 1: Get current network state
            X_current = self._get_opinion_matrix()
            A_current = self.network.get_adjacency_matrix()
            A_prev = self.network.network_history[-1] if self.network.network_history else A_current
            
            # Step 2: Generate posts and interpretations (LLM) or skip (no-LLM)
            current_topic = self.topics[0]  # Use single topic for entire simulation (string or [A,B])
            
            if self.llm_enabled:
                # Prepare prior-timestep neighbor posts (raw text only)
                if self.posts_history and len(self.posts_history[-1]) == len(self.agents):
                    prev_posts = self.posts_history[-1]
                    neighbor_posts_per_agent = []
                    for i in range(self.n_agents):
                        connections = A_prev[i, :]
                        neighbor_indices = np.where(connections == 1)[0]
                        # Use prior timestep posts as-is (already name-prefixed)
                        neighbor_texts = [prev_posts[j] for j in neighbor_indices if prev_posts[j] and prev_posts[j].strip()]
                        neighbor_posts_per_agent.append(neighbor_texts)
                else:
                    neighbor_posts_per_agent = None

                # Determine which agents are connected (degree > 0) at current timestep
                degrees = np.sum(A_current, axis=1)
                connected_indices = [i for i in range(self.n_agents) if degrees[i] > 0]

                # If we have previous posts, start from them; else create placeholders
                if self.posts_history and len(self.posts_history[-1]) == len(self.agents):
                    posts_full = list(self.posts_history[-1])
                else:
                    posts_full = [f"Agent {i}: (no new post)" for i in range(self.n_agents)]

                # Generate only for connected agents
                if connected_indices:
                    agents_subset = [self.agents[i] for i in connected_indices]
                    if neighbor_posts_per_agent is not None:
                        neighbor_subset = [neighbor_posts_per_agent[i] for i in connected_indices]
                    else:
                        neighbor_subset = None
                    gen_subset = self.llm_client.generate_posts(current_topic, agents_subset, neighbor_subset)
                    if len(gen_subset) != len(connected_indices):
                        raise ValueError(f"Posts subset mismatch: got {len(gen_subset)}, expected {len(connected_indices)}")
                    for idx, i in enumerate(connected_indices):
                        posts_full[i] = gen_subset[idx]

                # Store posts for this timestep
                posts = posts_full
                self.posts_history.append(posts_full)

                # Pairwise neighbor ratings at time k over current adjacency A[k]
                R_pairwise, individual_ratings = self.llm_client.rate_posts_pairwise(posts, current_topic, self.agents, A_current)
                # Aggregate perceived opinion per post j as mean over raters i connected to j
                neighbor_opinions = []
                for j in range(self.n_agents):
                    col = R_pairwise[:, j]
                    if np.all(np.isnan(col)):
                        # Use poster's own current opinion if no ratings
                        neighbor_opinions.append(float(self.agents[j].get_opinion()))
                    else:
                        neighbor_opinions.append(float(np.nanmean(col)))
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

                # Count API calls (post generation + rating)
                if posts is not None:
                    # Count post generation calls (one per connected agent)
                    connected_agents = np.sum(A_current.sum(axis=1) > 0)
                    self.api_call_count += connected_agents

                    # Count rating calls (one per connected pair)
                    rating_calls = 0
                    for i in range(self.n_agents):
                        neighbors = np.where(A_current[i] == 1)[0]
                        rating_calls += len(neighbors)
                    self.api_call_count += rating_calls

                    logger.info(f"Timestep {timestep}: {connected_agents} post generation calls, {rating_calls} rating calls, "
                              f"MAE: {divergence_metrics['mae']:.4f}, Correlation: {divergence_metrics['correlation']:.4f}")
            else:
                X_interpreted = X_current.copy()

            X_for_math = self._to_math_domain(X_interpreted)
            X_next_math = update_opinions_pure_degroot(X_for_math, A_current, self.epsilon)
            # Network remains unchanged in pure DeGroot model
            X_next_agent = self._to_agent_domain(X_next_math)
            self._update_agent_opinions(X_next_agent)
            
            # Store LLM opinions for comparison
            if self.llm_enabled:
                self.llm_opinion_history.append(X_interpreted.copy())

            # Step 6: Store current state
            if self.llm_enabled:
                self._store_current_state(posts, neighbor_opinions, individual_ratings)
            else:
                self._store_current_state(None, None, None)
            
            # Record simple timing
            _ = time.time() - timestep_start_time

            # Invoke per-timestep callback with partial results snapshot
            if self.on_timestep is not None:
                partial_results = self._get_partial_simulation_results()
                partial_results['completed_timesteps'] = len(self.mean_opinions)
                partial_results['total_timesteps'] = self.num_timesteps
                self.on_timestep(partial_results, timestep)
        
        self.is_running = False
        return self._get_simulation_results()
    
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
                'llm_enabled': self.llm_enabled,
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
        Get current opinion vector.
        
        Returns:
            Current opinion vector (n_agents,)
        """
        return np.array([agent.get_opinion() for agent in self.agents])

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
    
    def _store_current_state(self, posts: Optional[List[str]], interpretations: Optional[List[float]], individual_ratings: Optional[List[List[tuple[int, float]]]]):
        """Store current simulation state."""
        opinions = self._get_opinion_matrix()
        self.opinion_history.append(opinions.copy())
        
        # Calculate mean and standard deviation
        mean_opinion = np.mean(opinions)
        std_opinion = np.std(opinions)
        
        self.mean_opinions.append(mean_opinion)
        self.std_opinions.append(std_opinion)
        
        # Store a per-timestep network snapshot
        if self.llm_enabled and posts is not None and interpretations is not None and individual_ratings is not None:
            self._store_detailed_agent_state(posts, interpretations, individual_ratings)
        else:
            self._store_basic_agent_state()
    
    def _store_detailed_agent_state(self, posts: List[str], interpretations: List[float], individual_ratings: List[List[tuple[int, float]]]):
        """Store detailed information about each agent at current timestep."""
        current_adjacency = self.network.get_adjacency_matrix()
        current_opinions = self._get_opinion_matrix()
        import re
        
        # Ensure lists have correct length to avoid index errors
        if len(posts) != self.n_agents:
            posts = [f"Agent {i}: (no post available)" for i in range(self.n_agents)]
        if len(interpretations) != self.n_agents:
            interpretations = [float(self.agents[i].get_opinion()) for i in range(self.n_agents)]
        if len(individual_ratings) != self.n_agents:
            individual_ratings = [[] for _ in range(self.n_agents)]
        
        # Build incoming ratings map: for each target j, collect (source i, rating)
        incoming_map: Dict[int, List[tuple[int, float]]] = {j: [] for j in range(self.n_agents)}
        try:
            for i, out_list in enumerate(individual_ratings or []):
                if not out_list:
                    continue
                for (j, r) in out_list:
                    try:
                        incoming_map[int(j)].append((int(i), float(r)))
                    except Exception:
                        continue
        except Exception:
            pass

        # Prepare reply detection: mentions like "Agent 7"
        mention_pattern = re.compile(r"\bAgent\s+(\d+)\b", flags=re.IGNORECASE)

        timestep_data = {
            'timestep': self.current_timestep,
            'agents': [],
            'reply_edges': []
        }
                
        for i, agent in enumerate(self.agents):
            # Get agent data
            agent_post = posts[i]
            agent_interpretation = interpretations[i]
            
            # Format ratings received by this agent (who rated them and what they rated)
            ratings_received = [{'from_agent': int(src), 'rated_opinion': float(r)} for (src, r) in incoming_map.get(i, [])]

            agent_data = {
                'agent_id': i,
                'actual_opinion': float(current_opinions[i]),
                'post': agent_post,
                'ratings_received': ratings_received
            }
            
            timestep_data['agents'].append(agent_data)

            # Extract reply targets from post text
            try:
                text = str(agent_post or "")
                targets: List[int] = []
                for m in mention_pattern.findall(text):
                    try:
                        j = int(m)
                        if j != i:
                            targets.append(j)
                    except Exception:
                        continue
                if targets:
                    for j in sorted(set(targets)):
                        timestep_data['reply_edges'].append({'source': int(i), 'target': int(j)})
            except Exception:
                pass
        
        self.timesteps.append(timestep_data)
    
    def _store_basic_agent_state(self):
        """Store basic agent information for non-LLM simulations."""
        current_opinions = self._get_opinion_matrix()
        
        timestep_data = {
            'timestep': self.current_timestep,
            'agents': []
        }
                
        for i, agent in enumerate(self.agents):
            agent_data = {
                'agent_id': i,
                'actual_opinion': float(current_opinions[i]),
                'post': "No post available (LLM disabled)",
                'ratings_received': []
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
            'llm_enabled': self.llm_enabled,
            },
            'random_seed': self.random_seed,
            'topics': self.topics
        }
        # Always include timesteps for both LLM and non-LLM simulations
        results['timesteps'] = self.timesteps
        
        if self.llm_enabled:
            results['posts_history'] = self.posts_history
            results['interpretations_history'] = self.interpretations_history
            
            # Add LLM vs Pure DeGroot comparison metrics
            if len(self.llm_opinion_history) > 0:
                results['llm_vs_pure_degroot'] = {
                    'llm_opinion_history': [op.tolist() for op in self.llm_opinion_history],
                    'pure_degroot_opinion_history': [op.tolist() for op in self.pure_degroot_opinion_history],
                    'api_call_count': self.api_call_count,
                    'final_divergence': self._calculate_llm_degroot_divergence(
                        self.llm_opinion_history[-1], 
                        self.pure_degroot_opinion_history[-1]
                    ) if len(self.llm_opinion_history) > 0 else {}
                }
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
