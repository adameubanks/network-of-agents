"""
Streamlined simulation controller for the network of agents.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from tqdm import tqdm
import time
import logging
import concurrent.futures as cf

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
        self.model = model

        self.llm_enabled = llm_enabled
        self.on_timestep = on_timestep
        self.progress_callback = progress_callback

        # DeGroot-only configuration
        self.epsilon = float(epsilon)

        # Metrics for comparing LLM vs pure DeGroot
        self.llm_opinion_history = []
        self.pure_degroot_opinion_history = []
        
        # LLM simulation data storage
        self.posts_history = []
        self.interpretations_history = []
        self.individual_ratings_history = []
        self.neighbor_opinions_history = []
        self.pre_update_opinions_history = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize components
        self.network = NetworkModel(n_agents, random_seed)
        self.agents = self._initialize_agents()
        
        # Store initial opinions for Friedkin-Johnsen model
        self.initial_opinions = np.array([agent.get_opinion() for agent in self.agents])
        
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
        
        # Store initial state only on fresh runs
        if start_timestep == 0:
            if self.llm_enabled:
                self._store_current_state([], [], [])
            else:
                self._store_current_state(None, None, None)
        
        # Main simulation loop with enhanced progress tracking
        if progress_bar:
            topic_name = self.topics[0] if self.topics else "mathematical"
            iterator = tqdm(range(start_timestep, self.num_timesteps), 
                          desc=f"🔄 {topic_name} ({self.n_agents} agents)", 
                          unit="step", 
                          position=3,  # Position 3 for individual timestep progress
                          leave=False)  # Don't keep visible to avoid clutter
        else:
            iterator = range(start_timestep, self.num_timesteps)
        
        for timestep in iterator:
            timestep_start_time = time.time()
            self.current_timestep = timestep
            
            # Update progress bar with current status
            if progress_bar and self.llm_enabled:
                iterator.set_postfix(
                    step=f"{timestep + 1}/{self.num_timesteps}",
                    status="🤖 LLM processing..."
                )
            
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

                logger.info(f"Timestep {timestep}: MAE: {divergence_metrics['mae']:.4f}, Correlation: {divergence_metrics['correlation']:.4f}")
            else:
                X_interpreted = X_current.copy()

            X_for_math = self._to_math_domain(X_interpreted)
            
            # Use the specified model for opinion updates
            if self.model == "degroot":
                X_next_math = update_opinions_pure_degroot(X_for_math, A_current, self.epsilon)
            elif self.model == "friedkin_johnsen":
                # For Friedkin-Johnsen, we need initial opinions and lambda values
                # Using default lambda=0.1 for all agents (moderate stubbornness)
                lambda_values = np.full(self.n_agents, 0.1)
                X_0_math = self._to_math_domain(self.initial_opinions)
                X_next_math = update_opinions_friedkin_johnsen(X_for_math, A_current, lambda_values, X_0_math, self.epsilon)
            else:
                raise ValueError(f"Unknown model: {self.model}")
                
            # Network remains unchanged in pure models
            X_next_agent = self._to_agent_domain(X_next_math)
            self._update_agent_opinions(X_next_agent)
            
            # Store LLM opinions for comparison
            if self.llm_enabled:
                self.llm_opinion_history.append(X_interpreted.copy())
            
            # No early convergence check - run for full specified timesteps

            # Step 6: Store current state
            if self.llm_enabled:
                self._store_current_state(posts, neighbor_opinions, individual_ratings, pre_update_opinions)
            else:
                self._store_current_state(None, None, None)
            
            # Record timing
            timestep_time = time.time() - timestep_start_time

            # Update progress bar with completion status
            if progress_bar:
                mean_opinion = self.mean_opinions[-1] if self.mean_opinions else 0.0
                iterator.set_postfix(
                    step=f"{timestep + 1}/{self.num_timesteps}",
                    mean=f"{mean_opinion:.3f}",
                    time=f"{timestep_time:.1f}s"
                )
            
            # Call progress callback if provided (this updates the higher-level progress bars)
            if self.progress_callback:
                self.progress_callback(timestep + 1, self.num_timesteps)

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
    
    def _store_current_state(self, posts: Optional[List[str]], neighbor_opinions: Optional[List[float]], individual_ratings: Optional[List[List[tuple[int, float]]]], pre_update_opinions: Optional[np.ndarray] = None):
        """Store current simulation state."""
        opinions = self._get_opinion_matrix()
        self.opinion_history.append(opinions.copy())
        
        # Calculate mean and standard deviation
        mean_opinion = np.mean(opinions)
        std_opinion = np.std(opinions)
        
        self.mean_opinions.append(mean_opinion)
        self.std_opinions.append(std_opinion)
        
        # Store LLM data if available (posts are already stored in main loop)
        if self.llm_enabled and posts is not None and neighbor_opinions is not None and individual_ratings is not None:
            self.interpretations_history.append(neighbor_opinions.copy())
            self.individual_ratings_history.append(individual_ratings.copy())
            if pre_update_opinions is not None:
                self.pre_update_opinions_history.append(pre_update_opinions.copy())
            self._store_detailed_agent_state(posts, neighbor_opinions, individual_ratings, pre_update_opinions)
        else:
            self._store_basic_agent_state()
    
    def _store_detailed_agent_state(self, posts: List[str], neighbor_opinions: List[float], individual_ratings: List[List[tuple[int, float]]], pre_update_opinions: np.ndarray = None):
        """Store detailed information about each agent at current timestep."""
        current_adjacency = self.network.get_adjacency_matrix()
        current_opinions = pre_update_opinions if pre_update_opinions is not None else self._get_opinion_matrix()
        import re
        
        # Ensure lists have correct length to avoid index errors
        if len(posts) != self.n_agents:
            posts = [f"Agent {i}: (no post available)" for i in range(self.n_agents)]
        if len(neighbor_opinions) != self.n_agents:
            neighbor_opinions = [float(self.agents[i].get_opinion()) for i in range(self.n_agents)]
        if len(individual_ratings) != self.n_agents:
            individual_ratings = [[] for _ in range(self.n_agents)]
        
        # Build ratings matrix: who rated whom and what they rated
        ratings_matrix = {}
        try:
            for i, out_list in enumerate(individual_ratings or []):
                if not out_list:
                    continue
                for (j, r) in out_list:
                    try:
                        if j not in ratings_matrix:
                            ratings_matrix[j] = {}
                        ratings_matrix[int(j)][int(i)] = float(r)
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
            agent_interpretation = neighbor_opinions[i]
            
            # Format ratings received by this agent (who rated them and what they rated)
            ratings_received = []
            if i in ratings_matrix:
                for rater_id, rating in ratings_matrix[i].items():
                    ratings_received.append({
                        'rater_agent': int(rater_id), 
                        'rating': float(rating),
                        'rater_opinion': float(current_opinions[rater_id])
                    })

            agent_data = {
                'agent_id': i,
                'opinion': float(current_opinions[i]),
                'post': agent_post,
                'ratings_received': ratings_received,
                'neighbor_opinions': [float(current_opinions[j]) for j in range(self.n_agents) if current_adjacency[i, j] == 1]
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
        
    
    def _get_simulation_results(self) -> Dict[str, Any]:
        """
        Get simulation results in the new nested structure.
        
        Returns:
            Dictionary containing simulation results nested by model -> topic -> timestep -> agent interactions
        """
        # Get the current topic (assuming single topic per simulation)
        if self.topics and len(self.topics) > 0:
            current_topic = self.topics[0]
        elif not self.llm_enabled:
            current_topic = "pure_math_model"
        else:
            current_topic = "unknown"
        
        # Restructure timesteps data into the new nested format
        timesteps_nested = {}
        for timestep_data in self.timesteps:
            timestep_num = timestep_data['timestep']
            timesteps_nested[str(timestep_num)] = {
                'agents': {}
            }
            
            # Convert agent list to nested agent dict
            for agent_data in timestep_data['agents']:
                agent_id = str(agent_data['agent_id'])
                
                # Build interpretations_received dict from ratings_received
                interpretations_received = {}
                for rating_data in agent_data.get('ratings_received', []):
                    rater_id = str(rating_data['rater_agent'])
                    interpretations_received[rater_id] = rating_data['rating']
                
                # Get connected agent IDs from adjacency matrix
                current_adjacency = self.network.get_adjacency_matrix()
                connected_agents = [str(j) for j in range(self.n_agents) if current_adjacency[int(agent_id), j] == 1]
                
                # Handle both LLM and non-LLM agent data structures
                opinion_key = 'opinion' if 'opinion' in agent_data else 'actual_opinion'
                
                timesteps_nested[str(timestep_num)]['agents'][agent_id] = {
                    'opinion': agent_data[opinion_key],
                    'post': agent_data['post'],
                    'connected_to': connected_agents,
                    'interpretations_received': interpretations_received
                }
        
        # Build the new nested structure
        results = {
            'experiment_metadata': {
                'model': self.model,
                'topology': 'unknown',  # Will be set by runner
                'n_agents': self.n_agents,
                'num_timesteps': self.num_timesteps,
                'llm_enabled': self.llm_enabled,
                'random_seed': self.random_seed,
                'topics': self.topics,
                'convergence_metrics': {
                    'final_mean_opinion': self.mean_opinions[-1] if self.mean_opinions else 0.0,
                    'final_std_opinion': self.std_opinions[-1] if self.std_opinions else 0.0,
                    'converged': len(self.mean_opinions) < self.num_timesteps
                }
            },
            'results': {
                current_topic: {
                    'timesteps': timesteps_nested,
                    'summary_metrics': {
                        'opinion_history': [op.tolist() for op in self.opinion_history],
                        'mean_opinions': self.mean_opinions,
                        'std_opinions': self.std_opinions,
                        'final_opinions': self._get_opinion_matrix().tolist(),
                        'network_info': {
                            'adjacency_matrix': self.network.get_adjacency_matrix().tolist(),
                            'n_agents': self.n_agents
                        }
                    }
                }
            }
        }
        
        # Add LLM-specific data if enabled
        if self.llm_enabled:
            results['results'][current_topic]['llm_analysis'] = {
                'posts_history': self.posts_history,
                'interpretations_history': self.interpretations_history,
                'individual_ratings': self.individual_ratings_history,
                'neighbor_opinions': self.neighbor_opinions_history,
                'pre_update_opinions': [op.tolist() for op in self.pre_update_opinions_history]
            }
            
            # Add LLM vs Pure DeGroot comparison metrics
            if len(self.llm_opinion_history) > 0:
                results['results'][current_topic]['llm_vs_pure_degroot'] = {
                    'llm_opinion_history': [op.tolist() for op in self.llm_opinion_history],
                    'pure_degroot_opinion_history': [op.tolist() for op in self.pure_degroot_opinion_history],
                    'final_divergence': self._calculate_llm_degroot_divergence(
                        self.llm_opinion_history[-1], 
                        self.pure_degroot_opinion_history[-1]
                    ) if len(self.llm_opinion_history) > 0 else {},
                    'timestep_divergence': self._calculate_timestep_divergence()
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
