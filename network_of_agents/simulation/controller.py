"""
Simulation controller for managing the main simulation loop and data collection.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import time

from ..core.mathematics import update_opinions, update_edges
from ..network.graph_model import NetworkModel
from ..llm.agent import LLMAgent
from ..llm.litellm_client import LiteLLMClient
from ..data.storage import SimulationDataStorage


class SimulationController:
    """
    Main controller for the social network simulation.
    """
    
    def __init__(self, 
                 n_agents: int = 50,
                 n_topics: int = 3,
                 epsilon: float = 1e-6,
                 theta: int = 7,
                 num_timesteps: int = 180,
                 initial_connection_probability: float = 0.2,
                 llm_client: Optional[LiteLLMClient] = None,
                 topics: Optional[List[str]] = None,
                 agent_personas: Optional[List[str]] = None):
        """
        Initialize the simulation controller.
        
        Args:
            n_agents: Number of agents in the network
            n_topics: Number of topics for opinions
            epsilon: Small positive parameter for mathematical operations
            theta: Parameter for edge formation
            num_timesteps: Total number of simulation steps
            initial_connection_probability: Probability of initial connections
            llm_client: LiteLLM client for opinion generation
            topics: List of topics for opinions
            agent_personas: List of agent personas
        """
        self.n_agents = n_agents
        self.n_topics = n_topics
        self.epsilon = epsilon
        self.theta = theta
        self.num_timesteps = num_timesteps
        self.initial_connection_probability = initial_connection_probability
        
        # Initialize components
        self.llm_client = llm_client
        self.topics = topics or self._generate_default_topics()
        self.agent_personas = agent_personas or self._generate_default_personas()
        
        # Initialize network and agents
        self.network = NetworkModel(n_agents, initial_connection_probability)
        self.agents = self._initialize_agents()
        self.network.add_agents(self.agents)
        
        # Initialize data storage
        self.data_storage = SimulationDataStorage()
        
        # Simulation state
        self.current_timestep = 0
        self.is_running = False
        self.callbacks = []
    
    def _generate_default_topics(self) -> List[str]:
        """Generate default topics for the simulation."""
        return ["Topic 1", "Topic 2", "Topic 3"]
    
    def _generate_default_personas(self) -> List[str]:
        """Generate default agent personas."""
        personas = [
            "A conservative individual who values tradition and stability",
            "A liberal individual who values progress and change",
            "A moderate individual who seeks balance and compromise",
            "A libertarian individual who values individual freedom",
            "A progressive individual who advocates for social justice",
            "A traditionalist individual who respects established norms",
            "An activist individual who fights for causes they believe in",
            "A pragmatist individual who focuses on practical solutions",
            "An idealist individual who believes in perfect solutions",
            "A realist individual who accepts the world as it is"
        ]
        
        # Repeat personas if needed
        while len(personas) < self.n_agents:
            personas.extend(personas[:self.n_agents - len(personas)])
        
        return personas[:self.n_agents]
    
    def _initialize_agents(self) -> List[LLMAgent]:
        """Initialize LLM agents with personas and opinions."""
        agents = []
        
        for i in range(self.n_agents):
            persona = self.agent_personas[i]
            agent = LLMAgent(
                agent_id=i,
                persona=persona,
                topics=self.topics,
                llm_client=self.llm_client
            )
            
            # Initialize opinions using LLM if available, otherwise use random
            if self.llm_client:
                agent.initialize_opinions(self.topics, self.llm_client)
            else:
                # Fallback to random opinions
                random_opinions = np.random.rand(self.n_topics)
                agent.opinions = random_opinions
            
            agents.append(agent)
        
        return agents
    
    def add_callback(self, callback: Callable[[int, Dict[str, Any]], None]):
        """
        Add a callback function to be called at each timestep.
        
        Args:
            callback: Function to call with (timestep, data) arguments
        """
        self.callbacks.append(callback)
    
    def run_simulation(self, progress_bar: bool = True) -> Dict[str, Any]:
        """
        Run the complete simulation.
        
        Args:
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary containing simulation results
        """
        self.is_running = True
        self.current_timestep = 0
        
        # Initialize data storage
        self.data_storage.initialize(self.n_agents, self.n_topics, self.num_timesteps)
        
        # Store initial state
        self._store_current_state()
        
        # Main simulation loop
        iterator = tqdm(range(self.num_timesteps), desc="Running simulation") if progress_bar else range(self.num_timesteps)
        
        for timestep in iterator:
            self.current_timestep = timestep
            
            # Update opinions
            X_current = self._get_opinion_matrix()
            A_current = self.network.get_adjacency_matrix()
            
            X_next = update_opinions(X_current, A_current, self.epsilon)
            self._update_agent_opinions(X_next)
            
            # Update network topology
            A_next = update_edges(A_current, X_next, self.theta, self.epsilon)
            self.network.update_adjacency_matrix(A_next)
            
            # Store current state
            self._store_current_state()
            
            # Call callbacks
            self._call_callbacks(timestep)
        
        self.is_running = False
        return self.data_storage.get_simulation_results()
    
    def _get_opinion_matrix(self) -> np.ndarray:
        """Get the current opinion matrix from all agents."""
        opinion_matrix = np.zeros((self.n_agents, self.n_topics))
        
        for i, agent in enumerate(self.agents):
            opinion_matrix[i, :] = agent.get_opinions()
        
        return opinion_matrix
    
    def _update_agent_opinions(self, new_opinions: np.ndarray):
        """Update opinions for all agents."""
        for i, agent in enumerate(self.agents):
            agent.update_opinions(new_opinions[i, :])
    
    def _store_current_state(self):
        """Store the current simulation state."""
        # Store opinion matrix
        opinion_matrix = self._get_opinion_matrix()
        self.data_storage.store_opinions(self.current_timestep, opinion_matrix)
        
        # Store adjacency matrix
        adjacency_matrix = self.network.get_adjacency_matrix()
        self.data_storage.store_adjacency(self.current_timestep, adjacency_matrix)
        
        # Store network metrics
        metrics = {
            'density': self.network.get_network_density(),
            'average_degree': self.network.get_average_degree(),
            'clustering_coefficient': self.network.get_clustering_coefficient(),
            'num_components': len(self.network.get_connected_components()),
            'echo_chambers': len(self.network.get_echo_chambers())
        }
        self.data_storage.store_metrics(self.current_timestep, metrics)
        
        # Store agent-specific data
        agent_data = []
        for agent in self.agents:
            agent_data.append({
                'agent_id': agent.agent_id,
                'persona': agent.persona,
                'opinions': agent.get_opinions().tolist(),
                'degree': agent.get_degree(adjacency_matrix)
            })
        self.data_storage.store_agent_data(self.current_timestep, agent_data)
    
    def _call_callbacks(self, timestep: int):
        """Call all registered callbacks."""
        data = {
            'timestep': timestep,
            'opinions': self._get_opinion_matrix(),
            'adjacency': self.network.get_adjacency_matrix(),
            'metrics': {
                'density': self.network.get_network_density(),
                'average_degree': self.network.get_average_degree(),
                'clustering_coefficient': self.network.get_clustering_coefficient(),
                'num_components': len(self.network.get_connected_components()),
                'echo_chambers': len(self.network.get_echo_chambers())
            }
        }
        
        for callback in self.callbacks:
            try:
                callback(timestep, data)
            except Exception as e:
                print(f"Error in callback: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current simulation state.
        
        Returns:
            Dictionary containing current state
        """
        return {
            'timestep': self.current_timestep,
            'is_running': self.is_running,
            'opinions': self._get_opinion_matrix(),
            'adjacency': self.network.get_adjacency_matrix(),
            'network_metrics': {
                'density': self.network.get_network_density(),
                'average_degree': self.network.get_average_degree(),
                'clustering_coefficient': self.network.get_clustering_coefficient(),
                'num_components': len(self.network.get_connected_components()),
                'echo_chambers': len(self.network.get_echo_chambers())
            },
            'agents': [agent.to_dict() for agent in self.agents]
        }
    
    def save_simulation(self, filename: str):
        """
        Save the simulation state to a file.
        
        Args:
            filename: Name of the file to save to
        """
        simulation_data = {
            'parameters': {
                'n_agents': self.n_agents,
                'n_topics': self.n_topics,
                'epsilon': self.epsilon,
                'theta': self.theta,
                'num_timesteps': self.num_timesteps,
                'initial_connection_probability': self.initial_connection_probability,
                'topics': self.topics
            },
            'current_state': self.get_current_state(),
            'simulation_data': self.data_storage.get_simulation_results()
        }
        
        self.data_storage.save_to_file(filename, simulation_data)
    
    def load_simulation(self, filename: str):
        """
        Load a simulation state from a file.
        
        Args:
            filename: Name of the file to load from
        """
        simulation_data = self.data_storage.load_from_file(filename)
        
        # Update parameters
        params = simulation_data['parameters']
        self.n_agents = params['n_agents']
        self.n_topics = params['n_topics']
        self.epsilon = params['epsilon']
        self.theta = params['theta']
        self.num_timesteps = params['num_timesteps']
        self.initial_connection_probability = params['initial_connection_probability']
        self.topics = params['topics']
        
        # Update current state
        current_state = simulation_data['current_state']
        self.current_timestep = current_state['timestep']
        self.is_running = current_state['is_running']
        
        # Reconstruct agents
        self.agents = [LLMAgent.from_dict(agent_data) for agent_data in current_state['agents']]
        
        # Reconstruct network
        self.network = NetworkModel(self.n_agents, self.initial_connection_probability)
        self.network.add_agents(self.agents)
        self.network.adjacency_matrix = np.array(current_state['adjacency'])
        
        # Load simulation data
        self.data_storage.load_simulation_data(simulation_data['simulation_data']) 