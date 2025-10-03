"""
Simplified experiment runner for opinion dynamics research.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from .simulation.controller import Controller
from .llm_client import LLMClient
from .network.generator import create_network_model, get_network_params
from .core.mathematics import (
    update_opinions_pure_degroot, update_opinions_friedkin_johnsen,
    initialize_opinions_normal, get_network_info
)

# Topic mapping to match experimental design
TOPIC_MAPPING = {
    "immigration": (
        "Immigration is a good thing for this country",
        "Immigration is a bad thing for this country"
    ),
    "environment_economy": (
        "Prioritize environmental protection even if growth is curbed",
        "Prioritize economic growth even if the environment suffers"
    ),
    "corporate_activism": (
        "Companies should make statements about political/social issues",
        "Companies should not make statements about political/social issues"
    ),
    "gun_safety": (
        "Gun ownership increases safety",
        "Gun ownership reduces safety"
    ),
    "social_media_democracy": (
        "Social media has been good for democracy",
        "Social media has been bad for democracy"
    ),
    "toilet_paper": (
        "Toilet paper should go over the roll",
        "Toilet paper should go under the roll"
    ),
    "hot_dog_sandwich": (
        "A hot dog is a sandwich",
        "A hot dog is not a sandwich"
    ),
    "child_free_weddings": (
        "Child-free weddings are appropriate",
        "Child-free weddings are inappropriate"
    ),
    "restaurant_etiquette": (
        "Snapping fingers to get waiter attention is acceptable",
        "Snapping fingers to get waiter attention is unacceptable"
    ),
    "human_cloning": (
        "Human cloning is morally acceptable",
        "Human cloning is morally wrong"
    )
}

logger = logging.getLogger(__name__)

def get_topic_framing(topic_key: str) -> Tuple[str, str]:
    """Get the proper A vs B framing for a topic key."""
    if topic_key in TOPIC_MAPPING:
        return TOPIC_MAPPING[topic_key]
    else:
        # Fallback for unknown topics
        return (f"Option A for {topic_key}", f"Option B for {topic_key}")

class Runner:
    """Experiment runner for opinion dynamics research."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.output_dir = config.get("output_dir", "results")
        
        # Initialize LLM client if needed
        self.llm_client = None
        llm_config = config.get("llm", {})
        if llm_config.get("enabled", False):
            self.llm_client = LLMClient(
                model_name=llm_config.get("model", "gpt-5-mini"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
    
    def run_experiment(self, network_type: str, topic: str = None, 
                      n_agents: int = 50, max_steps: int = 100) -> Dict[str, Any]:
        """Run a single experiment"""
        logger.info(f"Running experiment: {network_type} with {n_agents} agents")
        
        # Get network parameters
        network_params = get_network_params(network_type, n_agents)
        
        # Create network
        network = create_network_model(network_type, network_params, 
                                     random_seed=self.config.get("random_seed", 42))
        adjacency = network.get_adjacency_matrix()
        
        # Use actual number of agents from the network (important for karate)
        actual_n_agents = adjacency.shape[0]
        
        # Initialize opinions
        opinions = initialize_opinions_normal(actual_n_agents, random_seed=self.config.get("random_seed", 42))
        opinion_history = [opinions.copy()]
        
        # Run simulation
        if self.llm_client and topic:
            # Get proper topic framing
            topic_framing = get_topic_framing(topic)
            
            # LLM simulation
            controller = Controller(
                llm_client=self.llm_client,
                n_agents=actual_n_agents,
                epsilon=1e-6,
                num_timesteps=max_steps,
                topics=[topic_framing],  # Pass the proper A vs B framing
                random_seed=self.config.get("random_seed", 42),
                llm_enabled=True
            )
            controller.network.adjacency_matrix = adjacency
            results = controller.run_simulation(progress_bar=False)
            
            return {
                "network_type": network_type,
                "topic": topic,
                "n_agents": actual_n_agents,
                "max_steps": max_steps,
                "final_opinions": results["final_opinions"],
                "opinion_history": results["opinion_history"],
                "mean_opinions": results["mean_opinions"],
                "network_info": get_network_info(adjacency),
                "llm_enabled": True,
                "timesteps": results.get("timesteps", []),
                "llm_vs_pure_degroot": results.get("llm_vs_pure_degroot", {}),
                "convergence_step": results.get("convergence_step", max_steps)
            }
        else:
            # Mathematical simulation
            for t in range(max_steps):
                math_opinions = (opinions + 1) / 2
                
                if network_type == "stubborn":
                    lambda_values = np.random.uniform(0.1, 0.9, actual_n_agents)
                    X_0 = math_opinions.copy()
                    new_math_opinions = update_opinions_friedkin_johnsen(
                        math_opinions, adjacency, lambda_values, X_0, epsilon=1e-6
                    )
                else:
                    new_math_opinions = update_opinions_pure_degroot(
                        math_opinions, adjacency, epsilon=1e-6
                    )
                
                opinions = 2 * new_math_opinions - 1
                opinion_history.append(opinions.copy())
                
                # Check convergence
                if t > 10 and np.mean(np.abs(opinions - opinion_history[-2])) < 1e-6:
                    break
            
            return {
                "network_type": network_type,
                "topic": topic,
                "n_agents": actual_n_agents,
                "max_steps": max_steps,
                "final_opinions": opinions.tolist(),
                "opinion_history": [op.tolist() for op in opinion_history],
                "mean_opinions": [np.mean(op) for op in opinion_history],
                "network_info": get_network_info(adjacency),
                "llm_enabled": False,
                "convergence_step": t + 1
            }
    
    def run_batch(self, network_types: List[str], topics: List[str] = None, 
                  n_agents: int = 50, max_steps: int = 100) -> Dict[str, Any]:
        """Run a batch of experiments"""
        logger.info(f"Running batch: {len(network_types)} networks, {len(topics) if topics else 0} topics")
        
        results = {}
        
        for network_type in network_types:
            results[network_type] = {}
            
            if topics:
                for topic in topics:
                    logger.info(f"  {network_type} + {topic}")
                    try:
                        result = self.run_experiment(network_type, topic, n_agents, max_steps)
                        results[network_type][topic] = result
                    except Exception as e:
                        logger.error(f"  Failed: {e}")
                        results[network_type][topic] = {"error": str(e)}
            else:
                logger.info(f"  {network_type} (mathematical)")
                try:
                    result = self.run_experiment(network_type, None, n_agents, max_steps)
                    results[network_type] = result
                except Exception as e:
                    logger.error(f"  Failed: {e}")
                    results[network_type] = {"error": str(e)}
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results to JSON file"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Determine experiment type based on results
            if len(results) > 1:
                # Multiple networks - full experiment
                filename = f"convergence_all_networks_{timestamp}.json"
            elif len(results) == 1:
                # Single network
                network_name = list(results.keys())[0]
                filename = f"single_network_{network_name}_{timestamp}.json"
            else:
                # Fallback
                filename = f"experiment_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj

def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run an experiment with the given configuration"""
    runner = Runner(config)
    
    # Get experiment parameters
    network_types = config.get("networks", ["smallworld", "scalefree", "random", "echo", "karate"])
    topics = config.get("topics", [])
    n_agents = config.get("n_agents", 50)
    max_steps = config.get("max_steps", 100)
    
    # Run experiments
    if topics:
        results = runner.run_batch(network_types, topics, n_agents, max_steps)
    else:
        results = runner.run_batch(network_types, None, n_agents, max_steps)
    
    # Save results
    runner.save_results(results)
    
    return results
