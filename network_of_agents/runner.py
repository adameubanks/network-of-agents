"""
Simplified experiment runner for opinion dynamics research.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .simulation.controller import Controller
from .llm_client import LLMClient
from .network.generator import create_network_model, get_network_params

TOPIC_MAPPING = {
    # Source: Gallup Poll
    "immigration": (
        "immigration is a good thing for this country",
        "immigration is a bad thing for this country"
    ),

    # Source: Gallup Poll
    "environment_economy": (
        "protection of the environment should be given priority, even at the risk of curbing economic growth",
        "economic growth should be given priority, even if the environment suffers to some extent"
    ),

    # Source: Ipsos/Axios Poll (Note: "appropriate" vs. "inappropriate")
    "corporate_activism": (
        "it is appropriate for American companies to make public statements about political or social issues",
        "it is inappropriate for American companies to make public statements about political or social issues"
    ),

    # Source: Pew Research Center
    "gun_safety": (
        "Gun ownership in this country does more to increase safety by allowing law-abiding citizens to protect themselves",
        "Gun ownership in this country does more to reduce safety by giving too many people access to firearms and increasing the risk of misuse"
    ),

    # Source: Pew Research Center
    "social_media_democracy": (
        "social media has been a good thing for democracy",
        "social media has been a bad thing for democracy"
    ),
    
    # Source: Gallup Poll
    "human_cloning": (
        "cloning humans is morally acceptable",
        "cloning humans is morally wrong"
    ),

    # --- Topics without direct survey matches found ---
    
    "toilet_paper": (
        "toilet paper should go over the roll",
        "toilet paper should go under the roll"
    ),

    "hot_dog_sandwich": (
        "a hot dog is a sandwich",
        "a hot dog is not a sandwich"
    ),

    "child_free_weddings": (
        "child-free weddings are appropriate",
        "child-free weddings are inappropriate"
    ),

    "restaurant_etiquette": (
        "snapping fingers to get waiter attention is acceptable",
        "snapping fingers to get waiter attention is unacceptable"
    )
}

logger = logging.getLogger(__name__)

def _get_convergence_steps(network_type: str, model: str, config: Dict[str, Any]) -> int:
    """Get convergence steps for network type from config."""
    if not config or "convergence_steps" not in config:
        raise ValueError(f"convergence_steps not specified in config for network type: {network_type}")
    
    convergence_steps = config["convergence_steps"]
    if isinstance(convergence_steps, dict):
        if network_type not in convergence_steps:
            raise ValueError(f"convergence_steps not specified for network type '{network_type}' in config")
        return convergence_steps[network_type]
    elif isinstance(convergence_steps, int):
        return convergence_steps
    else:
        raise ValueError(f"convergence_steps must be int or dict, got {type(convergence_steps)}")

def get_topic_framing(topic_key: str, reversed: bool = False) -> Tuple[str, str]:
    """Get topic framing for LLM prompts."""
    framing = TOPIC_MAPPING.get(topic_key, (f"Option A for {topic_key}", f"Option B for {topic_key}"))
    return (framing[1], framing[0]) if reversed else framing

class Runner:
    """Experiment runner for opinion dynamics research."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.output_dir = config.get("output_dir", "results")
        
        self.llm_client = None
        llm_config = config.get("llm", {})
        if llm_config.get("enabled", False):
            if "model" not in llm_config:
                raise ValueError("llm.model must be specified in config")
            if "max_tokens" not in llm_config:
                raise ValueError("llm.max_tokens must be specified in config")
            if "max_workers" not in llm_config:
                raise ValueError("llm.max_workers must be specified in config")
            if "timeout" not in llm_config:
                raise ValueError("llm.timeout must be specified in config")
            
            self.llm_client = LLMClient(
                model_name=llm_config["model"],
                api_key=os.getenv("OPENAI_API_KEY"),
                max_workers=llm_config["max_workers"],
                timeout=llm_config["timeout"],
                max_tokens=llm_config["max_tokens"]
            )
    
    def run_experiment(self, network_type: str, topic: str = None, 
                      n_agents: int = 50, model: str = "degroot", reversed: bool = False) -> Dict[str, Any]:
        """Run a single experiment"""
        logger.info(f"Running experiment: {network_type} with {n_agents} agents")
        
        # Get network parameters and create network
        network_params = get_network_params(network_type, n_agents)
        network = create_network_model(network_type, network_params, 
                                     random_seed=self.config.get("random_seed", 42))
        adjacency = network.get_adjacency_matrix()
        actual_n_agents = adjacency.shape[0]
        
        # Get convergence steps
        max_steps = _get_convergence_steps(network_type, model, self.config)
        
        # Create experiment directory structure
        timestamp = datetime.now().strftime("%m-%d-%H-%M")
        experiment_name = self.config.get("experiment", {}).get("name", "experiment")
        experiment_name = experiment_name.lower().replace(" ", "_").replace("-", "_")
        experiment_dir = f"{experiment_name}_{timestamp}"
        experiment_path = os.path.join(self.output_dir, experiment_dir)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Create plots directory
        plots_path = os.path.join(experiment_path, "plots")
        os.makedirs(plots_path, exist_ok=True)
        
        # Create topic-specific output file (temporary for streaming)
        topic_name = topic or "pure_math_model"
        output_file = os.path.join(experiment_path, f"{topic_name}_streaming.json")
        
        # Create controller with streaming output
        controller = Controller(
            llm_client=self.llm_client,
            n_agents=actual_n_agents,
            epsilon=self.config.get("epsilon"),
            connectivity=self.config.get("connectivity"),
            num_timesteps=max_steps,
            topics=[topic] if topic else None,
            random_seed=self.config.get("random_seed", 42),
            llm_enabled=self.llm_client is not None and topic is not None,
            model=model,
            reversed=reversed,
            output_file=output_file
        )
        controller.network.adjacency_matrix = adjacency
        
        # Run simulation
        results = controller.run_simulation(progress_bar=True)
        
        # Update topology in results
        if "experiment_metadata" in results:
            results["experiment_metadata"]["topology"] = network_type
        
        # Add streaming file path for later use
        results["streaming_file"] = output_file
        
        return results
    
    def run_batch(self, network_types: List[str], topics: List[str] = None, 
                  n_agents: int = 50, models: List[str] = None) -> List[Dict[str, Any]]:
        """Run experiments with flat execution - much simpler!"""
        if models is None:
            models = ["degroot"]
        
        # Get reversed setting from config
        reversed = self.config.get("reversed", False)
        
        # Create flat list of experiment configurations
        experiments = []
        for model in models:
            for network_type in network_types:
                if topics:
                    for topic in topics:
                        experiments.append({
                            'model': model,
                            'network_type': network_type, 
                            'topic': topic,
                            'n_agents': n_agents,
                            'reversed': reversed
                        })
                        # Add reversed version if symmetry testing enabled
                        if self.config.get("symmetry_testing", False):
                            experiments.append({
                                'model': model,
                                'network_type': network_type,
                                'topic': topic,
                                'n_agents': n_agents,
                                'reversed': True
                            })
                else:
                    experiments.append({
                        'model': model,
                        'network_type': network_type,
                        'topic': None,
                        'n_agents': n_agents,
                        'reversed': reversed
                    })
        
        # Run experiments with simple progress tracking
        results = []
        for i, exp in enumerate(experiments):
            try:
                result = self.run_experiment(
                    exp['network_type'], 
                    exp['topic'], 
                    exp['n_agents'], 
                    exp['model'], 
                    exp['reversed']
                )
                results.append(result)
                
                # Simple progress message
                topic_name = exp['topic'] or "math"
                if exp['reversed']:
                    topic_name += "_reversed"
                print(f"✅ [{i+1}/{len(experiments)}] {exp['model']} + {exp['network_type']} + {topic_name}")
                
            except Exception as e:
                print(f"❌ [{i+1}/{len(experiments)}] {exp['model']} + {exp['network_type']} + {exp['topic']} failed: {e}")
                results.append({"error": str(e), "config": exp})
        
        print("🎉 All experiments completed!")
        return results
    
    def save_results(self, results: List[Dict[str, Any]]) -> str:
        """Save results with simplified structure: topic JSON files and plots folder"""
        # Find the experiment directory (should be the same for all results)
        experiment_path = None
        for result in results:
            if "error" not in result:
                # Extract path from the first successful result
                streaming_file = result.get("streaming_file", "")
                if streaming_file:
                    experiment_path = os.path.dirname(streaming_file)
                    break
        
        if not experiment_path:
            # Fallback: create new directory
            timestamp = datetime.now().strftime("%m-%d-%H-%M")
            experiment_name = self.config.get("experiment", {}).get("name", "experiment")
            experiment_name = experiment_name.lower().replace(" ", "_").replace("-", "_")
            experiment_dir = f"{experiment_name}_{timestamp}"
            experiment_path = os.path.join(self.output_dir, experiment_dir)
            os.makedirs(experiment_path, exist_ok=True)
        
        # Create plots directory
        plots_dir = os.path.join(experiment_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Group results by topic
        results_by_topic = {}
        for result in results:
            if "error" in result:
                continue
                
            metadata = result.get("experiment_metadata", {})
            topics = metadata.get("topics", [])
            topic = topics[0] if topics else "pure_math_model"
            
            if topic not in results_by_topic:
                results_by_topic[topic] = []
            results_by_topic[topic].append(result)
        
        # Save each topic as a single JSON file
        for topic, topic_results in results_by_topic.items():
            # For pure math experiments, save each topology separately
            if topic == "pure_math" and len(topic_results) > 1:
                for result in topic_results:
                    metadata = result.get("experiment_metadata", {})
                    topology = metadata.get("topology", "unknown")
                    filename = f"{topic}_{topology}.json"
                    topic_file = os.path.join(experiment_path, filename)
                    serializable_data = self._make_json_serializable(result)
                    with open(topic_file, 'w') as f:
                        json.dump(serializable_data, f, indent=2)
                    logger.info(f"  📁 Saved topic: {filename}")
                    
                    # Remove streaming file after saving final result
                    streaming_file = result.get("streaming_file", "")
                    if streaming_file and os.path.exists(streaming_file):
                        os.remove(streaming_file)
                        logger.info(f"  🗑️ Removed streaming file: {os.path.basename(streaming_file)}")
            else:
                # Use the first (and should be only) result for this topic
                result = topic_results[0]
                
                # Save topic data as individual JSON file
                topic_file = os.path.join(experiment_path, f"{topic}.json")
                serializable_data = self._make_json_serializable(result)
                with open(topic_file, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
                
                logger.info(f"  📁 Saved topic: {topic}.json")
                
                # Remove streaming file after saving final result
                streaming_file = result.get("streaming_file", "")
                if streaming_file and os.path.exists(streaming_file):
                    os.remove(streaming_file)
                    logger.info(f"  🗑️ Removed streaming file: {os.path.basename(streaming_file)}")
        
        # Create experiment metadata
        metadata = {
            "experiment_timestamp": os.path.basename(experiment_path).split('_')[-2:],
            "experiment_directory": os.path.basename(experiment_path),
            "total_topics": len(results_by_topic),
            "successful_topics": len([r for r in results if "error" not in r]),
            "topics": list(results_by_topic.keys()),
            "config": self.config
        }
        
        metadata_file = os.path.join(experiment_path, "experiment_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Results saved: {experiment_path}")
        return experiment_path
    
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        else:
            return obj
    
def run_experiment(config: Dict[str, Any]) -> tuple:
    """Run an experiment with the given configuration - simplified!"""
    runner = Runner(config)
    
    # Get experiment parameters
    network_types = config.get("topologies", ["smallworld", "scalefree", "random", "echo", "karate"])
    topics = config.get("topics", [])
    n_agents = config.get("n_agents", 50)
    models = config.get("models", ["degroot"])
    
    # Run experiments with flat structure
    results = runner.run_batch(network_types, topics, n_agents, models)
    
    # Save results
    experiment_path = runner.save_results(results)
    
    return results, experiment_path

