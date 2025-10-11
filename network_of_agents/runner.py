"""
Simplified experiment runner for opinion dynamics research.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Load environment variables from .env file
load_dotenv()

from .simulation.controller import Controller
from .llm_client import LLMClient
from .network.generator import create_network_model, get_network_params
from .core.mathematics import (
    update_opinions_pure_degroot, update_opinions_friedkin_johnsen,
    initialize_opinions_normal, get_network_info, check_convergence
)

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

def _load_convergence_steps(network_type: str, model: str, config_convergence_steps: Dict[str, int], max_steps: int = 100) -> int:
    """Load convergence steps from config override or convergence data file."""
    if network_type in config_convergence_steps:
        return config_convergence_steps[network_type]
    
    try:
        with open("utils/convergence_data.json", 'r') as f:
            convergence_data = json.load(f)
        topology_data = convergence_data.get("topology_convergence", {}).get(network_type, {})
        if model == "degroot":
            return topology_data.get("degroot_convergence_steps", max_steps)
        elif model == "friedkin_johnsen":
            return topology_data.get("friedkin_johnsen_convergence_steps", max_steps)
        else:
            return topology_data.get("degroot_convergence_steps", max_steps)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load convergence data: {e}. Using default max_steps={max_steps}")
        return max_steps

def get_topic_framing(topic_key: str) -> Tuple[str, str]:
    """Get topic framing for LLM prompts."""
    return TOPIC_MAPPING.get(topic_key, (f"Option A for {topic_key}", f"Option B for {topic_key}"))

class Runner:
    """Experiment runner for opinion dynamics research."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.output_dir = config.get("output_dir", "results")
        
        self.llm_client = None
        llm_config = config.get("llm", {})
        if llm_config.get("enabled", False):
            model_name = llm_config.get("model", "gpt-5-mini")
            max_tokens = llm_config.get("max_tokens", 150) if "gpt-5" not in model_name.lower() else 150
            self.llm_client = LLMClient(
                model_name=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                max_workers=llm_config.get("max_workers", 100),
                timeout=llm_config.get("timeout", 15),
                max_tokens=max_tokens
            )
    
    def run_experiment(self, network_type: str, topic: str = None, 
                      n_agents: int = 50, max_steps: int = 100, model: str = "degroot",
                      progress_callback: Optional[Callable[[int, int], None]] = None,
                      pure_math: bool = False, checkpoint_interval: int = 10) -> Dict[str, Any]:
        """Run a single experiment"""
        try:
            logger.info(f"Running experiment: {network_type} with {n_agents} agents")
            
            # Get network parameters
            logger.info(f"Getting network params for {network_type} with {n_agents} agents")
            network_params = get_network_params(network_type, n_agents)
            logger.info(f"Network params: {network_params}")
        except Exception as e:
            logger.error(f"Error in run_experiment setup: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Create network
        network = create_network_model(network_type, network_params, 
                                     random_seed=self.config.get("random_seed", 42))
        adjacency = network.get_adjacency_matrix()
        
        # Use actual number of agents from the network (important for karate)
        actual_n_agents = adjacency.shape[0]
        
        # Initialize opinions
        opinions = initialize_opinions_normal(actual_n_agents, random_seed=self.config.get("random_seed", 42))
        opinion_history = [opinions.copy()]
        
        # If pure math mode, use Controller for detailed agent state tracking
        if pure_math:
            controller = Controller(
                llm_client=None,
                n_agents=actual_n_agents,
                epsilon=1e-6,
                num_timesteps=max_steps,
                topics=None,
                random_seed=self.config.get("random_seed", 42),
                llm_enabled=False,
                model=model,
                progress_callback=progress_callback,
                checkpoint_interval=checkpoint_interval
            )
            controller.network.adjacency_matrix = adjacency
            # Set initial opinions
            for i, agent in enumerate(controller.agents):
                agent.update_opinion(opinions[i])
            results = controller.run_simulation(progress_bar=True)
            
            # Update the topology in the results metadata
            if "experiment_metadata" in results:
                results["experiment_metadata"]["topology"] = network_type
            
            return results
        
        # Load convergence steps
        config_convergence_steps = self.config.get("convergence_steps", {})
        max_steps = _load_convergence_steps(network_type, model, config_convergence_steps, max_steps)
        logger.info(f"Using convergence steps: {network_type} = {max_steps} steps")
        
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
                topics=[topic],  # Pass the string topic key, not the tuple framing
                random_seed=self.config.get("random_seed", 42),
                llm_enabled=True,
                progress_callback=progress_callback,
                checkpoint_interval=checkpoint_interval
            )
            controller.network.adjacency_matrix = adjacency
            results = controller.run_simulation(progress_bar=True)
            
            # Update the topology in the results metadata
            if "experiment_metadata" in results:
                results["experiment_metadata"]["topology"] = network_type
            
            return results
        else:
            # Mathematical simulation using Controller for detailed data
            controller = Controller(
                llm_client=None,
                n_agents=actual_n_agents,
                epsilon=1e-6,
                num_timesteps=max_steps,
                topics=[topic] if topic else None,
                random_seed=self.config.get("random_seed", 42),
                llm_enabled=False,
                model=model,
                progress_callback=progress_callback,
                checkpoint_interval=checkpoint_interval
            )
            controller.network.adjacency_matrix = adjacency
            results = controller.run_simulation(progress_bar=True)
            
            logger.info(f"Mathematical simulation results keys: {list(results.keys()) if results else 'None'}")
            
            # Update the topology in the results metadata
            if "experiment_metadata" in results:
                results["experiment_metadata"]["topology"] = network_type
            
            return results
    
    def run_batch(self, network_types: List[str], topics: List[str] = None, 
                  n_agents: int = 50, max_steps: int = 100, models: List[str] = None,
                  checkpoint_interval: int = 10) -> Dict[str, Any]:
        """Run a batch of experiments with hierarchical progress tracking"""
        if models is None:
            models = ["degroot"]
        
        # Check if LLM is enabled
        llm_enabled = self.config.get("llm", {}).get("enabled", True)
        config_convergence_steps = self.config.get("convergence_steps", {})
        
        # Calculate total experiments for progress tracking
        total_experiments = len(models) * len(network_types) * (len(topics) if topics else 1)
        
        tqdm.write(f"🚀 Starting batch: {len(models)} models × {len(network_types)} topologies × {len(topics) if topics else 1} topics = {total_experiments} experiments")
        
        # New nested structure: model -> topology -> topic
        results = {}
        
        # No experiment-level progress bar - just run experiments
        for model in models:
                results[model] = {}
                for network_type in network_types:
                    results[model][network_type] = {}
                    
                    if topics:
                        for topic in topics:
                            try:
                                topology_max_steps = _load_convergence_steps(network_type, model, config_convergence_steps, max_steps)
                                
                                # No progress callback - let individual experiments handle their own progress
                                result = self.run_experiment(
                                    network_type, topic, n_agents, topology_max_steps, 
                                    model, None, pure_math=not llm_enabled, checkpoint_interval=checkpoint_interval
                                )
                                
                                # Store results
                                if "results" in result and topic in result["results"]:
                                    results[model][network_type][topic] = result["results"][topic]
                                else:
                                    results[model][network_type][topic] = result
                                
                                tqdm.write(f"✅ {model} + {network_type} + {topic} completed")
                                
                            except Exception as e:
                                results[model][network_type][topic] = {"error": str(e)}
                                tqdm.write(f"❌ {model} + {network_type} + {topic} failed: {e}")
                    else:
                        # Mathematical experiment
                        try:
                            topology_max_steps = _load_convergence_steps(network_type, model, config_convergence_steps, max_steps)
                            
                            # No progress callback - let individual experiments handle their own progress
                            result = self.run_experiment(
                                network_type, None, n_agents, topology_max_steps, 
                                model, None, pure_math=True, checkpoint_interval=checkpoint_interval
                            )
                            
                            results[model][network_type]["pure_math_model"] = result
                            tqdm.write(f"✅ {model} + {network_type} (mathematical) completed")
                            
                        except Exception as e:
                            results[model][network_type]["pure_math_model"] = {"error": str(e)}
                            tqdm.write(f"❌ {model} + {network_type} (mathematical) failed: {e}")
        
        tqdm.write("🎉 All experiments completed!")
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results with organized directory structure and simple naming"""
        timestamp = datetime.now().strftime("%m-%d-%H-%M")
        
        if filename is None:
            # Use experiment name from config, fallback to simple name
            experiment_name = self.config.get("experiment", {}).get("name", "experiment")
            # Clean the name for filesystem use
            experiment_name = experiment_name.lower().replace(" ", "_").replace("-", "_")
            experiment_dir = f"{experiment_name}_{timestamp}"
        else:
            experiment_dir = f"{filename}_{timestamp}"
        
        # Create organized directory structure
        experiment_path = os.path.join(self.output_dir, experiment_dir)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Create topic-specific folders and save individual topic results
        for model, model_data in results.items():
            if isinstance(model_data, dict):
                for topology, topology_data in model_data.items():
                    if isinstance(topology_data, dict):
                        for topic, topic_data in topology_data.items():
                            # Check if this is the new nested structure with 'results' key
                            if isinstance(topic_data, dict) and 'results' in topic_data:
                                # This is the new nested structure - extract the actual topic data
                                for actual_topic, actual_topic_data in topic_data['results'].items():
                                    if isinstance(actual_topic_data, dict) and 'timesteps' in actual_topic_data:
                                        # Use the actual topic name
                                        topic_name = actual_topic
                                        
                                        # Create topic-specific folder (this is where visualizations will go too)
                                        topic_dir = os.path.join(experiment_path, topic_name)
                                        os.makedirs(topic_dir, exist_ok=True)
                                        
                                        # Save detailed topic data in the same folder as visualizations
                                        topic_filepath = os.path.join(topic_dir, f"{model}_{topology}_{topic_name}_detailed.json")
                                        serializable_topic_data = self._make_json_serializable(actual_topic_data)
                                        with open(topic_filepath, 'w') as f:
                                            json.dump(serializable_topic_data, f, indent=2)
                                        
                                        logger.info(f"  📁 Saved detailed data: {topic_filepath}")
                            elif isinstance(topic_data, dict) and 'timesteps' in topic_data:
                                # This is the old flat structure - handle it directly
                                # For pure math experiments, use "pure_math_model" as topic
                                is_pure_math = not self.config.get("llm", {}).get("enabled", True)
                                if is_pure_math:
                                    topic_name = "pure_math_model"
                                else:
                                    topic_name = topic
                                
                                # Create topic-specific folder (this is where visualizations will go too)
                                topic_dir = os.path.join(experiment_path, topic_name)
                                os.makedirs(topic_dir, exist_ok=True)
                                
                                # Save detailed topic data in the same folder as visualizations
                                topic_filepath = os.path.join(topic_dir, f"{model}_{topology}_{topic_name}_detailed.json")
                                serializable_topic_data = self._make_json_serializable(topic_data)
                                with open(topic_filepath, 'w') as f:
                                    json.dump(serializable_topic_data, f, indent=2)
                                
                                logger.info(f"  📁 Saved detailed data: {topic_filepath}")
        
        # Create visualizations if enabled
        if self.config.get("analysis", {}).get("create_visualizations", True):
            logger.info("📊 Creating visualizations...")
            self._create_visualizations(results, experiment_path)
        
        # Save experiment metadata
        metadata = {
            "experiment_timestamp": timestamp,
            "experiment_directory": experiment_dir,
            "experiment_type": "pure_math" if not self.config.get("llm", {}).get("enabled", True) else "llm_based",
            "models": list(results.keys()),
            "topologies": list(set(topology for model_data in results.values() 
                                 if isinstance(model_data, dict)
                                 for topology in model_data.keys())),
            "topics": list(set(topic for model_data in results.values()
                             if isinstance(model_data, dict)
                             for topology_data in model_data.values()
                             if isinstance(topology_data, dict)
                             for topic in topology_data.keys())),
            "total_topics": len(set(topic for model_data in results.values()
                                  if isinstance(model_data, dict)
                                  for topology_data in model_data.values()
                                  if isinstance(topology_data, dict)
                                  for topic in topology_data.keys())),
            "naming_convention": "model_topic_topology_MM-DD-HH-MM"
        }
        
        metadata_filepath = os.path.join(experiment_path, "experiment_metadata.json")
        serializable_metadata = self._make_json_serializable(metadata)
        with open(metadata_filepath, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        
        logger.info(f"Results saved to organized structure: {experiment_path}")
        logger.info(f"  📁 Topic folders: {experiment_path}")
        logger.info(f"  📄 Metadata: {metadata_filepath}")
        
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
    
    def _create_visualizations(self, results: Dict[str, Any], experiment_path: str):
        """Create visualizations for the experiment results"""
        try:
            # Import visualization functions from run.py
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from run import create_visualizations
            
            # Flatten results structure for visualization (same as in run.py)
            flattened_results = {}
            for model, model_data in results.items():
                if isinstance(model_data, dict):
                    for topology, topology_data in model_data.items():
                        if isinstance(topology_data, dict):
                            for topic, topic_data in topology_data.items():
                                # Check if this is the new nested structure with 'results' key
                                if isinstance(topic_data, dict) and 'results' in topic_data:
                                    # This is the new nested structure - extract the actual topic data
                                    for actual_topic, actual_topic_data in topic_data['results'].items():
                                        if isinstance(actual_topic_data, dict) and 'summary_metrics' in actual_topic_data:
                                            # Extract summary metrics for visualization
                                            summary = actual_topic_data['summary_metrics']
                                            flattened_results[f"{topology}_{model}_{actual_topic}"] = {
                                                'opinion_history': summary.get('opinion_history', []),
                                                'mean_opinions': summary.get('mean_opinions', []),
                                                'std_opinions': summary.get('std_opinions', []),
                                                'final_opinions': summary.get('final_opinions', []),
                                                'network_info': summary.get('network_info', {}),
                                                'topic': actual_topic
                                            }
                                elif isinstance(topic_data, dict) and 'summary_metrics' in topic_data:
                                    # This is the old flat structure - handle it directly
                                    summary = topic_data['summary_metrics']
                                    flattened_results[f"{topology}_{model}_{topic}"] = {
                                        'opinion_history': summary.get('opinion_history', []),
                                        'mean_opinions': summary.get('mean_opinions', []),
                                        'std_opinions': summary.get('std_opinions', []),
                                        'final_opinions': summary.get('final_opinions', []),
                                        'network_info': summary.get('network_info', {}),
                                        'topic': topic
                                    }
            
            # Create visualizations in the organized structure
            logger.info(f"📊 Creating visualizations for {len(flattened_results)} results...")
            logger.info(f"📊 Flattened results keys: {list(flattened_results.keys())}")
            
            # For pure math experiments, we need to modify the config to include topics
            vis_config = self.config.copy()
            is_pure_math = not self.config.get("llm", {}).get("enabled", True)
            if is_pure_math and not vis_config.get("topics"):
                # Extract topics from the flattened results
                topics = set()
                for key, data in flattened_results.items():
                    if isinstance(data, dict) and "topic" in data:
                        topics.add(data["topic"])
                vis_config["topics"] = list(topics)
                logger.info(f"📊 Using extracted topics for visualization: {vis_config['topics']}")
            
            create_visualizations(flattened_results, experiment_path, vis_config)
            logger.info("✅ Visualizations created successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run an experiment with the given configuration"""
    runner = Runner(config)
    
    # Get experiment parameters
    network_types = config.get("topologies", ["smallworld", "scalefree", "random", "echo", "karate"])
    topics = config.get("topics", [])
    n_agents = config.get("n_agents", 50)
    checkpoint_interval = config.get("checkpoint_interval", 10)
    
    # Convergence steps are now loaded from convergence_data.json per topology
    
    # Get models to run
    models = config.get("models", ["degroot"])
    
    # Use a default max_steps for the run_batch call (will be overridden per topology)
    default_max_steps = 100
    
    # Run experiments
    if topics:
        results = runner.run_batch(network_types, topics, n_agents, default_max_steps, models, checkpoint_interval)
    else:
        results = runner.run_batch(network_types, None, n_agents, default_max_steps, models, checkpoint_interval)
    
    # Save results and return both results and path
    experiment_path = runner.save_results(results)
    
    return results, experiment_path

