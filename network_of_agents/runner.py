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
            model_name = llm_config.get("model", "gpt-5-mini")
            # For GPT-5 models, let the LLM client handle max_tokens internally
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
                      pure_math: bool = False) -> Dict[str, Any]:
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
        
        # If pure math mode, run mathematical convergence only
        if pure_math:
            return _run_pure_math_convergence(network_type, model, adjacency, opinions, max_steps)
        
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
                progress_callback=progress_callback
            )
            controller.network.adjacency_matrix = adjacency
            results = controller.run_simulation(progress_bar=False)
            
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
                progress_callback=progress_callback
            )
            controller.network.adjacency_matrix = adjacency
            results = controller.run_simulation(progress_bar=False)
            
            logger.info(f"Mathematical simulation results keys: {list(results.keys()) if results else 'None'}")
            
            # Update the topology in the results metadata
            if "experiment_metadata" in results:
                results["experiment_metadata"]["topology"] = network_type
            
            return results
    
    def run_batch(self, network_types: List[str], topics: List[str] = None, 
                  n_agents: int = 50, max_steps: int = 100, models: List[str] = None) -> Dict[str, Any]:
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
        
        # Create main experiment progress bar
        with tqdm(total=total_experiments, desc="🧪 Overall Experiments", unit="exp", position=0, leave=True) as main_pbar:
            for model_idx, model in enumerate(models):
                results[model] = {}
                
                # Create model-level progress bar
                model_total = len(network_types) * (len(topics) if topics else 1)
                with tqdm(total=model_total, desc=f"🤖 {model.upper()}", unit="exp", position=1, leave=False) as model_pbar:
                    for network_type in network_types:
                        results[model][network_type] = {}
                        
                        if topics:
                            # Create topic-level progress bar for this model+topology combination
                            with tqdm(total=len(topics), desc=f"📊 {network_type}", unit="topic", position=2, leave=False) as topic_pbar:
                                for topic in topics:
                                    try:
                                        # Get convergence steps for this specific combination
                                        topology_max_steps = _load_convergence_steps(network_type, model, config_convergence_steps, max_steps)
                                        
                                        # Create timestep progress callback for this specific experiment
                                        def create_timestep_callback(topic_name, network_name, model_name):
                                            def timestep_callback(completed, total):
                                                topic_pbar.set_postfix(
                                                    step=f"{completed}/{total}",
                                                    status="🔄 Running..."
                                                )
                                            return timestep_callback
                                        
                                        timestep_callback = create_timestep_callback(topic, network_type, model)
                                        
                                        # Run the experiment
                                        result = self.run_experiment(
                                            network_type, topic, n_agents, topology_max_steps, 
                                            model, timestep_callback, pure_math=not llm_enabled
                                        )
                                        
                                        # Store results
                                        if "results" in result and topic in result["results"]:
                                            results[model][network_type][topic] = result["results"][topic]
                                        else:
                                            results[model][network_type][topic] = result
                                        
                                        # Update progress bars
                                        topic_pbar.set_postfix(status="✅ Complete")
                                        topic_pbar.update(1)
                                        model_pbar.update(1)
                                        main_pbar.update(1)
                                        
                                        tqdm.write(f"  ✅ {model} + {network_type} + {topic} completed")
                                        
                                    except Exception as e:
                                        tqdm.write(f"  ❌ {model} + {network_type} + {topic} failed: {e}")
                                        results[model][network_type][topic] = {"error": str(e)}
                                        
                                        # Update progress bars even on failure
                                        topic_pbar.set_postfix(status="❌ Failed")
                                        topic_pbar.update(1)
                                        model_pbar.update(1)
                                        main_pbar.update(1)
                        else:
                            # Mathematical experiment (no topics)
                            try:
                                topology_max_steps = _load_convergence_steps(network_type, model, config_convergence_steps, max_steps)
                                
                                # Create timestep progress callback for mathematical experiment
                                def create_math_callback(network_name, model_name):
                                    def math_callback(completed, total):
                                        model_pbar.set_postfix(
                                            step=f"{completed}/{total}",
                                            status="🧮 Math processing..."
                                        )
                                    return math_callback
                                
                                math_callback = create_math_callback(network_type, model)
                                
                                result = self.run_experiment(
                                    network_type, None, n_agents, topology_max_steps, 
                                    model, math_callback, pure_math=True
                                )
                                
                                results[model][network_type]["pure_math_model"] = result
                                model_pbar.set_postfix(status="✅ Complete")
                                model_pbar.update(1)
                                main_pbar.update(1)
                                
                                tqdm.write(f"  ✅ {model} + {network_type} (mathematical) completed")
                                
                            except Exception as e:
                                tqdm.write(f"  ❌ {model} + {network_type} (mathematical) failed: {e}")
                                results[model][network_type]["pure_math_model"] = {"error": str(e)}
                                model_pbar.set_postfix(status="❌ Failed")
                                model_pbar.update(1)
                                main_pbar.update(1)
        
        tqdm.write("🎉 All experiments completed!")
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results with simplified directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Simplified directory structure
        if filename is None:
            experiment_dir = f"experiment_{timestamp}"
        else:
            experiment_dir = f"{filename}_{timestamp}"
        
        # Create main experiment directory
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
                                  for topic in topology_data.keys()))
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
        elif str(type(obj)) in ["<class 'numpy.int64'>", "<class 'numpy.int32'>", "<class 'numpy.int16'>", "<class 'numpy.int8'>"]:
            return int(obj)
        elif str(type(obj)) in ["<class 'numpy.float64'>", "<class 'numpy.float32'>", "<class 'numpy.float16'>"]:
            return float(obj)
        elif str(type(obj)) in ["<class 'numpy.bool_'>", "<class 'numpy.bool8'>"]:
            return bool(obj)
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
            is_pure_math = not self.llm_client or not self.config.get("llm", {}).get("enabled", True)
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
    
    # Convergence steps are now loaded from convergence_data.json per topology
    
    # Get models to run
    models = config.get("models", ["degroot"])
    
    # Use a default max_steps for the run_batch call (will be overridden per topology)
    default_max_steps = 100
    
    # Run experiments
    if topics:
        results = runner.run_batch(network_types, topics, n_agents, default_max_steps, models)
    else:
        results = runner.run_batch(network_types, None, n_agents, default_max_steps, models)
    
    # Save results and return both results and path
    experiment_path = runner.save_results(results)
    
    return results, experiment_path

def _run_pure_math_convergence(network_type: str, model: str, adjacency: np.ndarray, 
                             initial_opinions: np.ndarray, max_steps: int) -> Dict[str, Any]:
    """Run pure mathematical convergence without LLM."""
    opinions = initial_opinions.copy()
    opinion_history = [opinions.copy()]
    
    for step in range(max_steps):
        if model == "degroot":
            opinions = update_opinions_pure_degroot(opinions, adjacency)
        elif model == "friedkin_johnsen":
            lambda_values = np.full(len(opinions), 0.1)
            opinions = update_opinions_friedkin_johnsen(opinions, adjacency, lambda_values, initial_opinions)
        
        opinion_history.append(opinions.copy())
        
        if check_convergence(opinions, opinion_history[-2], threshold=1e-10):
            break
    
    return {
        "network_type": network_type,
        "model": model,
        "converged": check_convergence(opinions, opinion_history[-2], threshold=1e-10),
        "steps": len(opinion_history) - 1,
        "final_opinions": opinions.tolist(),
        "final_mean": float(np.mean(opinions)),
        "final_std": float(np.std(opinions)),
        "final_variance": float(np.var(opinions)),
        "opinion_history": [op.tolist() for op in opinion_history]
    }
