"""
Streamlined main script for opinion dynamics simulation.
"""

import json
import os
from typing import Dict, Any, Optional, List
import time
from datetime import datetime
import logging

from dotenv import load_dotenv
load_dotenv()

from network_of_agents.simulation.controller import Controller
from network_of_agents.llm_client import LLMClient
from network_of_agents.visualization import save_simulation_data, plot_mean_std, plot_individual_opinions

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def _resolve_logging_level(level_str: Optional[str]) -> int:
    if not level_str:
        return logging.WARNING
    mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    return mapping.get(level_str.upper(), logging.WARNING)

def configure_logging(config: Dict[str, Any]) -> None:
    """Configure global logging to be concise."""
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    # Silence noisy third-party loggers further
    for name in ['LiteLLM', 'litellm', 'httpx', 'httpcore', 'openai', 'urllib3']:
        logging.getLogger(name).setLevel(logging.ERROR)

def create_llm_client(config: Dict[str, Any]) -> Optional[LLMClient]:
    """
    Create LLM client from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LLM client
    """
    llm_config = config.get('llm', {})
    # If LLM is disabled, return None and skip API checks
    if llm_config.get('enabled', True) is False:
        return None
    api_key_env = llm_config.get('api_key_env', 'OPENAI_API_KEY')
    api_key = os.getenv(api_key_env)
    
    if not api_key:
        raise ValueError(f"No API key found in environment variable {api_key_env}")
    
    return LLMClient(api_key=api_key, model_name=llm_config.get('model_name'))

def _topic_label(topic: Any) -> str:
    """Human-readable label for printing/filenames. [A,B] -> "A vs B"."""
    try:
        if isinstance(topic, (list, tuple)) and len(topic) == 2:
            a = str(topic[0]).strip(); b = str(topic[1]).strip()
            return f"{a} vs {b}"
        return str(topic)
    except Exception:
        return str(topic)

def run_simulation(config: Dict[str, Any], topic: Any, random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a single simulation for one topic.
    
    Args:
        config: Configuration dictionary
        topic: Topic to simulate
        random_seed: Override random seed from config
        
    Returns:
        Simulation results for the topic
    """
    # Extract parameters
    sim_config = config['simulation']
    llm_config = config.get('llm', {})
    llm_enabled = llm_config.get('enabled', True)

    # Use provided seed or default from config
    if random_seed is None:
        random_seed = sim_config.get('random_seed')
        
    # Create LLM client
    llm_client = create_llm_client(config)
    
    # Create label for file paths and logs; keep topic object for prompts
    topic_label = _topic_label(topic)
    # Setup topic-organized file paths
    results_dir = 'results'
    topic_safe = topic_label.replace(' ', '_').replace('/', '_').replace('\\', '_')
    topic_dir = os.path.join(results_dir, topic_safe)
    os.makedirs(topic_dir, exist_ok=True)
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_agents = sim_config['n_agents']
    num_timesteps = sim_config['num_timesteps']
    model_name = ('no-llm' if not llm_enabled else llm_config.get('model_name', 'unknown'))
    base_name = f"{topic_safe}_{n_agents}_{num_timesteps}_{model_name}_{run_timestamp}"

    # Check for existing partial results to resume
    def _load_json(p: str) -> Optional[Dict[str, Any]]:
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    # Find partial result for THIS specific topic to resume from
    resume_data = None
    resume_path = None
    try:
        # Look for partial results in the topic-specific directory
        topic_candidates = [os.path.join(topic_dir, fn) for fn in os.listdir(topic_dir) if fn.endswith('.json')]
        topic_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for p in topic_candidates:
            data = _load_json(p)
            if data and data.get('is_partial') and data.get('topic') == topic_label:
                resume_path = p
                resume_data = data
                break
    except Exception:
        pass

    # Set up simple file paths
    if resume_data:
        base = os.path.splitext(os.path.basename(resume_path))[0]
        run_data_path = resume_path
    else:
        base = base_name
        run_data_path = f"{topic_dir}/{base}.json"

    def on_timestep(snapshot: Dict[str, Any], timestep_index: int) -> None:
        # Ensure topic present for downstream consumers
        snapshot['topic'] = topic_label
        # Save to a single per-run file, updating it each timestep
        save_simulation_data(snapshot, run_data_path, config)

    # Create controller with single topic (DeGroot-only)
    controller = Controller(
        n_agents=sim_config['n_agents'],
        num_timesteps=sim_config['num_timesteps'],
        llm_client=llm_client,
        topics=[topic],  # Single topic only, can be [A,B]
        random_seed=random_seed,
        llm_enabled=llm_enabled,
        on_timestep=on_timestep,
        resume_data=resume_data
    )
    
    # Run simulation
    print(f"Starting simulation for topic: {topic_label}")
    if llm_enabled:
        print(f"Mode: LLM (model={llm_config.get('model_name')})")
    else:
        print(f"Mode: NO-LLM")
    print(f"Opinion Dynamics Model: DEGROOT")
    print(f"Initial opinions: {[agent.get_opinion() for agent in controller.agents]}")
    if resume_data:
        done = len(controller.mean_opinions)
        print(f"Resuming from timestep {done}/{num_timesteps}")
    start_time = time.time()
    results = controller.run_simulation(progress_bar=True)
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Check if simulation was interrupted
    is_partial = results.get('is_partial', False)
    if is_partial:
        print(f"⚠️  SIMULATION INTERRUPTED")
        print(f"   Completed: {results.get('completed_timesteps', 0)}/{results.get('total_timesteps', 0)} timesteps")
        print(f"   Error: {results.get('error', 'Unknown error')}")
    else:
        print(f"✅ SIMULATION COMPLETED SUCCESSFULLY")
    
    # Print final summary for this topic
    final_opinions = results['final_opinions']
    final_avg = sum(final_opinions) / len(final_opinions)
    final_std = (sum((x - final_avg) ** 2 for x in final_opinions) / len(final_opinions)) ** 0.5
    print(f"Final average opinion for '{topic_label}': {final_avg:.3f}")
    print(f"Final opinion std dev for '{topic}': {final_std:.3f}")
    print(f"Final opinions: {[f'{op:.3f}' for op in final_opinions]}")
    
    # Add topic to results
    results['topic'] = topic_label
    
    return results

def run_simulations_iteratively(config: Dict[str, Any], random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run simulations for all topics iteratively.
    
    Args:
        config: Configuration dictionary
        random_seed: Override random seed from config
        
    Returns:
        Results for all topics
    """
    topics = config.get('topics', ['default topic'])
    all_results = {}
    
    for i, topic in enumerate(topics):
        topic_str = _topic_label(topic)
        print(f"\nTopic {i+1}/{len(topics)}: {topic_str}")
        print("-" * 30)
        
        try:
            results = run_simulation(config, topic, random_seed)
            all_results[topic_str] = results
        except Exception as e:
            print(f"Error simulating topic '{topic_str}': {e}")
            all_results[topic_str] = {'error': str(e)}
    
    return all_results

def main():
    """Main function with configuration-driven interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Opinion Dynamics Simulation")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return
    
    # Configure logging
    configure_logging(config)

    # Run simulation
    print("=" * 50)
    print("OPINION DYNAMICS SIMULATION")
    print("=" * 50)
    
    # Run all topics
    all_results = run_simulations_iteratively(config, random_seed=None)

    print("\n" + "=" * 50)
    print("SIMULATION COMPLETED")
    print("=" * 50)

    # Simple summary
    successful = sum(1 for r in all_results.values() if 'error' not in r)
    failed = len(all_results) - successful
    print(f"Topics completed: {successful}")
    print(f"Topics failed: {failed}")

    # Generate simple plots for successful runs
    for topic, results in all_results.items():
        if 'error' not in results:
            topic_safe = topic.replace(' ', '_').replace('/', '_').replace('\\', '_')
            topic_dir = os.path.join('results', topic_safe)
            os.makedirs(topic_dir, exist_ok=True)
            
            base_name = f"{topic_safe}_{config['simulation']['n_agents']}_{config['simulation']['num_timesteps']}"
            plot_mean_std(results, save_path=f"{topic_dir}/{base_name}_mean_std.png")
            plot_individual_opinions(results, save_path=f"{topic_dir}/{base_name}_individuals.png")
            
if __name__ == "__main__":
    main() 