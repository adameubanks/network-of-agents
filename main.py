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
from network_of_agents.visualization import save_simulation_data, plot_mean_std, plot_individual_opinions, plot_network_snapshots, render_network_video_from_results


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
    for name in [
        'LiteLLM', 'litellm', 'httpx', 'httpcore', 'openai', 'urllib3'
    ]:
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
    
    # Get model configuration
    model_name = llm_config.get('model_name')
        
    return LLMClient(api_key=api_key, model_name=model_name)


def run_simulation(config: Dict[str, Any], topic: str, 
                   random_seed: Optional[int] = None) -> Dict[str, Any]:
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

    # Noise removed
    
    # Use provided seed or default from config
    if random_seed is None:
        random_seed = sim_config.get('random_seed')
    

    
    # Create LLM client
    llm_client = create_llm_client(config)
    
    # Setup per-run snapshot naming and possible resume
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_agents = sim_config['n_agents']
    num_timesteps = sim_config['num_timesteps']
    model_name = ('no-llm' if not llm_enabled else llm_config.get('model_name', 'unknown'))
    topic_safe = topic.replace(' ', '_').replace('/', '_').replace('\\', '_')
    topic_dir = f"{results_dir}/{topic_safe}"
    os.makedirs(topic_dir, exist_ok=True)

    # Detect resume
    resume_cfg = config.get('resume', {})
    resume_enabled = bool(resume_cfg.get('enabled', False))
    resume_path = resume_cfg.get('path')
    resume_data: Optional[Dict[str, Any]] = None

    def _load_json(p: str) -> Optional[Dict[str, Any]]:
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    if resume_enabled and not resume_path:
        # Auto-pick latest partial for this topic
        try:
            candidates = [os.path.join(topic_dir, fn) for fn in os.listdir(topic_dir) if fn.endswith('.json')]
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for p in candidates:
                d = _load_json(p)
                if d and d.get('is_partial'):
                    resume_path = p; resume_data = d; break
        except Exception:
            pass
    elif resume_enabled and resume_path:
        resume_data = _load_json(resume_path)

    # Paths: reuse resume file if resuming, else create new
    if resume_enabled and resume_data and resume_path:
        run_data_path = resume_path
        base = os.path.splitext(os.path.basename(run_data_path))[0]
        run_plot_mean_path = os.path.join(topic_dir, f"{base}_mean_std.png")
        run_plot_individuals_path = os.path.join(topic_dir, f"{base}_individuals.png")
        graph_dir = topic_dir
        run_plot_graph_prefix = f"{base}_graph"
    else:
        run_data_path = f"{topic_dir}/{topic_safe}_{n_agents}_{num_timesteps}_{model_name}_{run_timestamp}.json"
        run_plot_mean_path = f"{topic_dir}/{topic_safe}_{n_agents}_{num_timesteps}_{model_name}_{run_timestamp}_mean_std.png"
        run_plot_individuals_path = f"{topic_dir}/{topic_safe}_{n_agents}_{model_name}_{run_timestamp}_individuals.png"
        graph_dir = topic_dir
        run_plot_graph_prefix = f"{topic_safe}_{n_agents}_{num_timesteps}_{model_name}_{run_timestamp}_graph"

    def on_timestep(snapshot: Dict[str, Any], timestep_index: int) -> None:
        # Ensure topic present for downstream consumers
        snapshot['topic'] = topic
        # Save to a single per-run file, updating it each timestep
        save_simulation_data(snapshot, run_data_path, config)

    # Create controller with single topic
    controller = Controller(
        n_agents=sim_config['n_agents'],
        epsilon=sim_config.get('epsilon', 0.001),
        theta=sim_config.get('theta', 7),
        num_timesteps=sim_config['num_timesteps'],
        llm_client=llm_client,
        topics=[topic],  # Single topic only
        random_seed=random_seed,

        llm_enabled=llm_enabled,
        on_timestep=on_timestep,
        resume_data=resume_data
    )
    
    # Run simulation
    print(f"Starting simulation for topic: {topic}")
    if llm_enabled:
        print(f"Mode: LLM (model={llm_config.get('model_name')})")
    else:
        print(f"Mode: NO-LLM")
    print(f"Initial opinions: {[agent.get_opinion() for agent in controller.agents]}")
    if resume_enabled and resume_data:
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
    print(f"Final average opinion for '{topic}': {final_avg:.3f}")
    print(f"Final opinion std dev for '{topic}': {final_std:.3f}")
    print(f"Final opinions: {[f'{op:.3f}' for op in final_opinions]}")
    
    # Add topic and run paths to results
    results['topic'] = topic
    results['run_data_path'] = run_data_path
    results['run_plot_mean_path'] = run_plot_mean_path
    results['run_plot_individuals_path'] = run_plot_individuals_path
    results['run_graph_dir'] = graph_dir
    results['run_plot_graph_prefix'] = run_plot_graph_prefix
    
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
        print(f"\nTopic {i+1}/{len(topics)}: {topic}")
        print("-" * 30)
        
        try:
            results = run_simulation(config, topic, random_seed)
            all_results[topic] = results
        except Exception as e:
            print(f"Error simulating topic '{topic}': {e}")
            all_results[topic] = {'error': str(e)}
    
    return all_results


def compare_topics(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare results across all topics.
    
    Args:
        all_results: Results for all topics
        
    Returns:
        Topic comparison summary
    """
    comparison = {
        'topics_analyzed': len(all_results),
        'failed_topics': 0,
        'topic_analyses': {}
    }
    
    for topic, results in all_results.items():
        if 'error' in results:
            comparison['failed_topics'] += 1
        
        comparison['topic_analyses'][topic] = {
            'status': 'error' if 'error' in results else 'success'
        }
    
    return comparison


def generate_default_filenames(topic: str, config: Dict[str, Any], is_partial: bool = False) -> tuple:
    """
    Generate default filenames for saving results.
    
    Args:
        topic: Topic name
        config: Configuration dictionary
        is_partial: Whether these are partial results
        
    Returns:
        Tuple of (data_filename, plot_filename)
    """
    # Create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract key parameters for filename from config
    n_agents = config['simulation']['n_agents']
    num_timesteps = config['simulation']['num_timesteps']
    llm_enabled = config.get('llm', {}).get('enabled', True)
    if not llm_enabled:
        noise_cfg = config.get('noise', {})
        model_name = 'no-llm-noise' if noise_cfg.get('enabled', False) else 'no-llm'
    else:
        model_name = config.get('llm', {}).get('model_name', 'unknown')
    
    # Create safe topic name for filename
    topic_safe = topic.replace(' ', '_').replace('/', '_').replace('\\', '_')
    
    # Add partial indicator if needed
    partial_suffix = "_PARTIAL" if is_partial else ""
    
    # Generate descriptive filename
    data_filename = f"{results_dir}/{topic_safe}_{n_agents}_{num_timesteps}_{model_name}_{timestamp}{partial_suffix}.json"
    plot_filename = f"{results_dir}/{topic_safe}_{n_agents}_{num_timesteps}_{model_name}_{timestamp}{partial_suffix}.png"
    
    return data_filename, plot_filename


def main():
    """Main function with configuration-driven interface."""
    # Load configuration
    try:
        config = load_config('config.json')
    except FileNotFoundError:
        print(f"Error: Configuration file 'config.json' not found")
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
    
    # Always run all topics iteratively
    all_results = run_simulations_iteratively(config, random_seed=None)

    print("\n" + "=" * 50)
    print("ALL TOPICS SIMULATION COMPLETED")
    print("=" * 50)

    # Compare topics
    comparison = compare_topics(all_results)

    print(f"Topics analyzed: {comparison['topics_analyzed']}")
    print(f"Topics failed: {comparison['failed_topics']}")

    print("\nDetailed Topic Analysis:")
    print("-" * 30)
    for topic, analysis in comparison['topic_analyses'].items():
        print(f"\nTopic: {topic}")
        if 'error' in analysis:
            print(f"  Error: {analysis['error']}")
        else:
            print(f"  Status: Success")

    # Save data and generate plots for each topic
    for topic, results in all_results.items():
        if 'error' not in results:
            # Prefer per-run paths from each run; fallback to generated names if absent
            data_path = results.get('run_data_path')
            plot_mean_path = results.get('run_plot_mean_path')
            plot_individuals_path = results.get('run_plot_individuals_path')
            # Prefer graph dir and prefix from results if present
            graph_dir_local = results.get('run_graph_dir')
            run_plot_graph_prefix_local = results.get('run_plot_graph_prefix')
            if not data_path or not plot_mean_path or not plot_individuals_path or not graph_dir_local or not run_plot_graph_prefix_local:
                # Fallback: reconstruct topic_dir and paths
                llm_cfg = config.get('llm', {})
                llm_on = llm_cfg.get('enabled', True)
                if not llm_on:
                    model = 'no-llm'
                else:
                    model = llm_cfg.get('model_name', 'unknown')
                sim_cfg = config['simulation']
                n = sim_cfg['n_agents']
                T = sim_cfg['num_timesteps']
                topic_safe_local = topic.replace(' ', '_').replace('/', '_').replace('\\', '_')
                results_dir_local = 'results'
                topic_dir_local = f"{results_dir_local}/{topic_safe_local}"
                os.makedirs(topic_dir_local, exist_ok=True)
                if not graph_dir_local:
                    graph_dir_local = topic_dir_local
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                data_path = data_path or f"{topic_dir_local}/{topic_safe_local}_{n}_{T}_{model}_{timestamp}.json"
                plot_mean_path = plot_mean_path or f"{topic_dir_local}/{topic_safe_local}_{n}_{T}_{model}_{timestamp}_mean_std.png"
                plot_individuals_path = plot_individuals_path or f"{topic_dir_local}/{topic_safe_local}_{n}_{T}_{model}_{timestamp}_individuals.png"
                run_plot_graph_prefix_local = run_plot_graph_prefix_local or f"{topic_safe_local}_{n}_{T}_{model}_{timestamp}_graph"
            save_simulation_data(results, data_path, config)
            plot_mean_std(results, save_path=plot_mean_path)
            plot_individual_opinions(results, save_path=plot_individuals_path)
            # Graph snapshots per timestep
            # Render video directly from results (no PNGs). 0.5s per frame (fps=2)
            render_network_video_from_results(results, graph_dir=graph_dir_local, file_prefix=run_plot_graph_prefix_local, fps=2)


if __name__ == "__main__":
    main() 