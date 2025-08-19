"""
Streamlined main script for opinion dynamics simulation.
"""

import json
import os
from typing import Dict, Any, Optional, List
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from network_of_agents.simulation.controller import Controller
from network_of_agents.llm_client import LLMClient
from network_of_agents.visualization import plot_opinion_evolution, save_simulation_data, print_simulation_summary


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
    
    # Get batch configuration
    batch_config = llm_config.get('batch', {})
    
    # Get temperature configuration
    generation_temperature = llm_config.get('generation_temperature', 0.9)
    rating_temperature = llm_config.get('rating_temperature', 0.1)
    
    return LLMClient(api_key=api_key, model_name=model_name, batch_config=batch_config, 
                    generation_temperature=generation_temperature, 
                    rating_temperature=rating_temperature)


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

    noise_config = config.get('noise', {})
    noise_enabled = noise_config.get('enabled', False)
    noise_mean = noise_config.get('mean', 0.0)
    noise_std = noise_config.get('std', 0.0)
    noise_clip = noise_config.get('clip', True)
    
    # Use provided seed or default from config
    if random_seed is None:
        random_seed = sim_config.get('random_seed')
    
    # Get temperature settings
    generation_temperature = llm_config.get('generation_temperature', 0.9)
    rating_temperature = llm_config.get('rating_temperature', 0.1)
    
    # Create LLM client
    llm_client = create_llm_client(config)
    
    # Create controller with single topic
    controller = Controller(
        n_agents=sim_config['n_agents'],
        epsilon=sim_config.get('epsilon', 0.001),
        theta=sim_config.get('theta', 7),
        num_timesteps=sim_config['num_timesteps'],
        initial_connection_probability=sim_config['initial_connection_probability'],
        llm_client=llm_client,
        topics=[topic],  # Single topic only
        random_seed=random_seed,
        generation_temperature=generation_temperature,
        rating_temperature=rating_temperature,
        llm_enabled=llm_enabled,
        noise_enabled=noise_enabled,
        noise_mean=noise_mean,
        noise_std=noise_std,
        noise_clip=noise_clip
    )
    
    # Run simulation
    print(f"Starting simulation for topic: {topic}")
    if llm_enabled:
        print(f"Mode: LLM (model={llm_config.get('model_name')}, gen_T={generation_temperature}, rate_T={rating_temperature})")
    else:
        print(f"Mode: NO-LLM (noise={'on' if noise_enabled else 'off'}, mean={noise_mean}, std={noise_std}, clip={noise_clip})")
    print(f"Initial opinions: {[agent.get_opinion() for agent in controller.agents]}")
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
    
    # Add topic to results
    results['topic'] = topic
    
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
    results_dir = config.get('output', {}).get('results_directory', 'results')
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
    
    # Run simulation
    print("=" * 50)
    print("OPINION DYNAMICS SIMULATION")
    print("=" * 50)
    
    # Check if single topic is specified in config
    single_topic = config.get('single_topic')
    
    if single_topic is not None:
        # Run single topic simulation
        results = run_simulation(config, single_topic, random_seed=None)
        
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETED")
        print("=" * 50)
        print(f"Topic: {results['topic']}")
        print(f"Agents: {results['simulation_params']['n_agents']}")
        print(f"Timesteps: {results['simulation_params']['num_timesteps']}")
        print(f"Random seed: {results['random_seed']}")
        
        # Check if results are partial
        is_partial = results.get('is_partial', False)
        if is_partial:
            print(f"⚠️  PARTIAL RESULTS - Simulation was interrupted")
            print(f"   Completed: {results.get('completed_timesteps', 0)}/{results.get('total_timesteps', 0)} timesteps")
            print(f"   Error: {results.get('error', 'Unknown error')}")
        else:
            print(f"✅ COMPLETE RESULTS - Simulation finished successfully")
        
        # Print detailed summary
        print_simulation_summary(results)
        
        # Determine save paths
        data_path, plot_path = generate_default_filenames(results['topic'], config, is_partial=is_partial)
        
        # Save data if requested
        save_simulation_data(results, data_path, config)
        
        # Generate and save plot if requested
        plot_opinion_evolution(results, save_path=plot_path)
        
    else:
        # Run all topics iteratively
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
                is_partial = results.get('is_partial', False)
                data_path, plot_path = generate_default_filenames(topic, config, is_partial=is_partial)
                save_simulation_data(results, data_path, config)
                
                plot_opinion_evolution(results, save_path=plot_path)


if __name__ == "__main__":
    main() 