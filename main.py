"""
Simplified main script for opinion dynamics simulation.
"""

import json
import os
import argparse
from typing import Dict, Any, Optional
import time

from dotenv import load_dotenv
load_dotenv()

from network_of_agents.simulation.controller import SimulationController
from network_of_agents.llm.litellm_client import LiteLLMClient


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


def create_llm_client(config: Dict[str, Any]) -> Optional[LiteLLMClient]:
    """
    Create LLM client from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LiteLLM client or None if not configured
    """
    llm_config = config.get('llm', {})
    model_name = llm_config.get('model', 'gpt-4o-mini')
    api_key_env = llm_config.get('api_key_env', 'OPENAI_API_KEY')
    api_key = os.getenv(api_key_env)
    
    if not api_key:
        print(f"Warning: No API key found in environment variable {api_key_env}")
        print("Simulation will run without LLM encoding/decoding")
        return None
    
    try:
        return LiteLLMClient(model_name=model_name, api_key=api_key)
    except Exception as e:
        print(f"Warning: Failed to create LLM client: {e}")
        print("Simulation will run without LLM encoding/decoding")
        return None


def run_simulation(config: Dict[str, Any], topic: Optional[str] = None, 
                   random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a single simulation.
    
    Args:
        config: Configuration dictionary
        topic: Override topic from config
        random_seed: Override random seed from config
        
    Returns:
        Simulation results
    """
    # Extract parameters
    sim_config = config['simulation']
    output_dir = config.get('output_directory', 'simulation_results')
    
    # Use provided topic or default from config
    if topic is None:
        topic = config.get('topic', 'Climate Change')
    
    # Use provided seed or default from config
    if random_seed is None:
        random_seed = sim_config.get('random_seed')
    
    # Create LLM client
    llm_client = create_llm_client(config)
    
    # Create controller
    controller = SimulationController(
        n_agents=sim_config['n_agents'],
        epsilon=sim_config['epsilon'],
        theta=sim_config['theta'],
        num_timesteps=sim_config['num_timesteps'],
        initial_connection_probability=sim_config['initial_connection_probability'],
        llm_client=llm_client,
        topics=[topic],
        random_seed=random_seed,
        initial_opinion_diversity=sim_config.get('initial_opinion_diversity')
    )
    
    # Run simulation
    print(f"Starting simulation for topic: {topic}")
    start_time = time.time()
    results = controller.run_simulation(progress_bar=True)
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Get basic results
    opinion_history = controller.data_storage.get_opinion_history()
    final_opinions = controller._get_opinion_matrix()
    final_adjacency = controller.network.get_adjacency_matrix()
    
    # Print final summary
    final_avg = final_opinions.mean()
    final_std = final_opinions.std()
    print(f"Final average opinion: {final_avg:.3f}")
    print(f"Final opinion std dev: {final_std:.3f}")
    
    return {
        'topic': topic,
        'opinion_history': [op.tolist() if op is not None else [] for op in opinion_history],
        'final_opinions': final_opinions.tolist(),
        'final_adjacency': final_adjacency.tolist(),
        'simulation_params': sim_config,
        'random_seed': random_seed
    }


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Run opinion dynamics simulation')
    parser.add_argument('--config', type=str, default='config.json', 
                       help='Configuration file path')
    parser.add_argument('--topic', type=str, help='Topic to simulate (overrides config)')
    parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
    parser.add_argument('--agents', type=int, help='Number of agents (overrides config)')
    parser.add_argument('--timesteps', type=int, help='Number of timesteps (overrides config)')
    
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
    
    # Override config with command-line arguments
    if args.agents is not None:
        config['simulation']['n_agents'] = args.agents
    if args.timesteps is not None:
        config['simulation']['num_timesteps'] = args.timesteps
    
    # Run simulation
    print("=" * 50)
    print("OPINION DYNAMICS SIMULATION")
    print("=" * 50)
    
    results = run_simulation(config, topic=args.topic, random_seed=args.seed)
    
    print("\n" + "=" * 50)
    print("SIMULATION COMPLETED")
    print("=" * 50)
    print(f"Topic: {results['topic']}")
    print(f"Agents: {results['simulation_params']['n_agents']}")
    print(f"Timesteps: {results['simulation_params']['num_timesteps']}")
    print(f"Random seed: {results['random_seed']}")
    print(f"Final average opinion: {sum(results['final_opinions']) / len(results['final_opinions']):.3f}")


if __name__ == "__main__":
    main() 