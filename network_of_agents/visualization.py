"""
Visualization module for opinion dynamics simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import json
import os
from datetime import datetime





def plot_opinion_evolution(results: Dict[str, Any], save_path: str = None):
    """
    Plot opinion evolution over time.
    
    Args:
        results: Simulation results dictionary
        save_path: Optional path to save the plot
    """
    mean_opinions = results['mean_opinions']
    std_opinions = results['std_opinions']
    opinion_history = results['opinion_history']
    topic = results.get('topic', 'Unknown Topic')
    
    timesteps = range(len(mean_opinions))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Mean opinion and standard deviation
    ax1.plot(timesteps, mean_opinions, 'b-', linewidth=2, label='Mean Opinion')
    ax1.fill_between(timesteps, 
                     [m - s for m, s in zip(mean_opinions, std_opinions)],
                     [m + s for m, s in zip(mean_opinions, std_opinions)],
                     alpha=0.3, color='blue', label='±1 Std Dev')
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Opinion Value')
    ax1.set_title(f'Opinion Evolution for Topic: {topic}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)
    
    # Plot 2: Individual agent opinions
    opinion_array = np.array(opinion_history)
    for i in range(opinion_array.shape[1]):
        ax2.plot(timesteps, opinion_array[:, i], 
                label=f'Agent {i}', alpha=0.7, linewidth=1.5)
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Opinion Value')
    ax2.set_title('Individual Agent Opinions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def save_simulation_data(results: Dict[str, Any], save_path: str, config: Dict[str, Any] = None):
    """
    Save simulation data to JSON file.
    
    Args:
        results: Simulation results dictionary
        save_path: Path to save the JSON file
        config: Configuration dictionary
    """
    # Create a copy of results for saving
    save_data = {
        'topic': results.get('topic', 'Unknown'),
        'mean_opinions': results['mean_opinions'],
        'std_opinions': results['std_opinions'],
        'opinion_history': results['opinion_history'],
        'posts_history': results['posts_history'],
        'interpretations_history': results['interpretations_history'],
        'final_opinions': results['final_opinions'],
        'simulation_params': results['simulation_params'],
        'random_seed': results['random_seed']
    }
    
    # Add metadata if config is provided
    if config is not None:
        save_data['metadata'] = {
            'run_timestamp': datetime.now().isoformat(),
            'config_source': 'config.json',
            'config_metadata': {
                'llm_config': config.get('llm', {}),
                'simulation_config': config.get('simulation', {}),
                'output_config': config.get('output', {})
            }
        }
    
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Simulation data saved to: {save_path}")


def print_simulation_summary(results: Dict[str, Any]):
    """
    Print a summary of simulation results.
    
    Args:
        results: Simulation results dictionary
    """
    topic = results.get('topic', 'Unknown Topic')
    mean_opinions = results['mean_opinions']
    std_opinions = results['std_opinions']
    posts_history = results['posts_history']
    interpretations_history = results['interpretations_history']
    
    print(f"\n{'='*60}")
    print(f"SIMULATION SUMMARY: {topic}")
    print(f"{'='*60}")
    
    print(f"Initial mean opinion: {mean_opinions[0]:.3f}")
    print(f"Final mean opinion: {mean_opinions[-1]:.3f}")
    print(f"Initial std dev: {std_opinions[0]:.3f}")
    print(f"Final std dev: {std_opinions[-1]:.3f}")
    
    # Check convergence
    if len(mean_opinions) > 10:
        recent_mean = mean_opinions[-10:]
        recent_std = std_opinions[-10:]
        mean_change = max(recent_mean) - min(recent_mean)
        std_change = max(recent_std) - min(recent_std)
        
        converged = mean_change < 0.01 and std_change < 0.01
        print(f"Converged: {'Yes' if converged else 'No'}")
        print(f"Mean change in last 10 timesteps: {mean_change:.3f}")
        print(f"Std dev change in last 10 timesteps: {std_change:.3f}")
    
    print(f"\nSample Posts (Timestep 0):")
    for i, post in enumerate(posts_history[0]):
        print(f"  Agent {i}: {post[:100]}...")
    
    print(f"\nSample Interpretations (Timestep 0):")
    for i, interpretations in enumerate(interpretations_history[0]):
        print(f"  Agent {i}: {[f'{x:.3f}' for x in interpretations[:3]]}...")
    
    print(f"{'='*60}") 