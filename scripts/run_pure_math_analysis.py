#!/usr/bin/env python3
"""
Pure mathematical analysis and visualization for canonical experimental configurations.

This script runs the pure mathematical models (DeGroot and Friedkin-Johnsen) 
without LLM involvement and generates visualizations of network topologies
and convergence trajectories.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, List
import logging

from network_of_agents.canonical_configs import CANONICAL_EXPERIMENTS, CANONICAL_TOPICS
from network_of_agents.network.graph_generator import create_network_model, get_network_info
from network_of_agents.core.mathematics import update_opinions_pure_degroot, update_opinions_friedkin_johnsen

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pure_math_simulation(experiment_name: str, max_timesteps: int = 200) -> Dict[str, Any]:
    """
    Run pure mathematical simulation for a given experiment configuration.
    
    Args:
        experiment_name: Name of the canonical experiment
        max_timesteps: Maximum number of timesteps
        
    Returns:
        Simulation results
    """
    logger.info(f"Running pure math simulation: {experiment_name}")
    
    # Get experiment configuration
    config = CANONICAL_EXPERIMENTS[experiment_name]
    
    # Create network
    network = create_network_model(
        config["topology"], 
        config["topology_params"], 
        random_seed=42
    )
    
    # Get network info
    network_info = get_network_info(network.get_adjacency_matrix())
    n_agents = network_info["n_agents"]
    
    # Initialize opinions
    if config["opinion_distribution"] == "normal":
        opinions = np.random.normal(
            config["opinion_params"]["mu"], 
            config["opinion_params"]["sigma"], 
            n_agents
        )
    else:
        raise ValueError(f"Unsupported opinion distribution: {config['opinion_distribution']}")
    
    # Clamp opinions to [-1, 1] range
    opinions = np.clip(opinions, -1.0, 1.0)
    
    # Convert to [0, 1] range for mathematical models
    math_opinions = (opinions + 1) / 2
    
    # Store initial state
    opinion_history = [opinions.copy()]
    math_opinion_history = [math_opinions.copy()]
    
    # Get adjacency matrix
    adjacency = network.get_adjacency_matrix()
    
    # Run simulation
    for timestep in range(max_timesteps):
        if config["model"] == "degroot":
            new_math_opinions = update_opinions_pure_degroot(
                math_opinions, 
                adjacency, 
                config["model_params"].get("epsilon", 1e-6)
            )
        elif config["model"] == "friedkin_johnsen":
            # Create susceptibility matrix
            lambda_val = config["model_params"].get("lambda", 0.8)
            stubborn_fraction = config["model_params"].get("stubborn_fraction", 0.1)
            
            n_stubborn = int(n_agents * stubborn_fraction)
            susceptibility = np.ones(n_agents) * lambda_val
            if n_stubborn > 0:
                # Make first n_stubborn agents stubborn
                susceptibility[:n_stubborn] = 0.0
            
            new_math_opinions = update_opinions_friedkin_johnsen(
                math_opinions,
                adjacency,
                susceptibility
            )
        else:
            raise ValueError(f"Unknown model: {config['model']}")
        
        # Update opinions
        math_opinions = new_math_opinions
        opinions = 2 * math_opinions - 1  # Convert back to [-1, 1]
        
        # Store history
        opinion_history.append(opinions.copy())
        math_opinion_history.append(math_opinions.copy())
        
        # Check convergence
        if timestep > 0:
            change = np.mean(np.abs(opinions - opinion_history[-2]))
            if change < 1e-6:
                logger.info(f"Converged at timestep {timestep}")
                break
    
    # Calculate final statistics
    final_mean = np.mean(opinions)
    final_std = np.std(opinions)
    
    return {
        "experiment_name": experiment_name,
        "config": config,
        "network_info": network_info,
        "opinion_history": [opinions.tolist() for opinions in opinion_history],
        "math_opinion_history": [opinions.tolist() for opinions in math_opinion_history],
        "final_mean": final_mean,
        "final_std": final_std,
        "converged": len(opinion_history) < max_timesteps,
        "timesteps": len(opinion_history) - 1,
        "adjacency_matrix": adjacency.tolist()
    }

def plot_network_topology(experiment_name: str, results: Dict[str, Any], save_dir: str):
    """
    Plot the network topology for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        results: Simulation results
        save_dir: Directory to save plots
    """
    adjacency = np.array(results["adjacency_matrix"])
    n_agents = adjacency.shape[0]
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adjacency)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, seed=42)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=100, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
    
    # Add labels for small networks
    if n_agents <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"{experiment_name}\n"
              f"Nodes: {n_agents}, Edges: {results['network_info']['total_edges']}, "
              f"Density: {results['network_info']['density']:.3f}")
    plt.axis('off')
    
    # Save plot
    filename = f"{experiment_name}_topology.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved topology plot: {filepath}")

def plot_convergence_trajectory(experiment_name: str, results: Dict[str, Any], save_dir: str):
    """
    Plot the convergence trajectory for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        results: Simulation results
        save_dir: Directory to save plots
    """
    opinion_history = np.array(results["opinion_history"])
    timesteps = len(opinion_history)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot individual agent trajectories
    for i in range(min(opinion_history.shape[1], 20)):  # Limit to first 20 agents for clarity
        plt.plot(range(timesteps), opinion_history[:, i], alpha=0.3, linewidth=0.5)
    
    # Plot mean trajectory
    mean_opinions = np.mean(opinion_history, axis=1)
    plt.plot(range(timesteps), mean_opinions, 'r-', linewidth=2, label='Mean Opinion')
    
    # Plot standard deviation
    std_opinions = np.std(opinion_history, axis=1)
    plt.fill_between(range(timesteps), 
                     mean_opinions - std_opinions, 
                     mean_opinions + std_opinions, 
                     alpha=0.3, color='red', label='±1 Std Dev')
    
    plt.xlabel('Timestep')
    plt.ylabel('Opinion')
    plt.title(f"{experiment_name} - Convergence Trajectory\n"
              f"Final Mean: {results['final_mean']:.4f}, "
              f"Final Std: {results['final_std']:.4f}, "
              f"Timesteps: {results['timesteps']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    filename = f"{experiment_name}_convergence.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved convergence plot: {filepath}")

def run_all_pure_math_analysis(save_dir: str = "results/pure_math") -> Dict[str, Any]:
    """
    Run pure mathematical analysis for all canonical experiments.
    
    Args:
        save_dir: Directory to save results and plots
        
    Returns:
        Summary of all results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Run all experiments
    all_results = {}
    
    for experiment_name in CANONICAL_EXPERIMENTS.keys():
        logger.info(f"Processing experiment: {experiment_name}")
        
        try:
            # Run simulation
            results = run_pure_math_simulation(experiment_name)
            all_results[experiment_name] = results
            
            # Generate plots
            plot_network_topology(experiment_name, results, save_dir)
            plot_convergence_trajectory(experiment_name, results, save_dir)
            
        except Exception as e:
            logger.error(f"Failed to process {experiment_name}: {e}")
            all_results[experiment_name] = {"error": str(e)}
    
    # Calculate summary statistics
    summary = {
        "total_experiments": len(all_results),
        "successful_experiments": len([r for r in all_results.values() if "error" not in r]),
        "experiment_summaries": {}
    }
    
    for experiment_name, results in all_results.items():
        if "error" not in results:
            summary["experiment_summaries"][experiment_name] = {
                "final_mean": results["final_mean"],
                "final_std": results["final_std"],
                "converged": results["converged"],
                "timesteps": results["timesteps"],
                "n_agents": results["network_info"]["n_agents"],
                "density": results["network_info"]["density"]
            }
    
    # Save results
    results_file = os.path.join(save_dir, "pure_math_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    summary_file = os.path.join(save_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Pure math analysis completed. Results saved to {save_dir}")
    return summary

def main():
    """Main function to run pure mathematical analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pure mathematical analysis")
    parser.add_argument("--save-dir", default="results/pure_math", help="Directory to save results")
    parser.add_argument("--experiment", help="Run specific experiment only")
    
    args = parser.parse_args()
    
    if args.experiment:
        # Run single experiment
        if args.experiment not in CANONICAL_EXPERIMENTS:
            logger.error(f"Unknown experiment: {args.experiment}")
            return
        
        results = run_pure_math_simulation(args.experiment)
        plot_network_topology(args.experiment, results, args.save_dir)
        plot_convergence_trajectory(args.experiment, results, args.save_dir)
        
        print(f"Results for {args.experiment}:")
        print(f"Final Mean: {results['final_mean']:.4f}")
        print(f"Final Std: {results['final_std']:.4f}")
        print(f"Converged: {results['converged']}")
        print(f"Timesteps: {results['timesteps']}")
    else:
        # Run all experiments
        summary = run_all_pure_math_analysis(args.save_dir)
        print(f"Analysis complete. Summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()
