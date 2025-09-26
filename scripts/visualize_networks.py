#!/usr/bin/env python3
"""
Quick network topology visualization script.

This script generates plots of all 6 canonical network topologies
without running simulations, useful for understanding the network structures.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any

from network_of_agents.canonical_configs import CANONICAL_EXPERIMENTS
from network_of_agents.network.graph_generator import create_network_model, get_network_info

def plot_network_topology(experiment_name: str, config: Dict[str, Any], save_dir: str):
    """
    Plot a single network topology.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        save_dir: Directory to save plots
    """
    # Create network
    network = create_network_model(
        config["topology"], 
        config["topology_params"], 
        random_seed=42
    )
    
    # Get network info
    network_info = get_network_info(network.get_adjacency_matrix())
    adjacency = network.get_adjacency_matrix()
    n_agents = network_info["n_agents"]
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adjacency)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Network topology
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes by degree
    degrees = dict(G.degree())
    node_colors = [degrees[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=100, alpha=0.7, cmap='viridis', ax=ax1)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, ax=ax1)
    
    # Add labels for small networks
    if n_agents <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
    
    ax1.set_title(f"{experiment_name}\n"
                  f"Nodes: {n_agents}, Edges: {network_info['total_edges']}\n"
                  f"Density: {network_info['density']:.3f}, "
                  f"Avg Degree: {network_info['average_degree']:.1f}")
    ax1.axis('off')
    
    # Plot 2: Degree distribution
    degrees = list(dict(G.degree()).values())
    ax2.hist(degrees, bins=min(20, len(set(degrees))), alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Degree')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Degree Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Max Degree: {max(degrees)}\nMin Degree: {min(degrees)}\nStd Dev: {np.std(degrees):.2f}"
    ax2.text(0.7, 0.9, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{experiment_name}_network_analysis.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved network analysis: {filepath}")

def create_comparison_plot(save_dir: str):
    """
    Create a comparison plot showing all network topologies.
    
    Args:
        save_dir: Directory to save plots
    """
    experiments = list(CANONICAL_EXPERIMENTS.keys())
    n_experiments = len(experiments)
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, experiment_name in enumerate(experiments):
        config = CANONICAL_EXPERIMENTS[experiment_name]
        
        # Create network
        network = create_network_model(
            config["topology"], 
            config["topology_params"], 
            random_seed=42
        )
        
        # Get network info
        network_info = get_network_info(network.get_adjacency_matrix())
        adjacency = network.get_adjacency_matrix()
        n_agents = network_info["n_agents"]
        
        # Create NetworkX graph
        G = nx.from_numpy_array(adjacency)
        
        # Use spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=50, alpha=0.7, ax=axes[i])
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.3, ax=axes[i])
        
        # Add title
        axes[i].set_title(f"{experiment_name}\n"
                         f"N={n_agents}, E={network_info['total_edges']}\n"
                         f"D={network_info['density']:.3f}", fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_experiments, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Canonical Network Topologies Comparison', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    filename = "all_networks_comparison.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {filepath}")

def main():
    """Main function to generate network visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize network topologies")
    parser.add_argument("--save-dir", default="results/visualizations", help="Directory to save plots")
    parser.add_argument("--experiment", help="Visualize specific experiment only")
    parser.add_argument("--comparison", action="store_true", help="Create comparison plot")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.experiment:
        # Visualize single experiment
        if args.experiment not in CANONICAL_EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            return
        
        config = CANONICAL_EXPERIMENTS[args.experiment]
        plot_network_topology(args.experiment, config, args.save_dir)
    else:
        # Visualize all experiments
        print("Generating network visualizations...")
        
        for experiment_name, config in CANONICAL_EXPERIMENTS.items():
            print(f"Processing {experiment_name}...")
            plot_network_topology(experiment_name, config, args.save_dir)
        
        # Create comparison plot
        if args.comparison:
            print("Creating comparison plot...")
            create_comparison_plot(args.save_dir)
        
        print(f"All visualizations saved to {args.save_dir}")

if __name__ == "__main__":
    main()
