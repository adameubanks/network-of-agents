#!/usr/bin/env python3
"""
Generate network visualizations with opinion values for the experimental design document.

This script creates individual network plots for each canonical configuration,
showing the network topology with opinion values (initialized with seed 42) 
visible on the nodes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any
import argparse

from network_of_agents.canonical_configs import CANONICAL_EXPERIMENTS
from network_of_agents.network.graph_generator import create_network_model, get_network_info

def generate_opinions(config: Dict[str, Any], n_agents: int, random_seed: int = 42) -> np.ndarray:
    """
    Generate initial opinions based on configuration.
    
    Args:
        config: Experiment configuration
        n_agents: Number of agents
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of initial opinions in [-1, 1] range
    """
    np.random.seed(random_seed)
    
    if config["opinion_distribution"] == "normal":
        mu = config["opinion_params"]["mu"]
        sigma = config["opinion_params"]["sigma"]
        opinions = np.random.normal(mu, sigma, n_agents)
        # Clip to [-1, 1] range
        opinions = np.clip(opinions, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown opinion distribution: {config['opinion_distribution']}")
    
    return opinions

def plot_network_with_opinions(experiment_name: str, config: Dict[str, Any], save_dir: str):
    """
    Plot a single network topology with opinion values visible.
    
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
    
    # Generate initial opinions
    opinions = generate_opinions(config, n_agents, random_seed=42)
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adjacency)
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot: Network topology with opinion values
    # Use different layouts based on network type to better show structure
    if config["topology"] == "watts_strogatz":
        # For small-world, use circular layout to show ring structure
        pos = nx.circular_layout(G)
    elif config["topology"] == "barabasi_albert":
        # For scale-free, use spring layout with larger k to show hub structure
        pos = nx.spring_layout(G, seed=42, k=8, iterations=300)
    elif config["topology"] == "stochastic_block_model":
        # For echo chambers, use spring layout to show community structure
        pos = nx.spring_layout(G, seed=42, k=6, iterations=300)
    elif config["topology"] == "zachary_karate_club":
        # For karate club, use spring layout
        pos = nx.spring_layout(G, seed=42, k=3, iterations=200)
    else:
        # For random and others, use spring layout with moderate k
        pos = nx.spring_layout(G, seed=42, k=5, iterations=200)
    
    # Color nodes by opinion values (red for negative, blue for positive)
    node_colors = opinions
    node_sizes = 300 + 100 * np.abs(opinions)  # Size based on opinion strength
    
    # For Friedkin-Johnsen, make stubborn agents visually distinct
    if config["model"] == "friedkin_johnsen":
        stubborn_fraction = config["model_params"].get("stubborn_fraction", 0.1)
        n_stubborn = int(n_agents * stubborn_fraction)
        
        # Draw stubborn agents with different style (larger, outlined)
        stubborn_nodes = list(range(n_stubborn))
        flexible_nodes = list(range(n_stubborn, n_agents))
        
        # Draw flexible agents first
        if flexible_nodes:
            flexible_colors = [opinions[i] for i in flexible_nodes]
            flexible_sizes = [node_sizes[i] for i in flexible_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=flexible_nodes, 
                                  node_color=flexible_colors, node_size=flexible_sizes, 
                                  alpha=0.8, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax)
        
        # Draw stubborn agents with outline
        if stubborn_nodes:
            stubborn_colors = [opinions[i] for i in stubborn_nodes]
            stubborn_sizes = [node_sizes[i] + 200 for i in stubborn_nodes]  # Larger
            nx.draw_networkx_nodes(G, pos, nodelist=stubborn_nodes, 
                                  node_color=stubborn_colors, node_size=stubborn_sizes, 
                                  alpha=0.9, cmap='RdBu_r', vmin=-1, vmax=1, ax=ax,
                                  edgecolors='black', linewidths=3)  # Black outline
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.6, width=0.8, ax=ax)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='black', linewidth=2, label='Stubborn Agents'),
            Patch(facecolor='lightgray', label='Flexible Agents')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
    else:
        # Draw all nodes normally for other models
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                      node_size=node_sizes, alpha=0.8, 
                                      cmap='RdBu_r', vmin=-1, vmax=1, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.6, width=0.8, ax=ax)
    
    # Add opinion values as labels
    if n_agents <= 50:
        labels = {i: f"{opinions[i]:.2f}" for i in range(n_agents)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Add colorbar (only if we have nodes from the normal drawing)
    if config["model"] != "friedkin_johnsen":
        cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
        cbar.set_label('Opinion Value', rotation=270, labelpad=15)
    else:
        # For Friedkin-Johnsen, create a colorbar manually
        from matplotlib.cm import RdBu_r
        from matplotlib.colors import Normalize
        sm = plt.cm.ScalarMappable(cmap=RdBu_r, norm=Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Opinion Value', rotation=270, labelpad=15)
    
    ax.set_title(f"{config['name']}\n"
                 f"Network: {config['topology']}\n"
                 f"Agents: {n_agents}, Edges: {network_info['total_edges']}\n"
                 f"Density: {network_info['density']:.3f}, "
                 f"Mean Opinion: {np.mean(opinions):.3f}",
                 fontsize=12, pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{experiment_name}_network_with_opinions.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved network visualization: {filepath}")
    return filepath

def generate_opinion_distribution_plot(save_dir: str = "paper/figures"):
    """
    Generate opinion distribution visualization for the Opinion Initialization section.
    
    Args:
        save_dir: Directory to save plots
    """
    # Generate opinions using the standard configuration
    config = CANONICAL_EXPERIMENTS["degroot_smallworld"]  # Use any config for opinion generation
    n_agents = 50
    opinions = generate_opinions(config, n_agents, random_seed=42)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot histogram
    ax.hist(opinions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(np.mean(opinions), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(opinions):.3f}')
    ax.axvline(np.median(opinions), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(opinions):.3f}')
    ax.set_xlabel('Opinion Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Initial Opinion Distribution\n(Seed 42, N=50)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Std: {np.std(opinions):.3f}\nMin: {np.min(opinions):.3f}\nMax: {np.max(opinions):.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    filename = "opinion_distribution.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved opinion distribution plot: {filepath}")
    return filepath

def create_all_network_visualizations(save_dir: str = "paper/figures"):
    """
    Create network visualizations for all canonical configurations.
    
    Args:
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    generated_files = []
    
    for experiment_name, config in CANONICAL_EXPERIMENTS.items():
        print(f"Generating visualization for {experiment_name}...")
        filepath = plot_network_with_opinions(experiment_name, config, save_dir)
        generated_files.append(filepath)
    
    print(f"\nAll network visualizations saved to {save_dir}")
    print("Generated files:")
    for filepath in generated_files:
        print(f"  - {filepath}")
    
    return generated_files

def main():
    """Main function to generate network visualizations."""
    parser = argparse.ArgumentParser(description="Generate network visualizations with opinion values")
    parser.add_argument("--save-dir", default="paper/figures", help="Directory to save plots")
    parser.add_argument("--experiment", help="Generate specific experiment only")
    
    args = parser.parse_args()
    
    if args.experiment:
        # Generate single experiment
        if args.experiment not in CANONICAL_EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available experiments: {list(CANONICAL_EXPERIMENTS.keys())}")
            return
        
        config = CANONICAL_EXPERIMENTS[args.experiment]
        os.makedirs(args.save_dir, exist_ok=True)
        plot_network_with_opinions(args.experiment, config, args.save_dir)
    else:
        # Generate all experiments
        create_all_network_visualizations(args.save_dir)
        
        # Generate opinion distribution plot
        print("Generating opinion distribution plot...")
        generate_opinion_distribution_plot(args.save_dir)

if __name__ == "__main__":
    main()
