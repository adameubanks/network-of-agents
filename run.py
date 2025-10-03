#!/usr/bin/env python3
"""
Script to run opinion dynamics experiments.
"""

import json
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from network_of_agents.runner import run_experiment

def create_visualizations(results, output_dir):
    """Create visualizations for the results"""
    print("📊 Creating visualizations...")
    
    for network_name, data in results.items():
        if isinstance(data, dict) and "error" in data:
            continue
            
        print(f"  Creating plots for {network_name}...")
        
        # Convert data back to numpy
        if "opinion_history" not in data:
            continue
            
        opinion_history = [np.array(op) for op in data["opinion_history"]]
        n_agents = len(opinion_history[0])
        
        # Create Plot 1: Opinion trajectories and mean evolution
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig1.suptitle(f"Opinion Dynamics: {network_name.title()} (n={n_agents})", fontsize=16)
        
        # Plot 1a: Opinion trajectories (all agents)
        for i in range(len(opinion_history[0])):
            ax1.plot([op[i] for op in opinion_history], alpha=0.7, linewidth=0.8)
        ax1.set_title(f"Opinion Trajectories (All {len(opinion_history[0])} Agents)")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Opinion")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1, 1)
        
        # Plot 1b: Mean opinion evolution with error bars
        mean_opinions = [np.mean(op) for op in opinion_history]
        std_opinions = [np.std(op) for op in opinion_history]
        
        time_steps = range(len(mean_opinions))
        ax2.plot(time_steps, mean_opinions, linewidth=2, color='red', label='Mean')
        ax2.fill_between(time_steps, 
                        np.array(mean_opinions) - np.array(std_opinions),
                        np.array(mean_opinions) + np.array(std_opinions),
                        alpha=0.3, color='red', label='±1 std')
        ax2.set_title("Mean Opinion Evolution with Error Bars")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Opinion")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(-1, 1)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{network_name}_dynamics.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create Plot 2: Network structure (initial and final)
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
        fig2.suptitle(f"Network Structure: {network_name.title()} (n={n_agents})", fontsize=16)
        
        # Plot 2a: Initial network structure (ring layout)
        if "network_info" in data and "adjacency_matrix" in data["network_info"]:
            adjacency = np.array(data["network_info"]["adjacency_matrix"])
            plot_network_ring(ax3, adjacency, opinion_history[0], "Initial Network")
        else:
            ax3.text(0.5, 0.5, "Network structure not available", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Initial Network Structure")
        
        # Plot 2b: Final network structure (ring layout)
        if "network_info" in data and "adjacency_matrix" in data["network_info"]:
            adjacency = np.array(data["network_info"]["adjacency_matrix"])
            plot_network_ring(ax4, adjacency, opinion_history[-1], "Final Network")
        else:
            ax4.text(0.5, 0.5, "Network structure not available", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Final Network Structure")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{network_name}_network.png", dpi=150, bbox_inches='tight')
        plt.close()

def plot_network_ring(ax, adjacency_matrix, opinions, title):
    """Plot network in ring layout with node colors based on opinions"""
    n_nodes = adjacency_matrix.shape[0]
    
    # Create ring coordinates
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Plot edges
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adjacency_matrix[i, j] > 0:
                ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.3, linewidth=0.5)
    
    # Plot nodes with colors based on opinions
    scatter = ax.scatter(x, y, c=opinions, s=100, cmap='RdBu_r', vmin=-1, vmax=1, 
                        edgecolors='black', linewidth=0.5)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Opinion', rotation=270, labelpad=15)

def main():
    parser = argparse.ArgumentParser(description="Run opinion dynamics experiments")
    parser.add_argument("--config", default="configs/config.json", help="Configuration file")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--networks", nargs="+", 
                       default=["smallworld", "scalefree", "random", "echo", "karate"],
                       help="Network types to test")
    parser.add_argument("--topics", nargs="+", help="Topics for LLM experiments")
    parser.add_argument("--n-agents", type=int, default=30, help="Number of agents")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum simulation steps")
    parser.add_argument("--llm", action="store_true", help="Enable LLM experiments")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load or create config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.info(f"Config file {args.config} not found, using defaults")
        config = {}
    
    # Override config with command line arguments
    config.update({
        "output_dir": args.output_dir,
        "networks": args.networks,
        "topics": args.topics or [],
        "n_agents": args.n_agents,
        "max_steps": args.max_steps,
        "random_seed": 42
    })
    
    # Update LLM config if provided
    if args.llm:
        if "llm" not in config:
            config["llm"] = {}
        config["llm"]["enabled"] = True
    
    logger.info("🧪 Starting Opinion Dynamics Experiment")
    logger.info("=" * 50)
    logger.info(f"Networks: {config['networks']}")
    logger.info(f"Topics: {config['topics'] or 'None (mathematical only)'}")
    logger.info(f"Agents: {config['n_agents']}")
    logger.info(f"Steps: {config['max_steps']}")
    
    llm_config = config.get("llm", {})
    if llm_config.get("enabled", False):
        logger.info(f"LLM: Enabled ({llm_config.get('model', 'gpt-5-mini')})")
    else:
        logger.info("LLM: Disabled")
    
    # Run experiments
    try:
        results = run_experiment(config)
        
        # Create visualizations
        create_visualizations(results, args.output_dir)
        
        logger.info("✅ Experiment completed successfully!")
        logger.info(f"📁 Results saved to {args.output_dir}/")
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()
