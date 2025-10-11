#!/usr/bin/env python3
"""
Script to run opinion dynamics experiments.
"""

import json
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from network_of_agents.runner import run_experiment

# Suppress verbose logging
os.environ["LITELLM_LOG"] = "ERROR"
logging.basicConfig(level=logging.WARNING, force=True)
for logger_name in ["litellm", "httpx", "openai", "urllib3", "requests"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

def create_visualizations(results, output_dir, config):
    """Create visualizations with simplified directory structure"""
    print("📊 Creating visualizations...")
    
    # Create plots directory directly in output_dir
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create plots for each result
    for result_name, data in results.items():
        if not isinstance(data, dict) or "opinion_history" not in data:
            continue
            
        print(f"  Creating plots for {result_name}...")
        
        opinion_history = [np.array(op) for op in data["opinion_history"]]
        n_agents = len(opinion_history[0])
        
        # Create convergence plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"{result_name.replace('_', ' ').title()} (n={n_agents})", fontsize=16)
        
        # Plot trajectories
        for i in range(n_agents):
            ax1.plot([op[i] for op in opinion_history], alpha=0.7, linewidth=0.8)
        ax1.set_title(f"Opinion Trajectories")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Opinion")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1, 1)
        
        # Plot mean evolution
        mean_opinions = [np.mean(op) for op in opinion_history]
        std_opinions = [np.std(op) for op in opinion_history]
        time_steps = range(len(mean_opinions))
        
        ax2.plot(time_steps, mean_opinions, linewidth=2, color='darkblue', label='Mean Opinion')
        ax2.fill_between(time_steps, 
                        np.array(mean_opinions) - np.array(std_opinions),
                        np.array(mean_opinions) + np.array(std_opinions),
                        alpha=0.3, color='lightblue', label='±1 Std')
        ax2.set_title("Mean Opinion Evolution")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Opinion")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(-1, 1)
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{result_name}_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create network plot if adjacency matrix available
        if "network_info" in data and "adjacency_matrix" in data["network_info"]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f"{result_name.replace('_', ' ').title()} Network (n={n_agents})", fontsize=16)
            
            adjacency = np.array(data["network_info"]["adjacency_matrix"])
            plot_network_ring(ax1, adjacency, opinion_history[0], "Initial Network")
            plot_network_ring(ax2, adjacency, opinion_history[-1], "Final Network")
            
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/{result_name}_network.png", dpi=300, bbox_inches='tight')
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
    parser = argparse.ArgumentParser(description="Run opinion dynamics experiments from config file")
    parser.add_argument("--config", default="configs/config.json", help="Configuration file")
    parser.add_argument("--output-dir", default="results", help="Output directory")
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
    
    # Use config file exactly as specified - no command line overrides
    config["output_dir"] = args.output_dir
    
    logger.info("🧪 Starting Opinion Dynamics Experiment")
    logger.info("=" * 50)
    logger.info(f"Topologies: {config.get('topologies', [])}")
    logger.info(f"Topics: {config.get('topics') or 'None (mathematical only)'}")
    logger.info(f"Agents: {config.get('n_agents', 50)}")
    logger.info(f"Convergence steps: {config.get('convergence_steps', {})}")
    
    llm_config = config.get("llm", {})
    if llm_config.get("enabled", False):
        logger.info(f"LLM: Enabled ({llm_config.get('model', 'gpt-5-mini')})")
    else:
        logger.info("LLM: Disabled")
    
    # Run experiments
    try:
        results, experiment_path = run_experiment(config)
        
        # Create visualizations
        # Flatten results structure for visualization
        flattened_results = {}
        for model, model_data in results.items():
            if isinstance(model_data, dict):
                for topology, topology_data in model_data.items():
                    if isinstance(topology_data, dict):
                        for topic, topic_data in topology_data.items():
                            if isinstance(topic_data, dict):
                                # Handle both LLM results (with summary_metrics) and pure math results (direct data)
                                if 'summary_metrics' in topic_data:
                                    # LLM results structure
                                    summary = topic_data['summary_metrics']
                                    flattened_results[f"{topology}_{model}_{topic}"] = {
                                        'opinion_history': summary.get('opinion_history', []),
                                        'mean_opinions': summary.get('mean_opinions', []),
                                        'std_opinions': summary.get('std_opinions', []),
                                        'final_opinions': summary.get('final_opinions', []),
                                        'network_info': summary.get('network_info', {}),
                                        'topic': topic
                                    }
                                elif 'opinion_history' in topic_data:
                                    # Pure math results structure
                                    flattened_results[f"{topology}_{model}_{topic}"] = {
                                        'opinion_history': topic_data.get('opinion_history', []),
                                        'mean_opinions': [],  # Will be calculated from opinion_history
                                        'std_opinions': [],   # Will be calculated from opinion_history
                                        'final_opinions': topic_data.get('final_opinions', []),
                                        'network_info': {'n_agents': len(topic_data.get('final_opinions', []))},
                                        'topic': topic
                                    }
        
        # Create visualizations in the organized structure
        logger.info(f"📊 Creating visualizations for {len(flattened_results)} results...")
        logger.info(f"📊 Flattened results keys: {list(flattened_results.keys())}")
        create_visualizations(flattened_results, experiment_path, config)
        
        logger.info("✅ Experiment completed successfully!")
        logger.info(f"📁 Results saved to organized structure: {experiment_path}")
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()
