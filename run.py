#!/usr/bin/env python3
"""
Simple script to run opinion dynamics experiments from config file.
"""

import json
import argparse
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Dict, Any
from network_of_agents.runner import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run opinion dynamics experiments from config file")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ Config file {args.config} not found!")
        return
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in config file: {e}")
        return
    
    # Set output directory
    config["output_dir"] = args.output_dir
    
    # Log experiment details
    logger.info("🧪 Starting Opinion Dynamics Experiment")
    logger.info("=" * 50)
    logger.info(f"Config: {args.config}")
    logger.info(f"Topologies: {config.get('topologies', [])}")
    logger.info(f"Topics: {config.get('topics') or 'None (mathematical only)'}")
    logger.info(f"Agents: {config.get('n_agents', 50)}")
    
    llm_config = config.get("llm", {})
    if llm_config.get("enabled", False):
        logger.info(f"LLM: Enabled ({llm_config.get('model', 'gpt-5-mini')})")
    else:
        logger.info("LLM: Disabled")
    
    # Run experiment
    try:
        results, experiment_path = run_experiment(config)
        logger.info("✅ Experiment completed successfully!")
        logger.info(f"📁 Results saved to: {experiment_path}")
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        raise

def create_visualizations(flattened_results: Dict[str, Any], experiment_path: str, config: Dict[str, Any]):
    """Create visualizations for experiment results"""
    logger = logging.getLogger(__name__)
    
    # Create plots directory
    plots_dir = os.path.join(experiment_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for result_key, data in flattened_results.items():
        try:
            # Parse the result key to extract components
            parts = result_key.split('_')
            if len(parts) >= 3:
                topology = parts[0]
                model = parts[1] 
                topic = '_'.join(parts[2:])  # Handle multi-word topics
            else:
                logger.warning(f"Could not parse result key: {result_key}")
                continue
                
            logger.info(f"📊 Creating visualizations for {topology}_{model}_{topic}")
            
            # Create convergence plot
            create_convergence_plot(data, plots_dir, topology, model, topic)
            
            # Create network plot
            create_network_plot(data, plots_dir, topology, model, topic)
            
        except Exception as e:
            logger.error(f"❌ Error creating visualization for {result_key}: {e}")

def create_convergence_plot(data: Dict[str, Any], plots_dir: str, topology: str, model: str, topic: str):
    """Create convergence plot showing opinion evolution over time with two subplots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left subplot: Individual agent trajectories
    opinion_history = data.get('opinion_history', [])
    if opinion_history:
        # Transpose to get agent trajectories (each row is one agent's trajectory)
        opinion_array = np.array(opinion_history).T
        n_agents = opinion_array.shape[0]
        
        # Plot each agent's trajectory
        for agent_id in range(n_agents):
            ax1.plot(opinion_array[agent_id], alpha=0.6, linewidth=0.8)
        
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Opinion Value')
        ax1.set_title(f'Individual Agent Trajectories ({n_agents} agents)')
        ax1.grid(True, alpha=0.3)
    
    # Right subplot: Mean ± 1 std
    mean_opinions = data.get('mean_opinions', [])
    std_opinions = data.get('std_opinions', [])
    
    if mean_opinions:
        ax2.plot(mean_opinions, label='Mean Opinion', linewidth=2, color='red')
        
        if std_opinions:
            mean_array = np.array(mean_opinions)
            std_array = np.array(std_opinions)
            ax2.fill_between(range(len(mean_opinions)), 
                            mean_array - std_array, 
                            mean_array + std_array, 
                            alpha=0.3, label='±1 Std Dev', color='red')
    
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Opinion Value')
    ax2.set_title('Mean Opinion ± 1 Standard Deviation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{topology.title()} {model.title()} {topic.title()} - Opinion Convergence', fontsize=14)
    
    # Save plot
    filename = f"{topology}_{model}_{topic}_convergence.png"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger = logging.getLogger(__name__)
    logger.info(f"  📈 Saved convergence plot: {filename}")

def create_network_plot(data: Dict[str, Any], plots_dir: str, topology: str, model: str, topic: str):
    """Create network plot showing network structure with initial and final opinions"""
    network_info = data.get('network_info', {})
    final_opinions = data.get('final_opinions', [])
    opinion_history = data.get('opinion_history', [])
    
    if not network_info or not final_opinions or not opinion_history:
        logger = logging.getLogger(__name__)
        logger.warning(f"No network data available for {topology}_{model}_{topic}")
        return
    
    # Get initial opinions (first timestep)
    initial_opinions = opinion_history[0] if opinion_history else []
    
    if not initial_opinions:
        logger = logging.getLogger(__name__)
        logger.warning(f"No initial opinion data available for {topology}_{model}_{topic}")
        return
    
    # Create network from adjacency matrix
    adjacency_matrix = np.array(network_info.get('adjacency_matrix', []))
    if adjacency_matrix.size == 0:
        return
        
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Create subplots with better spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plt.subplots_adjust(wspace=0.3)  # Add more space between subplots
    
    # Position nodes using spring layout (same for both plots)
    pos = nx.spring_layout(G, seed=42)
    
    # Determine color scale for both plots
    all_opinions = initial_opinions + final_opinions
    vmin, vmax = min(all_opinions), max(all_opinions)
    
    # Left subplot: Initial network with starting opinions
    initial_colors = plt.cm.RdYlBu_r(np.array(initial_opinions))
    nx.draw_networkx_nodes(G, pos, node_color=initial_colors, 
                          node_size=100, alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax1)
    
    ax1.set_title('Initial Network with Starting Opinions')
    ax1.axis('off')
    
    # Right subplot: Final network with ending opinions
    final_colors = plt.cm.RdYlBu_r(np.array(final_opinions))
    nx.draw_networkx_nodes(G, pos, node_color=final_colors, 
                          node_size=100, alpha=0.8, ax=ax2)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax2)
    
    ax2.set_title('Final Network with Ending Opinions')
    ax2.axis('off')
    
    # Add colorbars after both plots are drawn, positioned properly
    # Create a single colorbar that applies to both plots
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, 
                              norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    
    # Position colorbar on the right side of the figure
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.8, aspect=20)
    cbar.set_label('Opinion Value', rotation=270, labelpad=20)
    
    # Overall title
    fig.suptitle(f'{topology.title()} {model.title()} {topic.title()} - Network Evolution', fontsize=14)
    
    # Save plot
    filename = f"{topology}_{model}_{topic}_network.png"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger = logging.getLogger(__name__)
    logger.info(f"  🕸️ Saved network plot: {filename}")

if __name__ == "__main__":
    main()
