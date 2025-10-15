#!/usr/bin/env python3
"""
Simple script to run opinion dynamics experiments from config file.
"""

import json
import argparse
import logging
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Dict, Any, List
from network_of_agents.runner import run_experiment

# Suppress NumPy deprecation warnings from NetworkX
warnings.filterwarnings("ignore", message=".*alltrue.*", category=DeprecationWarning)

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
        
        # Generate visualizations
        logger.info("📊 Generating visualizations...")
        create_visualizations(results, experiment_path, config)
        logger.info("✅ Visualizations completed!")
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        raise

def create_visualizations(results: List[Dict[str, Any]], experiment_path: str, config: Dict[str, Any]):
    """Create visualizations for experiment results"""
    logger = logging.getLogger(__name__)
    
    # Create plots directory
    plots_dir = os.path.join(experiment_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Group results by topic
    results_by_topic = {}
    for result in results:
        if "error" in result:
            continue
            
        metadata = result.get("experiment_metadata", {})
        topics = metadata.get("topics", [])
        topic = topics[0] if topics else "pure_math_model"
        
        if topic not in results_by_topic:
            results_by_topic[topic] = []
        results_by_topic[topic].append(result)
    
    # Create visualizations for each topic
    for topic, topic_results in results_by_topic.items():
        for result in topic_results:
            try:
                # Extract metadata
                metadata = result.get("experiment_metadata", {})
                model = metadata.get("model", "unknown")
                topology = metadata.get("topology", "unknown")
                
                # Prepare data for plotting
                data = {
                    'opinion_history': result.get('opinion_history', []),
                    'mean_opinions': result.get('mean_opinions', []),
                    'std_opinions': result.get('std_opinions', []),
                    'final_opinions': result.get('final_opinions', []),
                    'network_info': result.get('network_info', {})
                }
                
                if not data['opinion_history']:
                    logger.warning(f"No opinion history found for {topic}")
                    continue
                    
                logger.info(f"📊 Creating visualizations for {topic}")
                
                # Create convergence plot with simplified naming
                create_convergence_plot(data, plots_dir, topology, model, topic)
                
                # Create network plot with simplified naming
                create_network_plot(data, plots_dir, topology, model, topic)
                
            except Exception as e:
                logger.error(f"❌ Error creating visualization for {topic}: {e}")

def create_convergence_plot(data: Dict[str, Any], plots_dir: str, topology: str, model: str, topic: str):
    """Create convergence plot showing opinion evolution over time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left subplot: Individual agent trajectories
    opinion_history = data.get('opinion_history', [])
    if opinion_history:
        opinion_array = np.array(opinion_history).T
        n_agents = opinion_array.shape[0]
        
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
    
    fig.suptitle(f'{topology.title()} {model.title()} {topic.title()} - Opinion Convergence', fontsize=14)
    
    filename = f"{topic}_convergence.png"
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
    
    initial_opinions = opinion_history[0] if opinion_history else []
    
    if not initial_opinions:
        logger = logging.getLogger(__name__)
        logger.warning(f"No initial opinion data available for {topology}_{model}_{topic}")
        return
    
    adjacency_matrix = np.array(network_info.get('adjacency_matrix', []))
    if adjacency_matrix.size == 0:
        return
        
    G = nx.from_numpy_array(adjacency_matrix)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plt.subplots_adjust(wspace=0.3)
    
    # Position nodes in a circle
    pos = nx.circular_layout(G)
    
    # Use full opinion range for consistent color mapping
    vmin, vmax = -1.0, 1.0
    
    # Initial network
    initial_colors = plt.cm.RdYlBu_r((np.array(initial_opinions) - vmin) / (vmax - vmin))
    nx.draw_networkx_nodes(G, pos, node_color=initial_colors, 
                          node_size=100, alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax1)
    ax1.set_title('Initial Network with Starting Opinions')
    ax1.axis('off')
    
    # Final network
    final_colors = plt.cm.RdYlBu_r((np.array(final_opinions) - vmin) / (vmax - vmin))
    nx.draw_networkx_nodes(G, pos, node_color=final_colors, 
                          node_size=100, alpha=0.8, ax=ax2)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax2)
    ax2.set_title('Final Network with Ending Opinions')
    ax2.axis('off')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, 
                              norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.8, aspect=20)
    cbar.set_label('Opinion Value', rotation=270, labelpad=20)
    
    fig.suptitle(f'{topology.title()} {model.title()} {topic.title()} - Network Evolution', fontsize=14)
    
    filename = f"{topic}_network.png"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger = logging.getLogger(__name__)
    logger.info(f"  🕸️ Saved network plot: {filename}")


if __name__ == "__main__":
    main()
