"""
Visualization module for opinion dynamics simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import json
import os
from datetime import datetime
import networkx as nx
from matplotlib.patches import Patch
import glob
import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
    
    # Check if results are partial
    is_partial = results.get('is_partial', False)
    if is_partial:
        completed_timesteps = results.get('completed_timesteps', len(mean_opinions))
        total_timesteps = results.get('total_timesteps', len(mean_opinions))
        error_msg = results.get('error', 'Unknown error')
        title_suffix = f" (PARTIAL - {completed_timesteps}/{total_timesteps} timesteps)"
    else:
        title_suffix = ""
    
    timesteps = range(len(mean_opinions))
    
    # Create figure with subplots (much taller for improved readability)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 24), constrained_layout=True)
    
    # Plot 1: Mean opinion and standard deviation
    ax1.plot(timesteps, mean_opinions, 'b-', linewidth=2, label='Mean Opinion')
    ax1.fill_between(timesteps, 
                     [m - s for m, s in zip(mean_opinions, std_opinions)],
                     [m + s for m, s in zip(mean_opinions, std_opinions)],
                     alpha=0.3, color='blue', label='±1 Std Dev')
    
    # Add vertical line for partial results to show where simulation stopped
    if is_partial and len(mean_opinions) > 0:
        ax1.axvline(x=len(mean_opinions)-1, color='red', linestyle='--', alpha=0.7, 
                   label=f'Simulation stopped (timestep {len(mean_opinions)})')
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Opinion Value')
    ax1.set_title(f'Opinion Evolution for Topic: {topic}{title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)
    
    # Plot 2: Individual agent opinions
    opinion_array = np.array(opinion_history)
    for i in range(opinion_array.shape[1]):
        ax2.plot(timesteps, opinion_array[:, i], 
                label=f'Agent {i}', alpha=0.7, linewidth=1.5)
    
    # Add vertical line for partial results
    if is_partial and len(opinion_history) > 0:
        ax2.axvline(x=len(opinion_history)-1, color='red', linestyle='--', alpha=0.7, 
                   label=f'Simulation stopped (timestep {len(opinion_history)})')
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Opinion Value')
    ax2.set_title('Individual Agent Opinions')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)
    
    # Add error message for partial results
    if is_partial:
        fig.suptitle(f"⚠️ PARTIAL RESULTS - Simulation interrupted due to: {error_msg}", 
                    fontsize=12, color='red', y=0.98)
    
    # With constrained_layout=True above, tight_layout is unnecessary
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    


def plot_mean_std(results: Dict[str, Any], save_path: str = None):
    """
    Plot mean opinion with ±1 std deviation as a single figure.
    
    Args:
        results: Simulation results dictionary
        save_path: Optional path to save the plot
    """
    mean_opinions = results['mean_opinions']
    std_opinions = results['std_opinions']
    topic = results.get('topic', 'Unknown Topic')

    # Check if results are partial
    is_partial = results.get('is_partial', False)
    if is_partial:
        completed_timesteps = results.get('completed_timesteps', len(mean_opinions))
        total_timesteps = results.get('total_timesteps', len(mean_opinions))
        error_msg = results.get('error', 'Unknown error')
        title_suffix = f" (PARTIAL - {completed_timesteps}/{total_timesteps} timesteps)"
    else:
        title_suffix = ""

    timesteps = range(len(mean_opinions))

    plt.figure(figsize=(16, 12))
    plt.plot(timesteps, mean_opinions, 'b-', linewidth=2, label='Mean Opinion')
    plt.fill_between(timesteps,
                     [m - s for m, s in zip(mean_opinions, std_opinions)],
                     [m + s for m, s in zip(mean_opinions, std_opinions)],
                     alpha=0.3, color='blue', label='±1 Std Dev')

    # Add vertical line for partial results to show where simulation stopped
    if is_partial and len(mean_opinions) > 0:
        plt.axvline(x=len(mean_opinions) - 1, color='red', linestyle='--', alpha=0.7,
                    label=f'Simulation stopped (timestep {len(mean_opinions)})')

    plt.xlabel('Timestep')
    plt.ylabel('Opinion Value')
    plt.title(f'Mean Opinion for Topic: {topic}{title_suffix}')
    plt.grid(True, alpha=0.3)
    plt.ylim(-1, 1)
    plt.legend()

    if is_partial:
        plt.suptitle(f"⚠️ PARTIAL RESULTS - Simulation interrupted due to: {error_msg}",
                     fontsize=12, color='red', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')



def plot_individual_opinions(results: Dict[str, Any], save_path: str = None):
    """
    Plot individual agent opinions as a single figure.
    
    Args:
        results: Simulation results dictionary
        save_path: Optional path to save the plot
    """
    opinion_history = results['opinion_history']
    topic = results.get('topic', 'Unknown Topic')

    # Check if results are partial
    is_partial = results.get('is_partial', False)
    if is_partial:
        completed_timesteps = results.get('completed_timesteps', len(opinion_history))
        total_timesteps = results.get('total_timesteps', len(opinion_history))
        error_msg = results.get('error', 'Unknown error')
        title_suffix = f" (PARTIAL - {completed_timesteps}/{total_timesteps} timesteps)"
    else:
        title_suffix = ""

    timesteps = range(len(opinion_history))

    plt.figure(figsize=(16, 12))
    opinion_array = np.array(opinion_history)
    if opinion_array.ndim == 2 and opinion_array.shape[0] > 0 and opinion_array.shape[1] > 0:
        for i in range(opinion_array.shape[1]):
            plt.plot(timesteps, opinion_array[:, i], label=f'Agent {i}', alpha=0.7, linewidth=1.5)

    # Add vertical line for partial results
    if is_partial and len(opinion_history) > 0:
        plt.axvline(x=len(opinion_history) - 1, color='red', linestyle='--', alpha=0.7,
                    label=f'Simulation stopped (timestep {len(opinion_history)})')

    plt.xlabel('Timestep')
    plt.ylabel('Opinion Value')
    plt.title(f'Individual Agent Opinions for Topic: {topic}{title_suffix}')
    plt.grid(True, alpha=0.3)
    plt.ylim(-1, 1)
    plt.legend(loc='best', ncol=2, fontsize='small')

    if is_partial:
        plt.suptitle(f"⚠️ PARTIAL RESULTS - Simulation interrupted due to: {error_msg}",
                     fontsize=12, color='red', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')



def plot_network_snapshots(results: Dict[str, Any], graph_dir: str, file_prefix: str) -> List[str]:
    """
    Plot the agent connection network for every timestep as separate figures.
    Nodes colored by opinion sign at that timestep: green (>0), red (<0), gray (==0).
    Images are saved under graph_dir with the given file_prefix, one per timestep.
    
    Args:
        results: Simulation results dictionary
        graph_dir: Directory to save graph snapshots (will be created if missing)
        file_prefix: Filename prefix for saved images (without directory)
    
    Returns:
        List of saved file paths
    """
    os.makedirs(graph_dir, exist_ok=True)

    timesteps_data = results.get('timesteps', [])
    if not timesteps_data:
        print("No timestep graph data available; skipping network snapshots.")
        return []

    topic = results.get('topic', 'Unknown Topic')
    is_partial = results.get('is_partial', False)
    if is_partial:
        completed_timesteps = results.get('completed_timesteps', len(timesteps_data))
        total_timesteps = results.get('total_timesteps', len(timesteps_data))
        partial_suffix = f" (PARTIAL - {completed_timesteps}/{total_timesteps} timesteps)"
    else:
        partial_suffix = ""

    # Determine number of agents
    sim_params = results.get('simulation_params', {})
    n_agents = sim_params.get('n_agents')
    if n_agents is None:
        # Fallback: infer from first timestep
        first_agents = timesteps_data[0].get('agents', [])
        n_agents = len(first_agents)

    # Build a fixed layout from the first available timestep
    def build_graph_from_timestep(timestep_index: int) -> nx.Graph:
        agents_info = timesteps_data[timestep_index].get('agents', [])
        G = nx.Graph()
        G.add_nodes_from(range(n_agents))
        added_edges = set()
        for agent_info in agents_info:
            i = agent_info.get('agent_id')
            neighbors = agent_info.get('connected_agents', [])
            for j in neighbors:
                a, b = (i, j) if i <= j else (j, i)
                if (a, b) not in added_edges:
                    added_edges.add((a, b))
                    G.add_edge(a, b)
        return G

    # Opinion-based layout: x = opinion value, y = small per-node jitter
    rng = np.random.default_rng(0)
    y_jitter = rng.normal(0, 0.1, n_agents)

    saved_paths: List[str] = []
    for t in range(len(timesteps_data)):
        agents_info = timesteps_data[t].get('agents', [])
        # Build graph for this timestep
        Gt = nx.Graph()
        Gt.add_nodes_from(range(n_agents))
        added_edges = set()
        for agent_info in agents_info:
            i = agent_info.get('agent_id')
            neighbors = agent_info.get('connected_agents', [])
            for j in neighbors:
                a, b = (i, j) if i <= j else (j, i)
                if (a, b) not in added_edges:
                    added_edges.add((a, b))
                    Gt.add_edge(a, b)

        # Node colors and positions by opinion
        opinions = [0.0] * n_agents
        for agent_info in agents_info:
            i = agent_info.get('agent_id')
            opinions[i] = float(agent_info.get('opinion', 0.0))
        pos = {i: (opinions[i], float(y_jitter[i])) for i in range(n_agents)}
        colors = []
        for val in opinions:
            if val > 0.0:
                colors.append('#2ca02c')  # green
            elif val < 0.0:
                colors.append('#d62728')  # red
            else:
                colors.append('#7f7f7f')  # gray

        plt.figure(figsize=(12, 12))
        nx.draw_networkx_edges(Gt, pos, edge_color='#cccccc', width=1.0, alpha=0.6)
        nx.draw_networkx_nodes(Gt, pos, node_color=colors, node_size=400, edgecolors='black', linewidths=0.5)
        plt.title(f"Network at timestep {t} for Topic: {topic}{partial_suffix}")
        legend_handles = [
            Patch(facecolor='#2ca02c', edgecolor='black', label='Opinion > 0'),
            Patch(facecolor='#d62728', edgecolor='black', label='Opinion < 0'),
            Patch(facecolor='#7f7f7f', edgecolor='black', label='Opinion = 0')
        ]
        plt.legend(handles=legend_handles, loc='best')
        plt.axis('off')
        plt.tight_layout()

        save_path = os.path.join(graph_dir, f"{file_prefix}_timestep_{t:03d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        saved_paths.append(save_path)

    return saved_paths

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
        'final_opinions': results['final_opinions'],
        'simulation_params': results['simulation_params'],
        'random_seed': results['random_seed']
    }
    # Include text histories only if present
    if 'posts_history' in results:
        save_data['posts_history'] = results['posts_history']
    if 'interpretations_history' in results:
        save_data['interpretations_history'] = results['interpretations_history']
    if 'timesteps' in results:
        save_data['timesteps'] = results['timesteps']
    
    # Add partial result information if applicable
    if results.get('is_partial', False):
        save_data['is_partial'] = True
        save_data['interrupted_at_timestep'] = results.get('interrupted_at_timestep')
        save_data['completed_timesteps'] = results.get('completed_timesteps')
        save_data['total_timesteps'] = results.get('total_timesteps')
        save_data['error'] = results.get('error')
    
    # Add metadata if config is provided
    if config is not None:
        save_data['metadata'] = {
            'run_timestamp': datetime.now().isoformat(),
            'config_source': 'config.json',
            'config_metadata': {
                'llm_config': config.get('llm', {}),
                'simulation_config': config.get('simulation', {})
            }
        }
    
    # Save to file
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    


def print_simulation_summary(results: Dict[str, Any]):
    """
    Print a summary of simulation results.
    
    Args:
        results: Simulation results dictionary
    """
    topic = results.get('topic', 'Unknown Topic')
    mean_opinions = results['mean_opinions']
    std_opinions = results['std_opinions']
    posts_history = results.get('posts_history', [])
    interpretations_history = results.get('interpretations_history', [])
    
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
    
    if posts_history and len(posts_history) > 0 and len(posts_history[0]) > 0:
        print(f"\nSample Posts (Timestep 0):")
        for i, post in enumerate(posts_history[0]):
            print(f"  Agent {i}: {post[:100]}...")
    
    if interpretations_history and len(interpretations_history) > 0 and len(interpretations_history[0]) > 0:
        for i, interpretations in enumerate(interpretations_history[0]):
            print(f"  Agent {i}: {[f'{x:.3f}' for x in interpretations[:3]]}...")
    
    print(f"{'='*60}") 


def render_network_video(graph_dir: str, file_prefix: str, fps: int = 2, output_path: str = None) -> str:
    """
    Stitch timestep PNGs into an MP4 video.
    """
    pattern = os.path.join(graph_dir, f"{file_prefix}_timestep_*.png")
    frames = sorted(glob.glob(pattern))
    if not frames:
        return ""
    if output_path is None:
        output_path = os.path.join(graph_dir, f"{file_prefix}.mp4")
    with imageio.get_writer(output_path, fps=fps) as w:
        for fp in frames:
            w.append_data(imageio.imread(fp))
    print(f"Video saved to: {output_path}")
    # Remove frames after video creation
    for fp in frames:
        try:
            os.remove(fp)
        except Exception:
            pass
    print("Frame PNGs removed after video generation")
    return output_path


def render_network_video_from_results(results: Dict[str, Any], graph_dir: str, file_prefix: str, fps: int = 2) -> str:
    """
    Render network snapshots directly to an MP4 without saving PNGs.
    """
    os.makedirs(graph_dir, exist_ok=True)
    timesteps_data = results.get('timesteps', [])
    if not timesteps_data:
        print("No timesteps found in results; skipping video rendering.")
        return ""
    topic = results.get('topic', 'Unknown Topic')
    is_partial = results.get('is_partial', False)
    if is_partial:
        completed_timesteps = results.get('completed_timesteps', len(timesteps_data))
        total_timesteps = results.get('total_timesteps', len(timesteps_data))
        partial_suffix = f" (PARTIAL - {completed_timesteps}/{total_timesteps} timesteps)"
    else:
        partial_suffix = ""
    sim_params = results.get('simulation_params', {})
    n_agents = sim_params.get('n_agents')
    if n_agents is None:
        first_agents = timesteps_data[0].get('agents', [])
        n_agents = len(first_agents)
    def build_graph_from_timestep(ti: int) -> nx.Graph:
        agents_info = timesteps_data[ti].get('agents', [])
        G = nx.Graph()
        G.add_nodes_from(range(n_agents))
        added = set()
        for agent_info in agents_info:
            i = agent_info.get('agent_id')
            for j in agent_info.get('connected_agents', []):
                a, b = (i, j) if i <= j else (j, i)
                if (a, b) not in added:
                    added.add((a, b))
                    G.add_edge(a, b)
        return G
    # Opinion-based layout setup
    rng = np.random.default_rng(0)
    y_jitter = rng.normal(0, 0.1, n_agents)
    output_path = os.path.join(graph_dir, f"{file_prefix}.mp4")
    print(f"Rendering network video with {len(timesteps_data)} frames to: {output_path}")
    with imageio.get_writer(output_path, fps=fps) as w:
        for t in range(len(timesteps_data)):
            agents_info = timesteps_data[t].get('agents', [])
            Gt = nx.Graph(); Gt.add_nodes_from(range(n_agents))
            added = set()
            for agent_info in agents_info:
                i = agent_info.get('agent_id')
                for j in agent_info.get('connected_agents', []):
                    a, b = (i, j) if i <= j else (j, i)
                    if (a, b) not in added:
                        added.add((a, b)); Gt.add_edge(a, b)
            opinions = [0.0] * n_agents
            for agent_info in agents_info:
                opinions[agent_info.get('agent_id')] = float(agent_info.get('opinion', 0.0))
            colors = ['#2ca02c' if v > 0 else ('#d62728' if v < 0 else '#7f7f7f') for v in opinions]
            pos = {i: (opinions[i], float(y_jitter[i])) for i in range(n_agents)}
            fig = plt.figure(figsize=(12, 12), dpi=100)
            FigureCanvas(fig)
            nx.draw_networkx_edges(Gt, pos, edge_color='#cccccc', width=1.0, alpha=0.6)
            nx.draw_networkx_nodes(Gt, pos, node_color=colors, node_size=400, edgecolors='black', linewidths=0.5)
            plt.title(f"Network at timestep {t} for Topic: {topic}{partial_suffix}")
            plt.axis('off')
            plt.tight_layout()
            fig.canvas.draw()
            w_w, w_h = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((w_h, w_w, 3))
            w.append_data(frame)
            plt.close(fig)
    print(f"Video saved to: {output_path}")
    return output_path