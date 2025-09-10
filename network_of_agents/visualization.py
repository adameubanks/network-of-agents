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
import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Patch
import re

def _timesteps_as_list(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return timesteps as a list for both dict- and list-backed results."""
    ts = results.get('timesteps', [])
    if isinstance(ts, dict):
        try:
            return [ts[k] for k in sorted(ts.keys(), key=lambda x: int(x))]
        except Exception:
            return list(ts.values())
    if isinstance(ts, list):
        return ts
    return []


def _derive_opinion_history_from_timesteps(results: Dict[str, Any]) -> List[List[float]]:
    """If opinion_history missing, derive it from timesteps agents' opinions."""
    timesteps_data = _timesteps_as_list(results)
    if not timesteps_data:
        return []
    # Infer number of agents from first timestep
    first_agents = (timesteps_data[0] or {}).get('agents', [])
    n_agents = len(first_agents)
    history: List[List[float]] = []
    for ts in timesteps_data:
        opinions = [0.0] * n_agents
        for a in ts.get('agents', []) or []:
            try:
                idx = int(a.get('agent_id'))
                opinions[idx] = float(a.get('opinion', 0.0))
            except Exception:
                continue
        history.append(opinions)
    return history
    
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
    opinion_history = _derive_opinion_history_from_timesteps(results)
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

    timesteps_data = _timesteps_as_list(results)
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
    # Build lean structure: timesteps as dict, random_seed nested, drop opinion_history
    sim_params = dict(results.get('simulation_params', {}))
    if 'random_seed' in results:
        sim_params['random_seed'] = results['random_seed']
    timesteps_list = results.get('timesteps', [])
    if isinstance(timesteps_list, list):
        ts_dict = {str(ts.get('timestep', i)): ts for i, ts in enumerate(timesteps_list) if isinstance(ts, dict)}
    elif isinstance(timesteps_list, dict):
        ts_dict = {str(k): v for k, v in timesteps_list.items()}
    else:
        ts_dict = {}
    save_data = {
        'topic': results.get('topic', 'Unknown'),
        'final_opinions': results.get('final_opinions', []),
        'timesteps': ts_dict,
        # Move summary fields to bottom after writing file via json.dump order-preserving dict
    }
    # Intentionally avoid duplicating timesteps as a dict; keep JSON lean
    
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
    
    # Append summary stats at the end (mean/std/simulation_params), preserving order
    save_data['mean_opinions'] = results.get('mean_opinions', [])
    save_data['std_opinions'] = results.get('std_opinions', [])
    save_data['simulation_params'] = sim_params

    # Save to file
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    


    


    


def render_network_video_from_results(results: Dict[str, Any], graph_dir: str, file_prefix: str, fps: int = 2) -> str:
    """
    Render network snapshots directly to an MP4 without saving PNGs.
    """
    os.makedirs(graph_dir, exist_ok=True)
    timesteps_data = _timesteps_as_list(results)
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


    


def plot_initial_network_graph(adjacency_matrix: np.ndarray, opinions: np.ndarray, 
                              topic: str, save_path: str = None) -> None:
    """
    Plot the initial network graph at timestep 0 to show sparsity/density.
    
    Args:
        adjacency_matrix: Network adjacency matrix
        opinions: Initial opinion values for each agent
        topic: Topic being discussed
        save_path: Optional path to save the plot
    """
    import networkx as nx
    
    n_agents = adjacency_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n_agents))
    
    # Add edges based on adjacency matrix
    for i in range(n_agents):
        for j in range(i+1, n_agents):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)
    
    # Calculate network statistics
    num_edges = G.number_of_edges()
    max_possible_edges = n_agents * (n_agents - 1) // 2
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    avg_degree = 2 * num_edges / n_agents if n_agents > 0 else 0
    
    # Create layout based on opinions
    pos = {}
    for i in range(n_agents):
        # Position nodes by opinion (x-axis) and add some y-jitter
        y_jitter = np.random.normal(0, 0.1)
        pos[i] = (opinions[i], y_jitter)
    
    # Color nodes by opinion
    colors = []
    for val in opinions:
        if val > 0.0:
            colors.append('#2ca02c')  # green
        elif val < 0.0:
            colors.append('#d62728')  # red
        else:
            colors.append('#7f7f7f')  # gray
    
    plt.figure(figsize=(14, 10))
    
    # Draw the network
    nx.draw_networkx_edges(G, pos, edge_color='#cccccc', width=1.0, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=400, 
                          edgecolors='black', linewidths=0.5)
    
    # Add node labels
    nx.draw_networkx_labels(G, pos, {i: str(i) for i in range(n_agents)}, 
                           font_size=8, font_weight='bold')
    
    # Create legend
    legend_handles = [
        Patch(facecolor='#2ca02c', edgecolor='black', label='Opinion > 0'),
        Patch(facecolor='#d62728', edgecolor='black', label='Opinion < 0'),
        Patch(facecolor='#7f7f7f', edgecolor='black', label='Opinion = 0')
    ]
    plt.legend(handles=legend_handles, loc='upper right')
    
    # Add network statistics as text
    stats_text = f"""Network Statistics:
• Agents: {n_agents}
• Edges: {num_edges}
• Density: {density:.3f}
• Avg Degree: {avg_degree:.1f}
• Max Possible Edges: {max_possible_edges}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.title(f"Initial Network Graph - Topic: {topic}\n"
              f"Network Density: {density:.1%} | Average Degree: {avg_degree:.1f}")
    plt.xlabel("Opinion Value (-1 to 1)")
    plt.ylabel("Random Y-offset")
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Initial network graph saved to: {save_path}")
    else:
        plt.show()


def find_reply_edges(results: Dict[str, Any]) -> List[Dict[str, int]]:
    """
    Return explicit reply edges per timestep if present; otherwise, infer from mentions.
    Each item: {timestep, source, target}
    """
    timesteps_data = _timesteps_as_list(results)
    if not timesteps_data:
        return []
    edges: List[Dict[str, int]] = []
    for t_idx, ts in enumerate(timesteps_data):
        explicit = ts.get('reply_edges')
        if isinstance(explicit, list) and explicit:
            for e in explicit:
                try:
                    edges.append({'timestep': int(t_idx), 'source': int(e.get('source')), 'target': int(e.get('target'))})
                except Exception:
                    continue
        else:
            # Fallback: infer from mentions this timestep
            agents_info = ts.get('agents', []) or []
            pattern = re.compile(r"\bAgent\s+(\d+)\b", flags=re.IGNORECASE)
            for a in agents_info:
                try:
                    i = int(a.get('agent_id'))
                except Exception:
                    continue
                post = str(a.get('post', '') or '')
                for m in pattern.findall(post):
                    try:
                        j = int(m)
                        if j != i:
                            edges.append({'timestep': int(t_idx), 'source': i, 'target': j})
                    except Exception:
                        continue
    return edges


def print_reply_edges(results: Dict[str, Any]) -> None:
    """Print a concise list of reply edges per timestep."""
    edges = find_reply_edges(results)
    if not edges:
        print("No reply edges found.")
        return
    for e in edges:
        print(f"t={e['timestep']} | Agent {e['source']} -> Agent {e['target']}")

    