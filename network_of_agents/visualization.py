"""
Visualization module for opinion dynamics simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import json
import os
from datetime import datetime
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
                # Try both 'opinion' and 'actual_opinion' fields
                opinion_value = a.get('opinion', a.get('actual_opinion', 0.0))
                opinions[idx] = float(opinion_value)
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
    timesteps = range(len(mean_opinions))

    plt.figure(figsize=(16, 12))
    plt.plot(timesteps, mean_opinions, 'b-', linewidth=2, label='Mean Opinion')
    plt.fill_between(timesteps,
                     [m - s for m, s in zip(mean_opinions, std_opinions)],
                     [m + s for m, s in zip(mean_opinions, std_opinions)],
                     alpha=0.3, color='blue', label='±1 Std Dev')

    plt.xlabel('Timestep')
    plt.ylabel('Opinion Value')
    plt.title(f'Mean Opinion for Topic: {topic}')
    plt.grid(True, alpha=0.3)
    plt.ylim(-1, 1)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_individual_opinions(results: Dict[str, Any], save_path: str = None):
    """
    Plot individual agent opinions as a single figure.
    
    Args:
        results: Simulation results dictionary
        save_path: Optional path to save the plot
    """
    opinion_history = _derive_opinion_history_from_timesteps(results)
    topic = results.get('topic', 'Unknown Topic')
    timesteps = range(len(opinion_history))

    plt.figure(figsize=(16, 12))
    opinion_array = np.array(opinion_history)
    if opinion_array.ndim == 2 and opinion_array.shape[0] > 0 and opinion_array.shape[1] > 0:
        for i in range(opinion_array.shape[1]):
            plt.plot(timesteps, opinion_array[:, i], label=f'Agent {i}', alpha=0.7, linewidth=1.5)

    plt.xlabel('Timestep')
    plt.ylabel('Opinion Value')
    plt.title(f'Individual Agent Opinions for Topic: {topic}')
    plt.grid(True, alpha=0.3)
    plt.ylim(-1, 1)
    plt.legend(loc='best', ncol=2, fontsize='small')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


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
    }
    
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

def generate_graphs_for_topic(topic_dir: str, json_filename: str = None) -> None:
    """
    Generate all visualization graphs for a topic directory.
    
    Args:
        topic_dir: Directory containing the JSON results file
        json_filename: Optional specific JSON filename (if None, finds the first .json file)
    """
    import glob
    
    if json_filename is None:
        json_files = glob.glob(os.path.join(topic_dir, "*.json"))
        if not json_files:
            print(f"No JSON files found in {topic_dir}")
            return
        json_filename = json_files[0]
    
    json_path = os.path.join(topic_dir, json_filename)
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return
    
    print(f"Loading data from: {json_path}")
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Extract topic name for file naming
    topic_name = results.get('topic', 'unknown_topic')
    safe_topic_name = re.sub(r'[^\w\s-]', '', topic_name).strip()
    safe_topic_name = re.sub(r'[-\s]+', '_', safe_topic_name)
    
    # Generate mean/std plot
    mean_std_path = os.path.join(topic_dir, f"{safe_topic_name}_mean_std.png")
    print(f"Generating mean/std plot: {mean_std_path}")
    plot_mean_std(results, mean_std_path)
    
    # Generate individual opinions plot
    individuals_path = os.path.join(topic_dir, f"{safe_topic_name}_individuals.png")
    print(f"Generating individual opinions plot: {individuals_path}")
    plot_individual_opinions(results, individuals_path)
    
    print(f"Graph generation complete for topic: {topic_name}")

def generate_all_company_statements_graphs(results_dir: str = "/home/adam/Projects/IDeA/network-of-agents/results") -> None:
    """Generate graphs for all company_statements topics."""
    company_dirs = [
        "company_statements_on_social_issues_are_important_vs_company_statements_on_social_issues_are_not_important",
        "company_statements_on_social_issues_are_not_important_vs_company_statements_on_social_issues_are_important"
    ]
    
    for topic_dir_name in company_dirs:
        topic_dir = os.path.join(results_dir, topic_dir_name)
        if os.path.exists(topic_dir):
            print(f"\n{'='*60}")
            print(f"Processing: {topic_dir_name}")
            print(f"{'='*60}")
            generate_graphs_for_topic(topic_dir)
        else:
            print(f"Directory not found: {topic_dir}")