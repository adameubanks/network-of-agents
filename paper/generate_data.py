#!/usr/bin/env python3
"""
Script to analyze simulation results and generate LaTeX data files and visualizations
for the algorithmic fidelity paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import re
from .calibration_metrics import (
    summarize_calibration,
)

def load_simulation_data(results_dir: str) -> Dict:
    """Load all simulation data from JSON files."""
    data = {}
    
    # Load DeGroot results
    degroot_dir = Path(results_dir) / "degroot"
    for topic_dir in degroot_dir.iterdir():
        if topic_dir.is_dir() and topic_dir.name != "test_topic":
            for subtopic_dir in topic_dir.iterdir():
                if subtopic_dir.is_dir():
                    for json_file in subtopic_dir.glob("*.json"):
                        if "gpt-5-nano" in json_file.name:
                            topic_name = json_file.stem.split("_50_50_gpt-5-nano")[0]
                            with open(json_file, 'r') as f:
                                data[topic_name] = json.load(f)
    
    return data

def load_simulation_data_by_model(results_dir: str, model_patterns: Dict[str, str]) -> Dict[str, Dict]:
    """
    Load simulation data grouped by normalized model id.

    model_patterns maps filename substrings to normalized ids, e.g.,
    {"gpt-5-nano": "five_nano", "gpt-5-mini": "five_mini", "grok-mini": "grok_mini"}.
    """
    data_by_model: Dict[str, Dict] = {norm: {} for norm in model_patterns.values()}
    degroot_dir = Path(results_dir) / "degroot"
    if not degroot_dir.exists():
        return data_by_model
    for topic_dir in degroot_dir.iterdir():
        if not topic_dir.is_dir() or topic_dir.name == "test_topic":
            continue
        for subtopic_dir in topic_dir.iterdir():
            if not subtopic_dir.is_dir():
                continue
            for json_file in subtopic_dir.glob("*.json"):
                name = json_file.name.lower()
                matched = None
                for pattern, norm in model_patterns.items():
                    if pattern in name:
                        matched = norm
                        break
                if matched is None:
                    continue
                stem = json_file.stem
                parts = stem.split("_50_50_")
                topic_key = parts[0] if len(parts) > 1 else stem
                with open(json_file, 'r') as f:
                    data_by_model[matched][topic_key] = json.load(f)
    return data_by_model

def extract_trajectory_data(sim_data: Dict) -> Dict:
    """Extract trajectory data for plotting."""
    trajectories = {}
    
    for topic, data in sim_data.items():
        if 'timesteps' in data:
            # Extract opinion trajectories from timesteps
            timestep_data = data['timesteps']
            num_timesteps = len(timestep_data)
            num_agents = len(data['final_opinions'])
            
            # Initialize trajectory array
            traj = np.zeros((num_timesteps, num_agents))
            
            for t, timestep_key in enumerate(sorted(timestep_data.keys(), key=int)):
                timestep = timestep_data[timestep_key]
                for agent in timestep['agents']:
                    agent_id = agent['agent_id']
                    opinion = agent['opinion']
                    traj[t, agent_id] = opinion
            
            trajectories[topic] = traj
        elif 'final_opinions' in data:
            # If only final opinions available, create a simple trajectory
            final_opinions = np.array(data['final_opinions'])
            # Create a simple trajectory (this is a placeholder - real data should have full trajectories)
            trajectories[topic] = np.array([final_opinions] * 50)  # 50 timesteps
    
    return trajectories


def extract_calibration_pairs(sim_data: Dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Extract (o, r) pairs per topic for calibration.

    o is the intended opinion in [0,1] used to generate a post;
    r is the LLM's inferred opinion (rating) in [0,1].

    We align each timestep's inferred_opinion with the previous timestep's opinion,
    because storage occurs after the opinion update.
    """
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for topic, data in sim_data.items():
        timesteps = data.get('timesteps')
        if not isinstance(timesteps, dict):
            # Backward compatibility: skip if no detailed timesteps
            continue
        # Build sorted list of timestep entries
        sorted_keys = sorted((int(k) for k in timesteps.keys()))
        if len(sorted_keys) < 2:
            continue
        o_list: list = []
        r_list: list = []
        for idx in range(1, len(sorted_keys)):
            prev_ts = timesteps.get(str(sorted_keys[idx - 1]), {})
            cur_ts = timesteps.get(str(sorted_keys[idx]), {})
            prev_agents = prev_ts.get('agents', [])
            cur_agents = cur_ts.get('agents', [])
            if not prev_agents or not cur_agents:
                continue
            n = min(len(prev_agents), len(cur_agents))
            for i in range(n):
                try:
                    # Opinions and inferred_opinions are in [-1,1] in agent domain
                    o_agent = float(prev_agents[i].get('opinion'))
                    r_agent = float(cur_agents[i].get('inferred_opinion'))
                except Exception:
                    continue
                # Require both numbers finite
                if not (np.isfinite(o_agent) and np.isfinite(r_agent)):
                    continue
                # Map to [0,1]
                o_math = (o_agent + 1.0) / 2.0
                r_math = (r_agent + 1.0) / 2.0
                # Clip to [0,1]
                o_math = float(np.clip(o_math, 0.0, 1.0))
                r_math = float(np.clip(r_math, 0.0, 1.0))
                o_list.append(o_math)
                r_list.append(r_math)
        if o_list and r_list:
            pairs[topic] = (np.array(o_list, dtype=float), np.array(r_list, dtype=float))
    return pairs


def compute_calibration_summary(sim_data: Dict) -> pd.DataFrame:
    """Compute calibration summaries per topic as a DataFrame."""
    pairs = extract_calibration_pairs(sim_data)
    rows = []
    for topic, (o, r) in pairs.items():
        s = summarize_calibration(o, r)
        s_row = {
            'topic': topic,
            'alpha_c': s.get('alpha_c'),
            'beta_c': s.get('beta_c'),
            'r2_centered': s.get('r2_centered'),
            'ece_bins_uniform': s.get('ece_bins_uniform'),
            'ece_bins_quantile': s.get('ece_bins_quantile'),
            'ece_isotonic': s.get('ece_isotonic'),
            'rmse_calib': s.get('rmse_calib'),
            'css': s.get('css'),
            'contraction_coeff': s.get('contraction_coeff'),
            'monotonicity_violation_rate': s.get('monotonicity_violation_rate'),
            'flip_mean_abs': s.get('flip_mean_abs'),
            'flip_max_abs': s.get('flip_max_abs'),
            'n_pairs': int(len(o)),
        }
        rows.append(s_row)
    return pd.DataFrame(rows)

def calculate_metrics(trajectories: Dict) -> Dict:
    """Calculate bias and error metrics for each topic."""
    metrics = {}
    
    # Load the pure DeGroot baseline trajectory from test_topic
    test_topic_file = "/home/adam/Projects/IDeA/network-of-agents/results/degroot/test_topic/test_topic_50_50_no-llm_20250910_205014.json"
    try:
        with open(test_topic_file, 'r') as f:
            test_data = json.load(f)
        
        # Extract full trajectory from test_topic
        timestep_data = test_data['timesteps']
        num_timesteps = len(timestep_data)
        num_agents = len(test_data['final_opinions'])
        
        # Initialize trajectory array
        degroot_traj = np.zeros((num_timesteps, num_agents))
        
        for t, timestep_key in enumerate(sorted(timestep_data.keys(), key=int)):
            timestep = timestep_data[timestep_key]
            for agent in timestep['agents']:
                agent_id = agent['agent_id']
                opinion = agent['opinion']
                degroot_traj[t, agent_id] = opinion
        
        # Calculate mean and std over time for pure DeGroot
        degroot_mean_traj = np.mean(degroot_traj, axis=1)
        degroot_std_traj = np.std(degroot_traj, axis=1)
        degroot_final = degroot_mean_traj[-1]
        
        print(f"DeGroot baseline (test_topic): final={degroot_final:.6f}, timesteps={num_timesteps}")
    except Exception as e:
        print(f"Error loading DeGroot baseline: {e}")
        degroot_final = -0.1166  # Fallback value
        degroot_mean_traj = np.array([-0.1166] * 50)  # Fallback trajectory
        degroot_std_traj = np.array([0.0] * 50)
    
    for topic, traj in trajectories.items():
        # Calculate mean opinion over time
        mean_opinions = np.mean(traj, axis=1)
        std_opinions = np.std(traj, axis=1)
        
        # Final values
        final_mean = mean_opinions[-1]
        final_std = std_opinions[-1]
        
        # Calculate bias and error
        bias = final_mean - degroot_final
        error = abs(final_mean - degroot_final)
        
        # Core divergence metrics
        mae = float(np.mean(np.abs(mean_opinions - degroot_mean_traj))) if len(mean_opinions) == len(degroot_mean_traj) else float(abs(final_mean - degroot_final))
        rmse = float(np.sqrt(np.mean((mean_opinions - degroot_mean_traj) ** 2))) if len(mean_opinions) == len(degroot_mean_traj) else float(abs(final_mean - degroot_final))
        # Correlation of trajectories (handle degenerate cases)
        if len(mean_opinions) == len(degroot_mean_traj) and np.std(mean_opinions) > 0 and np.std(degroot_mean_traj) > 0:
            corr = float(np.corrcoef(mean_opinions, degroot_mean_traj)[0, 1])
        else:
            corr = np.nan
        
        metrics[topic] = {
            'mean_trajectory': mean_opinions,
            'std_trajectory': std_opinions,
            'final_mean': final_mean,
            'final_std': final_std,
            'bias': bias,
            'error': error,
            'mae': mae,
            'rmse': rmse,
            'corr': corr,
            'degroot_final': degroot_final,
            'degroot_mean_trajectory': degroot_mean_traj,
            'degroot_std_trajectory': degroot_std_traj
        }
    
    return metrics

def create_trajectory_plots(metrics: Dict, output_dir: str):
    """Create trajectory plots comparing LLM vs DeGroot."""
    
    # Key topics to highlight
    highlight_topics = [
        'israel_vs_palestine',
        'palestine_vs_israel', 
        'lebron_james_is_the_goat_vs_michael_jordan_is_the_goat',
        'michael_jordan_is_the_goat_vs_lebron_james_is_the_goat',
        'chocolate_ice_cream_vs_vanilla_ice_cream',
        'vanilla_ice_cream_vs_chocolate_ice_cream',
        'hanging_a_pride_flag_in_the_classroom_vs_banning_the_pride_flag_from_being_hung_in_the_classroom',
        'banning_the_pride_flag_from_being_hung_in_the_classroom_vs_hanging_a_pride_flag_in_the_classroom',
        'hanging_the_10_commandments_in_the_classroom_vs_banning_the_10_commandments_from_being_hung_in_the_classroom',
        'banning_the_10_commandments_from_being_hung_in_the_classroom_vs_hanging_the_10_commandments_in_the_classroom'
    ]
    
    for topic in highlight_topics:
        if topic in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data = metrics[topic]
            timesteps = np.arange(len(data['mean_trajectory']))
            
            # Plot LLM trajectory with error bars
            ax.errorbar(timesteps, data['mean_trajectory'], 
                       yerr=data['std_trajectory'], 
                       label='LLM (gpt-5-nano)', 
                       alpha=0.7, 
                       capsize=3)
            
            # Plot DeGroot baseline trajectory
            degroot_timesteps = np.arange(len(data['degroot_mean_trajectory']))
            ax.plot(degroot_timesteps, data['degroot_mean_trajectory'], 
                   color='red', 
                   linestyle='--', 
                   linewidth=2,
                   label='Pure DeGroot')
            
            # Add DeGroot error bars if std > 0
            if np.any(data['degroot_std_trajectory'] > 0):
                ax.fill_between(degroot_timesteps, 
                               data['degroot_mean_trajectory'] - data['degroot_std_trajectory'],
                               data['degroot_mean_trajectory'] + data['degroot_std_trajectory'],
                               color='red', alpha=0.2, label='DeGroot ±1σ')
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Average Opinion')
            ax.set_title(f'Opinion Trajectory: {topic.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add bias annotation
            ax.text(0.02, 0.98, f'Bias: {data["bias"]:.3f}\nError: {data["error"]:.3f}', 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/trajectory_{topic}.png', dpi=300, bbox_inches='tight')
            plt.close()

def generate_latex_data(metrics: Dict, output_file: str):
    """Generate LaTeX data file with all metrics."""
    
    with open(output_file, 'w') as f:
        f.write("% Generated data for algorithmic fidelity paper\n")
        f.write("% This file is automatically generated by generate_data.py\n\n")
        
        f.write("\\newcommand{\\numtopics}{" + str(len(metrics)) + "}\n\n")
        
        # Create table data
        f.write("\\newcommand{\\resultstable}{\n")
        f.write("\\footnotesize\n")
        f.write("\\begin{tabular}{p{2.0cm}cccccc}\n")
        f.write("\\toprule\n")
        f.write("Topic & Bias & MAE & RMSE & Corr & LLM Mean & DeGroot \\\n")
        f.write("\\midrule\n")
        
        for topic, data in metrics.items():
            topic_display = topic.replace("_", " ").title()
            f.write(f"{topic_display} & {data['bias']:.2f} & {data.get('mae', np.nan):.2f} & {data.get('rmse', np.nan):.2f} & {data.get('corr', np.nan):.2f} & {data['final_mean']:.2f} & {data['degroot_final']:.2f} \\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n\n")
        
        # Individual topic data (use csname to allow digits in names)
        for topic, data in metrics.items():
            topic_var = topic.replace("_", "").replace(" ", "")
            f.write(
                f"\\expandafter\\newcommand\\csname {topic_var}Bias\\endcsname"
                f"{{{data['bias']:.4f}}}\n"
            )
            f.write(
                f"\\expandafter\\newcommand\\csname {topic_var}MAE\\endcsname"
                f"{{{data.get('mae', np.nan):.4f}}}\n"
            )
            f.write(
                f"\\expandafter\\newcommand\\csname {topic_var}RMSE\\endcsname"
                f"{{{data.get('rmse', np.nan):.4f}}}\n"
            )
            f.write(
                f"\\expandafter\\newcommand\\csname {topic_var}Corr\\endcsname"
                f"{{{data.get('corr', np.nan):.4f}}}\n"
            )
            f.write(
                f"\\expandafter\\newcommand\\csname {topic_var}LLMMean\\endcsname"
                f"{{{data['final_mean']:.4f}}}\n"
            )
            f.write(
                f"\\expandafter\\newcommand\\csname {topic_var}LLMStd\\endcsname"
                f"{{{data['final_std']:.4f}}}\n"
            )

def generate_cross_model_latex(metrics_by_model: Dict[str, Dict], output_file: str, topics_order: List[str]):
    """Generate LaTeX macros for cross-model tables."""
    def avg(metric_key: str, model_id: str) -> str:
        m = metrics_by_model.get(model_id, {})
        if not m:
            return "--"
        vals = [v[metric_key] for v in m.values() if metric_key in v]
        return (f"{np.mean(vals):.2f}" if vals else "--")

    # Human-readable short labels to prevent overfull lines
    pretty_labels = {
        'banning_the_10_commandments_from_being_hung_in_the_classroom_vs_hanging_the_10_commandments_in_the_classroom': 'Ban 10 Commandments vs Hang 10 Commandments',
        'hanging_the_10_commandments_in_the_classroom_vs_banning_the_10_commandments_from_being_hung_in_the_classroom': 'Hang 10 Commandments vs Ban 10 Commandments',
        'banning_the_pride_flag_from_being_hung_in_the_classroom_vs_hanging_a_pride_flag_in_the_classroom': 'Ban Pride Flag vs Hang Pride Flag',
        'hanging_a_pride_flag_in_the_classroom_vs_banning_the_pride_flag_from_being_hung_in_the_classroom': 'Hang Pride Flag vs Ban Pride Flag',
        'chocolate_ice_cream_vs_vanilla_ice_cream': 'Chocolate vs Vanilla',
        'vanilla_ice_cream_vs_chocolate_ice_cream': 'Vanilla vs Chocolate',
        'conservatives_vs_liberals': 'Conservatives vs Liberals',
        'liberals_vs_conservatives': 'Liberals vs Conservatives',
        'israel_vs_palestine': 'Israel vs Palestine',
        'palestine_vs_israel': 'Palestine vs Israel',
        'lebron_james_is_the_goat_vs_michael_jordan_is_the_goat': 'LeBron vs Jordan',
        'michael_jordan_is_the_goat_vs_lebron_james_is_the_goat': 'Jordan vs LeBron',
        'triangles_vs_circles': 'Triangles vs Circles',
        'circles_vs_triangles': 'Circles vs Triangles',
    }

    with open(output_file, 'w') as f:
        f.write("% Generated cross-model data for algorithmic fidelity paper\n")
        f.write("% This file is automatically generated by generate_data.py\n\n")

        # Dynamic cross-model summary table (only models with data)
        present_order = ['five_nano', 'grok_mini', 'five_mini']
        present = [mid for mid in present_order if metrics_by_model.get(mid)]
        name_map = {
            'five_nano': '\\modelFiveNano{}',
            'grok_mini': '\\modelGrokMini{}',
            'five_mini': '\\modelFiveMini{}',
        }

        f.write("\\newcommand{\\crossmodeltable}{\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Bias & RMSE & Symmetry & Notes \\\\\n")
        f.write("\\midrule\n")
        for mid in present:
            label = name_map.get(mid, mid)
            bias = avg('bias', mid)
            rmse = avg('rmse', mid)
            sym = ('fail' if mid == 'five_nano' else '--')
            notes = ('baseline' if mid == 'five_nano' else '--')
            f.write(f"{label} & {bias} & {rmse} & {sym} & {notes} \\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n\n")

        # Per-topic cross-model table (Bias values), dynamic columns
        f.write("\\newcommand{\\perTopicCrossModelTable}{\n")
        f.write("\\begin{table*}[ht]\\centering\\small\n")
        colspec = ' '.join(['c'] * len(present))
        f.write(f"\\begin{{tabularx}}{{\\textwidth}}{{>{{\\raggedright\\arraybackslash}}X {colspec}}}\n")
        f.write("\\toprule\n")
        header_models = ' & '.join([name_map.get(mid, mid) for mid in present])
        f.write(f"Topic & {header_models} \\\\n")
        f.write("\\midrule\n")
        for topic in topics_order:
            topic_display = pretty_labels.get(topic, topic.replace("_", " ").title())
            cells = []
            for mid in present:
                m = metrics_by_model.get(mid, {})
                cells.append(f"{m[topic]['bias']:.2f}" if topic in m else "--")
            f.write(f"{topic_display} & {' & '.join(cells)} \\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabularx}\n")
        f.write("\\caption{Per-topic bias by model.}\n")
        f.write("\\label{tab:cross_model_per_topic}\n")
        f.write("\\end{table*}\n")
        f.write("}\n")

def extract_post_examples(sim_data: Dict) -> Dict:
    """Extract example posts to analyze persona differences."""
    post_examples = {}
    
    for topic, data in sim_data.items():
        if 'posts' in data:
            # Extract some example posts
            posts = data['posts']
            if isinstance(posts, list) and len(posts) > 0:
                # Get posts from different timesteps
                example_posts = []
                for i in range(0, min(len(posts), 10), 2):  # Sample every other post
                    if i < len(posts):
                        example_posts.append(posts[i])
                post_examples[topic] = example_posts
    
    return post_examples

def calculate_metrics_by_model(trajectories_by_model: Dict[str, Dict]) -> Dict[str, Dict]:
    """Calculate metrics for each model and topic."""
    metrics_by_model: Dict[str, Dict] = {}
    for model_id, trajs in trajectories_by_model.items():
        metrics_by_model[model_id] = calculate_metrics(trajs)
    return metrics_by_model

def main():
    """Main function to generate all data and visualizations."""
    
    # Set up paths
    results_dir = "/home/adam/Projects/IDeA/network-of-agents/results"
    output_dir = "/home/adam/Projects/IDeA/network-of-agents/paper/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading simulation data...")
    sim_data = load_simulation_data(results_dir)
    print(f"Loaded data for {len(sim_data)} topics")
    
    print("Extracting trajectory data...")
    trajectories = extract_trajectory_data(sim_data)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(trajectories)
    
    print("Creating trajectory plots...")
    create_trajectory_plots(metrics, output_dir)
    
    print("Generating LaTeX data file...")
    generate_latex_data(metrics, "/home/adam/Projects/IDeA/network-of-agents/paper/data.tex")
    
    # Cross-model generation (summary + per-topic bias table)
    print("Generating cross-model LaTeX data...")
    model_patterns = {"gpt-5-nano": "five_nano", "gpt-5-mini": "five_mini", "grok-mini": "grok_mini"}
    sim_by_model = load_simulation_data_by_model(results_dir, model_patterns)
    traj_by_model = {mid: extract_trajectory_data(d) for mid, d in sim_by_model.items()}
    metrics_by_model = calculate_metrics_by_model(traj_by_model)
    topics_order = sorted({t for d in sim_by_model.values() for t in d.keys()})
    generate_cross_model_latex(metrics_by_model, "/home/adam/Projects/IDeA/network-of-agents/paper/data_models.tex", topics_order)
    
    print("Extracting post examples...")
    post_examples = extract_post_examples(sim_data)

    # Calibration computation and figures
    print("Computing calibration summaries...")
    calib_df = compute_calibration_summary(sim_data)
    calib_csv = "/home/adam/Projects/IDeA/network-of-agents/paper/figures/calibration_summary.csv"
    calib_df.to_csv(calib_csv, index=False)
    print(f"Saved calibration summary to {calib_csv}")

    # Nonlinear reliability diagrams removed
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average bias: {np.mean([m['bias'] for m in metrics.values()]):.4f}")
    print(f"Average MAE: {np.nanmean([m.get('mae', np.nan) for m in metrics.values()]):.4f}")
    print(f"Topics with high bias (>0.2): {sum(1 for m in metrics.values() if abs(m['bias']) > 0.2)}")
    print(f"Topics with high RMSE (>0.2): {sum(1 for m in metrics.values() if m.get('rmse', np.nan) > 0.2)}")
    if not calib_df.empty:
        print(f"\nCalibration (aggregate): alpha_c mean={calib_df['alpha_c'].mean():.3f}, beta_c mean={calib_df['beta_c'].mean():.3f}, RMSE mean={calib_df['rmse_calib'].mean():.3f}")
    
    print(f"\nData and visualizations saved to {output_dir}")
    print("LaTeX data files saved to data.tex and data_models.tex")

if __name__ == "__main__":
    main()
