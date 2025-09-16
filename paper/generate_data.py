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
        
        # Calculate L2 norm error for fixed-point error (scaled appropriately)
        l2_error = abs(final_mean - degroot_final) * np.sqrt(50)  # Scale by sqrt(n) for proper L2 norm
        
        metrics[topic] = {
            'mean_trajectory': mean_opinions,
            'std_trajectory': std_opinions,
            'final_mean': final_mean,
            'final_std': final_std,
            'bias': bias,
            'error': error,
            'l2_error': l2_error,
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
        f.write("\\begin{tabular}{p{2.0cm}cccc}\n")
        f.write("\\toprule\n")
        f.write("Topic & Bias & Error & LLM Mean & DeGroot \\\\\n")
        f.write("\\midrule\n")
        
        for topic, data in metrics.items():
            topic_display = topic.replace("_", " ").title()
            f.write(f"{topic_display} & {data['bias']:.2f} & {data['l2_error']:.2f} & {data['final_mean']:.2f} & {data['degroot_final']:.2f} \\\\\n")
        
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
                f"\\expandafter\\newcommand\\csname {topic_var}Error\\endcsname"
                f"{{{data['l2_error']:.4f}}}\n"
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

        # Summary macros
        f.write(f"\\newcommand{{\\cmBiasFiveNano}}{{{avg('bias', 'five_nano')}}}\n")
        f.write(f"\\newcommand{{\\cmErrorFiveNano}}{{{avg('l2_error', 'five_nano')}}}\n")
        f.write(f"\\newcommand{{\\cmSymFiveNano}}{{--}}\n")

        f.write(f"\\newcommand{{\\cmBiasFiveMini}}{{{avg('bias', 'five_mini')}}}\n")
        f.write(f"\\newcommand{{\\cmErrorFiveMini}}{{{avg('l2_error', 'five_mini')}}}\n")
        f.write(f"\\newcommand{{\\cmSymFiveMini}}{{--}}\n")

        f.write(f"\\newcommand{{\\cmBiasGrokMini}}{{{avg('bias', 'grok_mini')}}}\n")
        f.write(f"\\newcommand{{\\cmErrorGrokMini}}{{{avg('l2_error', 'grok_mini')}}}\n")
        f.write(f"\\newcommand{{\\cmSymGrokMini}}{{--}}\n\n")

        # Cross-model table macro
        f.write("\\newcommand{\\crossmodeltable}{\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Bias & Error & Symmetry & Notes \\\\n")
        f.write("\\midrule\n")
        f.write("\\modelFiveNano{} (this work) & \\cmBiasFiveNano & \\cmErrorFiveNano & fail & baseline \\\\n")
        f.write("\\modelFiveMini{} (pending) & \\cmBiasFiveMini & \\cmErrorFiveMini & \\cmSymFiveMini & pending \\\\n")
        f.write("\\modelGrokMini{} (pending) & \\cmBiasGrokMini & \\cmErrorGrokMini & \\cmSymGrokMini & pending \\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n\n")

        # Per-topic cross-model table macro (Bias values) - self-contained table*
        f.write("\\newcommand{\\perTopicCrossModelTable}{\n")
        f.write("\\begin{table*}[ht]\\centering\\small\n")
        f.write("\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X c c c}\n")
        f.write("\\toprule\n")
        f.write("Topic & \\modelFiveNano{} & \\modelFiveMini{} & \\modelGrokMini{} \\\\n")
        f.write("\\midrule\n")
        for topic in topics_order:
            topic_display = pretty_labels.get(topic, topic.replace("_", " ").title())
            def cell(model_id: str) -> str:
                m = metrics_by_model.get(model_id, {})
                if topic in m:
                    return f"{m[topic]['bias']:.2f}"
                return "--"
            f.write(f"{topic_display} & {cell('five_nano')} & {cell('five_mini')} & {cell('grok_mini')} \\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabularx}\n")
        f.write("\\caption{Per-topic bias by model (placeholders for \\modelFiveMini{} and \\modelGrokMini{}).}\n")
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
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average bias: {np.mean([m['bias'] for m in metrics.values()]):.4f}")
    print(f"Average error: {np.mean([m['error'] for m in metrics.values()]):.4f}")
    print(f"Topics with high bias (>0.2): {sum(1 for m in metrics.values() if abs(m['bias']) > 0.2)}")
    print(f"Topics with high error (>4.0): {sum(1 for m in metrics.values() if m['error'] > 4.0)}")
    
    print(f"\nData and visualizations saved to {output_dir}")
    print("LaTeX data files saved to data.tex and data_models.tex")

if __name__ == "__main__":
    main()
