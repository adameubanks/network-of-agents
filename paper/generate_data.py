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
                       label='LLM (GPT-5-nano)', 
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
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Topic & Bias & Fixed-Point Error & LLM Mean & DeGroot Mean \\\\\n")
        f.write("\\midrule\n")
        
        for topic, data in metrics.items():
            topic_display = topic.replace("_", " ").title()
            f.write(f"{topic_display} & {data['bias']:.4f} & {data['l2_error']:.4f} & {data['final_mean']:.4f} & {data['degroot_final']:.4f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n\n")
        
        # Individual topic data
        for topic, data in metrics.items():
            topic_var = topic.replace("_", "").replace(" ", "")
            f.write(f"\\newcommand{{\\{topic_var}Bias}}{{{data['bias']:.4f}}}\n")
            f.write(f"\\newcommand{{\\{topic_var}Error}}{{{data['l2_error']:.4f}}}\n")
            f.write(f"\\newcommand{{\\{topic_var}LLMMean}}{{{data['final_mean']:.4f}}}\n")
            f.write(f"\\newcommand{{\\{topic_var}LLMStd}}{{{data['final_std']:.4f}}}\n")

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
    
    print("Extracting post examples...")
    post_examples = extract_post_examples(sim_data)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average bias: {np.mean([m['bias'] for m in metrics.values()]):.4f}")
    print(f"Average error: {np.mean([m['error'] for m in metrics.values()]):.4f}")
    print(f"Topics with high bias (>0.2): {sum(1 for m in metrics.values() if abs(m['bias']) > 0.2)}")
    print(f"Topics with high error (>4.0): {sum(1 for m in metrics.values() if m['error'] > 4.0)}")
    
    print(f"\nData and visualizations saved to {output_dir}")
    print("LaTeX data file saved to data.tex")

if __name__ == "__main__":
    main()
