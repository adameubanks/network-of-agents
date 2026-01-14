#!/usr/bin/env python3
"""
Create a summary figure showing RMSE across topics for different models/framings.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def normalize_topic_name(topic):
    """Normalize topic names to match across experiments."""
    topic_map = {
        'democracy': 'social_media_democracy',
        'weddings': 'child_free_weddings',
        'etiquette': 'restaurant_etiquette',
        'sandwich': 'hot_dog_sandwich',
        'economy': 'environment_economy',
        'safety': 'gun_safety',
        'activism': 'corporate_activism',
        'paper': 'toilet_paper',
        'cloning': 'human_cloning'
    }
    t_lower = topic.lower().replace(' ', '_')
    return topic_map.get(t_lower, t_lower)

def create_rmse_comparison_figure(data, output_path):
    """Create a bar chart comparing final difference from DeGroot baseline across topics for different conditions.
    Shows positive vs negative convergence direction."""
    
    # Group data by experiment and topic
    by_exp = defaultdict(dict)
    for r in data:
        exp = r['experiment']
        topic = normalize_topic_name(r['topic'])
        by_exp[exp][topic] = r
    
    # Get all topics
    all_topics = set()
    for exp_data in by_exp.values():
        all_topics.update(exp_data.keys())
    all_topics = sorted(all_topics)
    
    # Prepare data for plotting - use final_diff instead of rmse
    topics_display = [t.replace('_', ' ').title() for t in all_topics]
    nano_a_diff = []
    nano_b_diff = []
    mini_a_diff = []
    mini_b_diff = []
    
    for topic in all_topics:
        # Use None for missing data instead of 0
        nano_a_entry = by_exp.get('nano-a', {}).get(topic)
        nano_b_entry = by_exp.get('nano-b', {}).get(topic)
        mini_a_entry = by_exp.get('mini', {}).get(topic)
        mini_b_entry = by_exp.get('mini-b', {}).get(topic)
        
        nano_a_diff.append(nano_a_entry.get('final_diff') if nano_a_entry else None)
        nano_b_diff.append(nano_b_entry.get('final_diff') if nano_b_entry else None)
        mini_a_diff.append(mini_a_entry.get('final_diff') if mini_a_entry else None)
        mini_b_diff.append(mini_b_entry.get('final_diff') if mini_b_entry else None)
    
    # Create figure with horizontal bars (topics on y-axis)
    # Use centered bars to show positive/negative convergence
    # Make it taller so bars are thicker
    fig, ax = plt.subplots(figsize=(6, 12))
    
    y = np.arange(len(topics_display))
    height = 0.22  # Bar height - adjusted to prevent overlap
    
    # Helper function to plot bars with None handling
    def plot_bars(y_pos, values, height, label, color):
        for i, val in enumerate(values):
            if val is not None:
                if val >= 0:
                    ax.barh(y_pos[i], val, height, left=0, label=label if i == 0 else '', 
                           alpha=0.8, color=color, linewidth=0.5)
                else:
                    ax.barh(y_pos[i], abs(val), height, left=val, label=label if i == 0 else '', 
                           alpha=0.8, color=color, linewidth=0.5)
    
    plot_bars(y - 1.5*height, nano_a_diff, height, 'Nano, A vs B', '#f28e2b')
    plot_bars(y - 0.5*height, nano_b_diff, height, 'Nano, B vs A', '#59a14f')
    plot_bars(y + 0.5*height, mini_a_diff, height, 'Mini, A vs B', '#e15759')
    plot_bars(y + 1.5*height, mini_b_diff, height, 'Mini, B vs A', '#76b7b2')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2.0)
    ax.set_xlabel('Convergence Direction Bias ($\Delta_{\mathrm{conv}}$)', fontsize=18)
    ax.set_ylabel('Topic', fontsize=18)
    ax.set_title('Convergence Direction Bias: Difference from DeGroot Baseline\n(Positive = converges more positive, Negative = converges more negative)', 
                 fontsize=20, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(topics_display, fontsize=16)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=20, frameon=True, 
              columnspacing=1.2, handlelength=2.0)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, alpha=0.3, axis='x', linewidth=1.5)
    
    # Set symmetric x-axis limits
    all_diffs = [d for d in nano_a_diff + nano_b_diff + mini_a_diff + mini_b_diff if d is not None]
    if all_diffs:
        max_abs = max(abs(d) for d in all_diffs)
        ax.set_xlim(-max_abs * 1.1, max_abs * 1.1)
    
    # Adjust layout to maximize plot area width and height
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    # Make sure plot uses full width and more height
    ax.set_position([0.1, 0.15, 0.9, 0.8])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    import sys
    import os
    
    data_path = "results/divergence_analysis/divergence_metrics.json"
    output_path = "paper/figures/rmse_comparison.png"
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    data = load_data(data_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_rmse_comparison_figure(data, output_path)

