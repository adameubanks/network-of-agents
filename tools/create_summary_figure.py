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
    """Create a bar chart comparing RMSE across topics for different conditions."""
    
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
    
    # Prepare data for plotting
    topics_display = [t.replace('_', ' ').title() for t in all_topics]
    nano_a_rmse = []
    nano_b_rmse = []
    mini_rmse = []
    
    for topic in all_topics:
        nano_a_rmse.append(by_exp.get('nano-a', {}).get(topic, {}).get('rmse', 0))
        nano_b_rmse.append(by_exp.get('nano-b', {}).get(topic, {}).get('rmse', 0))
        mini_rmse.append(by_exp.get('mini', {}).get(topic, {}).get('rmse', 0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(topics_display))
    width = 0.25
    
    bars1 = ax.bar(x - width, nano_a_rmse, width, label='Nano, A vs B', alpha=0.8)
    bars2 = ax.bar(x, nano_b_rmse, width, label='Nano, B vs A', alpha=0.8)
    bars3 = ax.bar(x + width, mini_rmse, width, label='Mini, A vs B', alpha=0.8)
    
    ax.set_xlabel('Topic', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('RMSE by Topic and Condition', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(topics_display, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(nano_a_rmse), max(nano_b_rmse), max(mini_rmse)) * 1.1)
    
    plt.tight_layout()
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

