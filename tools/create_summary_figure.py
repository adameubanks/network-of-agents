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
    mini_a_rmse = []
    mini_b_rmse = []
    
    for topic in all_topics:
        nano_a_rmse.append(by_exp.get('nano-a', {}).get(topic, {}).get('rmse', 0))
        nano_b_rmse.append(by_exp.get('nano-b', {}).get(topic, {}).get('rmse', 0))
        mini_a_rmse.append(by_exp.get('mini', {}).get(topic, {}).get('rmse', 0))
        mini_b_rmse.append(by_exp.get('mini-b', {}).get(topic, {}).get('rmse', 0))
    
    # Create figure with horizontal bars (topics on y-axis)
    fig, ax = plt.subplots(figsize=(8, 10))
    
    y = np.arange(len(topics_display))
    height = 0.2
    
    bars1 = ax.barh(y - 1.5*height, nano_a_rmse, height, label='Nano, A vs B', alpha=0.8, color='#f28e2b')
    bars2 = ax.barh(y - 0.5*height, nano_b_rmse, height, label='Nano, B vs A', alpha=0.8, color='#59a14f')
    bars3 = ax.barh(y + 0.5*height, mini_a_rmse, height, label='Mini, A vs B', alpha=0.8, color='#e15759')
    bars4 = ax.barh(y + 1.5*height, mini_b_rmse, height, label='Mini, B vs A', alpha=0.8, color='#76b7b2')
    
    ax.set_xlabel('RMSE', fontsize=11)
    ax.set_ylabel('Topic', fontsize=11)
    ax.set_title('RMSE by Topic and Condition', fontsize=12, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(topics_display, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    all_rmse = nano_a_rmse + nano_b_rmse + mini_a_rmse + mini_b_rmse
    ax.set_xlim(0, max(all_rmse) * 1.1 if all_rmse else 1.0)
    
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

