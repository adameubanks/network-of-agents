#!/usr/bin/env python3
"""
Generate a single JSON file with all experimental results for the paper.
This is the source of truth for all numbers in the paper.
"""

import json
import os
import numpy as np
from tools.analyze_divergence import load_mean_opinions, compute_divergence_metrics
from tools.plot_utils import to_long_topic_key

def get_mean_final_opinion(data_path, topic_key):
    """Get mean final opinion from a data file."""
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        if 'results' in data:
            if topic_key in data['results']:
                topic_data = data['results'][topic_key]
            else:
                topic_data = list(data['results'].values())[0]
            
            if 'summary_metrics' in topic_data and 'mean_opinions' in topic_data['summary_metrics']:
                means = topic_data['summary_metrics']['mean_opinions']
                return means[-1] if means else None
            if 'opinion_history' in topic_data:
                final_opinions = topic_data['opinion_history'][-1]
                return np.mean(final_opinions) if final_opinions else None
        elif 'mean_opinions' in data:
            means = data['mean_opinions']
            return means[-1] if means else None
        elif 'opinion_history' in data:
            final_opinions = data['opinion_history'][-1]
            return np.mean(final_opinions) if final_opinions else None
    except:
        pass
    return None

def get_pattern(final_diff):
    """Determine pattern based on final_diff."""
    return 'stable' if abs(final_diff) < 0.3 else 'different'

def generate_all_results():
    """Generate complete results JSON from all experimental data."""
    pure_path = "results/experiments/baseline/pure_math/pure_math_smallworld.json"
    pure = load_mean_opinions(pure_path, "pure_math")
    
    nano_a_dir = "results/experiments/llm/a_vs_b_nano"
    nano_b_dir = "results/experiments/llm/b_vs_a_nano"
    mini_a_dir = "results/experiments/llm/a_vs_b_mini"
    mini_b_dir = "results/experiments/llm/b_vs_a_mini"
    
    topics = [f[:-5] for f in os.listdir(nano_a_dir) if f.endswith('.json') and f != 'experiment_metadata.json']
    
    topic_display_names = {
        'sandwich': 'Hot dog sandwich',
        'immigration': 'Immigration',
        'activism': 'Corporate activism',
        'economy': 'Environment economy',
        'etiquette': 'Restaurant etiquette',
        'safety': 'Gun safety',
        'democracy': 'Social media democracy',
        'cloning': 'Human cloning',
        'paper': 'Toilet paper',
        'weddings': 'Child-free weddings',
    }
    
    all_results = []
    
    for topic in sorted(topics):
        long_key = to_long_topic_key(topic)
        display_name = topic_display_names.get(topic, topic.replace('_', ' ').title())
        
        # Nano A vs B
        nano_a_path = os.path.join(nano_a_dir, f"{topic}.json")
        if os.path.exists(nano_a_path):
            try:
                nano_a = load_mean_opinions(nano_a_path, topic)
                metrics = compute_divergence_metrics(pure, nano_a)
                mean_final = get_mean_final_opinion(nano_a_path, topic)
                all_results.append({
                    'topic': display_name,
                    'model': 'nano',
                    'framing': 'A vs B',
                    'rmse': float(metrics['rmse']),
                    'mae': float(metrics['mae']),
                    'corr': float(metrics['correlation']),
                    'final_diff': float(metrics['final_diff']),
                    'early_rmse': float(metrics['early_rmse']),
                    'late_rmse': float(metrics['late_rmse']),
                    'pattern': get_pattern(metrics['final_diff']),
                    'mean_final': float(mean_final) if mean_final is not None else None
                })
            except Exception as e:
                print(f"Error loading {topic} nano A vs B: {e}")
        
        # Nano B vs A
        nano_b_path = os.path.join(nano_b_dir, f"{long_key}.json")
        if os.path.exists(nano_b_path):
            try:
                nano_b = load_mean_opinions(nano_b_path, long_key)
                metrics = compute_divergence_metrics(pure, nano_b)
                mean_final = get_mean_final_opinion(nano_b_path, long_key)
                all_results.append({
                    'topic': display_name,
                    'model': 'nano',
                    'framing': 'B vs A',
                    'rmse': float(metrics['rmse']),
                    'mae': float(metrics['mae']),
                    'corr': float(metrics['correlation']),
                    'final_diff': float(metrics['final_diff']),
                    'early_rmse': float(metrics['early_rmse']),
                    'late_rmse': float(metrics['late_rmse']),
                    'pattern': get_pattern(metrics['final_diff']),
                    'mean_final': float(mean_final) if mean_final is not None else None
                })
            except Exception as e:
                print(f"Error loading {topic} nano B vs A: {e}")
        
        # Mini A vs B
        mini_a_path = os.path.join(mini_a_dir, "degroot", long_key, f"{long_key}.json")
        if not os.path.exists(mini_a_path):
            mini_a_path = os.path.join(mini_a_dir, "degroot", long_key, f"{long_key}_streaming.json")
        if os.path.exists(mini_a_path):
            try:
                mini_a = load_mean_opinions(mini_a_path, long_key)
                metrics = compute_divergence_metrics(pure, mini_a)
                mean_final = get_mean_final_opinion(mini_a_path, long_key)
                all_results.append({
                    'topic': display_name,
                    'model': 'mini',
                    'framing': 'A vs B',
                    'rmse': float(metrics['rmse']),
                    'mae': float(metrics['mae']),
                    'corr': float(metrics['correlation']),
                    'final_diff': float(metrics['final_diff']),
                    'early_rmse': float(metrics['early_rmse']),
                    'late_rmse': float(metrics['late_rmse']),
                    'pattern': get_pattern(metrics['final_diff']),
                    'mean_final': float(mean_final) if mean_final is not None else None
                })
            except Exception as e:
                print(f"Error loading {topic} mini A vs B: {e}")
        
        # Mini B vs A
        mini_b_path = os.path.join(mini_b_dir, "degroot", long_key, f"{long_key}.json")
        if not os.path.exists(mini_b_path):
            mini_b_path = os.path.join(mini_b_dir, "degroot", long_key, f"{long_key}_streaming.json")
        if os.path.exists(mini_b_path):
            try:
                mini_b = load_mean_opinions(mini_b_path, long_key)
                if len(mini_b) > 0:
                    metrics = compute_divergence_metrics(pure, mini_b)
                    mean_final = get_mean_final_opinion(mini_b_path, long_key)
                    all_results.append({
                        'topic': display_name,
                        'model': 'mini',
                        'framing': 'B vs A',
                        'rmse': float(metrics['rmse']),
                        'mae': float(metrics['mae']),
                        'corr': float(metrics['correlation']),
                        'final_diff': float(metrics['final_diff']),
                        'early_rmse': float(metrics['early_rmse']),
                        'late_rmse': float(metrics['late_rmse']),
                        'pattern': get_pattern(metrics['final_diff']),
                        'mean_final': float(mean_final) if mean_final is not None else None
                    })
            except Exception as e:
                print(f"Error loading {topic} mini B vs A: {e}")
    
    # Compute summary statistics for Table 1
    by_condition = {}
    for r in all_results:
        key = f"{r['model']}, {r['framing']}"
        if key not in by_condition:
            by_condition[key] = []
        by_condition[key].append(r)
    
    summary_stats = {}
    for condition, results in by_condition.items():
        rmse_vals = [r['rmse'] for r in results]
        mae_vals = [r['mae'] for r in results]
        corr_vals = [r['corr'] for r in results]
        
        summary_stats[condition] = {
            'mean_rmse': float(np.mean(rmse_vals)),
            'std_rmse': float(np.std(rmse_vals)),
            'mean_mae': float(np.mean(mae_vals)),
            'std_mae': float(np.std(mae_vals)),
            'mean_corr': float(np.mean(corr_vals)),
            'std_corr': float(np.std(corr_vals)),
            'n': len(results)
        }
    
    # Compute framing effects for Table 2
    framing_effects = {}
    for topic in topic_display_names.values():
        nano_a = next((r for r in all_results if r['topic'] == topic and r['model'] == 'nano' and r['framing'] == 'A vs B'), None)
        nano_b = next((r for r in all_results if r['topic'] == topic and r['model'] == 'nano' and r['framing'] == 'B vs A'), None)
        mini_a = next((r for r in all_results if r['topic'] == topic and r['model'] == 'mini' and r['framing'] == 'A vs B'), None)
        mini_b = next((r for r in all_results if r['topic'] == topic and r['model'] == 'mini' and r['framing'] == 'B vs A'), None)
        
        nano_effect = None
        if nano_a and nano_b and nano_a['mean_final'] is not None and nano_b['mean_final'] is not None:
            nano_effect = float(nano_b['mean_final'] - nano_a['mean_final'])
        
        mini_effect = None
        if mini_a and mini_b and mini_a['mean_final'] is not None and mini_b['mean_final'] is not None:
            mini_effect = float(mini_b['mean_final'] - mini_a['mean_final'])
        
        if nano_effect is not None or mini_effect is not None:
            framing_effects[topic] = {
                'nano': nano_effect,
                'mini': mini_effect
            }
    
    output = {
        'all_results': all_results,
        'summary_stats': summary_stats,
        'framing_effects': framing_effects
    }
    
    return output

def main():
    print("Generating complete results JSON from experimental data...")
    results = generate_all_results()
    
    output_file = "results/paper_data.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved {len(results['all_results'])} entries to {output_file}")
    print(f"✓ Summary stats for {len(results['summary_stats'])} conditions")
    print(f"✓ Framing effects for {len(results['framing_effects'])} topics")

if __name__ == "__main__":
    main()
