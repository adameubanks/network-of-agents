#!/usr/bin/env python3
"""
Comprehensive analysis of divergence patterns between pure math and LLM-based simulations.
"""

import json
import os
import numpy as np
import argparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

def load_mean_opinions(data_path: str, topic: str = None) -> np.ndarray:
    """Load mean opinions from various JSON structures."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Try different structures
    if "mean_opinions" in data:
        return np.array(data["mean_opinions"], dtype=float)
    if "summary_metrics" in data and "mean_opinions" in data["summary_metrics"]:
        return np.array(data["summary_metrics"]["mean_opinions"], dtype=float)
    if "results" in data:
        if topic and topic in data["results"]:
            sm = data["results"][topic].get("summary_metrics", {})
            if "mean_opinions" in sm:
                return np.array(sm["mean_opinions"], dtype=float)
        # Try first result if topic not specified
        if data["results"]:
            first_key = next(iter(data["results"].keys()))
            sm = data["results"][first_key].get("summary_metrics", {})
            if "mean_opinions" in sm:
                return np.array(sm["mean_opinions"], dtype=float)
    
    # Fallback: compute from opinion_history
    if "opinion_history" in data:
        history = data["opinion_history"]
        means = [np.mean(step) for step in history]
        return np.array(means, dtype=float)
    
    raise KeyError(f"Could not find mean_opinions in {data_path}")

def compute_divergence_metrics(pure: np.ndarray, llm: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive divergence metrics."""
    n = min(len(pure), len(llm))
    pure_n = pure[:n]
    llm_n = llm[:n]
    
    # Basic metrics
    rmse = float(np.sqrt(np.mean((pure_n - llm_n) ** 2)))
    mae = float(np.mean(np.abs(pure_n - llm_n)))
    max_diff = float(np.max(np.abs(pure_n - llm_n)))
    
    # Correlation
    corr = float(np.corrcoef(pure_n, llm_n)[0, 1]) if len(pure_n) > 1 else 0.0
    
    # Directional bias
    mean_diff = float(np.mean(llm_n - pure_n))
    
    # Convergence differences
    pure_final = pure_n[-1] if len(pure_n) > 0 else 0.0
    llm_final = llm_n[-1] if len(llm_n) > 0 else 0.0
    final_diff = llm_final - pure_final
    
    # Early vs late divergence
    early_n = max(1, n // 3)
    late_n = n - early_n
    early_rmse = float(np.sqrt(np.mean((pure_n[:early_n] - llm_n[:early_n]) ** 2)))
    late_rmse = float(np.sqrt(np.mean((pure_n[early_n:] - llm_n[early_n:]) ** 2))) if late_n > 0 else 0.0
    
    # Divergence rate (how quickly they separate)
    diffs = np.abs(pure_n - llm_n)
    if len(diffs) > 1:
        divergence_rate = float(np.mean(np.diff(diffs)))  # Average increase in difference
    else:
        divergence_rate = 0.0
    
    return {
        "rmse": rmse,
        "mae": mae,
        "max_diff": max_diff,
        "correlation": corr,
        "mean_bias": mean_diff,
        "final_diff": final_diff,
        "early_rmse": early_rmse,
        "late_rmse": late_rmse,
        "divergence_rate": divergence_rate,
        "n_steps": n
    }

def analyze_topic(pure_path: str, llm_path: str, topic: str) -> Dict[str, any]:
    """Analyze divergence for a single topic."""
    try:
        pure = load_mean_opinions(pure_path, "pure_math")
        llm = load_mean_opinions(llm_path, topic)
        
        metrics = compute_divergence_metrics(pure, llm)
        metrics["topic"] = topic
        
        # Determine divergence pattern
        n = min(len(pure), len(llm))
        pure_n = pure[:n]
        llm_n = llm[:n]
        
        # Check if LLM converges to different value
        final_diff_abs = abs(metrics["final_diff"])
        if final_diff_abs > 0.2:
            pattern = "different_convergence"
        elif metrics["divergence_rate"] > 0.01:
            pattern = "diverging"
        elif metrics["divergence_rate"] < -0.01:
            pattern = "converging"
        else:
            pattern = "stable_divergence"
        
        # Check bias direction
        if metrics["mean_bias"] > 0.1:
            bias_dir = "positive"
        elif metrics["mean_bias"] < -0.1:
            bias_dir = "negative"
        else:
            bias_dir = "neutral"
        
        metrics["pattern"] = pattern
        metrics["bias_direction"] = bias_dir
        
        return metrics
    except Exception as e:
        print(f"Error analyzing {topic}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze divergence patterns")
    parser.add_argument("--pure", required=True, help="Path to pure_math_smallworld.json")
    parser.add_argument("--nano-a", help="Path to a_vs_b nano directory")
    parser.add_argument("--nano-b", help="Path to b_vs_a nano directory")
    parser.add_argument("--mini", help="Path to a_vs_b_mini directory")
    parser.add_argument("--out", default="results/divergence_analysis", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    all_results = []
    
    # Analyze nano-a if provided
    if args.nano_a:
        print("Analyzing nano-a (a_vs_b)...")
        nano_a_dir = args.nano_a
        topics = [f[:-5] for f in os.listdir(nano_a_dir) if f.endswith('.json') and f != 'experiment_metadata.json']
        
        for topic in topics:
            llm_path = os.path.join(nano_a_dir, f"{topic}.json")
            result = analyze_topic(args.pure, llm_path, topic)
            if result:
                result["experiment"] = "nano-a"
                all_results.append(result)
    
    # Analyze nano-b if provided
    if args.nano_b:
        print("Analyzing nano-b (b_vs_a)...")
        nano_b_dir = args.nano_b
        topics = [f[:-5] for f in os.listdir(nano_b_dir) if f.endswith('.json') and f != 'experiment_metadata.json']
        
        for topic in topics:
            llm_path = os.path.join(nano_b_dir, f"{topic}.json")
            result = analyze_topic(args.pure, llm_path, topic)
            if result:
                result["experiment"] = "nano-b"
                all_results.append(result)
    
    # Analyze mini if provided
    if args.mini:
        print("Analyzing mini (a_vs_b_mini)...")
        mini_dir = args.mini
        from tools.plot_utils import to_long_topic_key
        
        # Find topics from degroot subdirectory
        degroot_dir = os.path.join(mini_dir, "degroot")
        if os.path.exists(degroot_dir):
            topics = [d for d in os.listdir(degroot_dir) if os.path.isdir(os.path.join(degroot_dir, d))]
            
            for topic_long in topics:
                topic_short = topic_long.replace("_", " ")
                llm_path = os.path.join(degroot_dir, topic_long, f"{topic_long}.json")
                if os.path.exists(llm_path):
                    result = analyze_topic(args.pure, llm_path, topic_short)
                    if result:
                        result["experiment"] = "mini"
                        all_results.append(result)
    
    if not all_results:
        print("No results to analyze!")
        return
    
    # Aggregate analysis
    print("\n" + "="*60)
    print("DIVERGENCE ANALYSIS SUMMARY")
    print("="*60)
    
    # Group by experiment
    by_experiment = defaultdict(list)
    for r in all_results:
        by_experiment[r["experiment"]].append(r)
    
    for exp_name, results in by_experiment.items():
        print(f"\n{exp_name.upper()}:")
        print(f"  Topics: {len(results)}")
        avg_rmse = np.mean([r["rmse"] for r in results])
        avg_corr = np.mean([r["correlation"] for r in results])
        avg_final_diff = np.mean([r["final_diff"] for r in results])
        print(f"  Avg RMSE: {avg_rmse:.4f}")
        print(f"  Avg Correlation: {avg_corr:.4f}")
        print(f"  Avg Final Difference: {avg_final_diff:.4f}")
        
        # Pattern distribution
        patterns = [r["pattern"] for r in results]
        pattern_counts = {p: patterns.count(p) for p in set(patterns)}
        print(f"  Patterns: {pattern_counts}")
        
        # Bias distribution
        biases = [r["bias_direction"] for r in results]
        bias_counts = {b: biases.count(b) for b in set(biases)}
        print(f"  Bias Directions: {bias_counts}")
    
    # Topic-level analysis
    print("\n" + "-"*60)
    print("TOPIC-LEVEL ANALYSIS:")
    print("-"*60)
    
    topics_seen = set(r["topic"] for r in all_results)
    for topic in sorted(topics_seen):
        topic_results = [r for r in all_results if r["topic"] == topic]
        print(f"\n{topic}:")
        for r in topic_results:
            print(f"  {r['experiment']}: RMSE={r['rmse']:.4f}, "
                  f"Corr={r['correlation']:.4f}, "
                  f"FinalDiff={r['final_diff']:.4f}, "
                  f"Pattern={r['pattern']}, "
                  f"Bias={r['bias_direction']}")
    
    # Save detailed results
    output_file = os.path.join(args.out, "divergence_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")
    
    # Create visualization
    if len(all_results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # RMSE by topic and experiment
        ax = axes[0, 0]
        experiments = list(set(r["experiment"] for r in all_results))
        topics = sorted(set(r["topic"] for r in all_results))
        x = np.arange(len(topics))
        width = 0.8 / len(experiments)
        
        for i, exp in enumerate(experiments):
            rmse_values = []
            for topic in topics:
                topic_r = [r for r in all_results if r["topic"] == topic and r["experiment"] == exp]
                rmse_values.append(topic_r[0]["rmse"] if topic_r else 0)
            ax.bar(x + i*width, rmse_values, width, label=exp)
        
        ax.set_xlabel("Topic")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE by Topic and Experiment")
        ax.set_xticks(x + width * (len(experiments)-1) / 2)
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Correlation by topic
        ax = axes[0, 1]
        for exp in experiments:
            corr_values = []
            for topic in topics:
                topic_r = [r for r in all_results if r["topic"] == topic and r["experiment"] == exp]
                corr_values.append(topic_r[0]["correlation"] if topic_r else 0)
            ax.plot(topics, corr_values, marker='o', label=exp)
        
        ax.set_xlabel("Topic")
        ax.set_ylabel("Correlation")
        ax.set_title("Correlation with Pure Math")
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Final difference by topic
        ax = axes[1, 0]
        for exp in experiments:
            diff_values = []
            for topic in topics:
                topic_r = [r for r in all_results if r["topic"] == topic and r["experiment"] == exp]
                diff_values.append(topic_r[0]["final_diff"] if topic_r else 0)
            ax.plot(topics, diff_values, marker='o', label=exp)
        
        ax.set_xlabel("Topic")
        ax.set_ylabel("Final Difference (LLM - Pure)")
        ax.set_title("Final Convergence Difference")
        ax.set_xticklabels(topics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Pattern distribution
        ax = axes[1, 1]
        patterns = [r["pattern"] for r in all_results]
        pattern_counts = {p: patterns.count(p) for p in set(patterns)}
        ax.bar(pattern_counts.keys(), pattern_counts.values())
        ax.set_xlabel("Divergence Pattern")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Divergence Patterns")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(args.out, "divergence_analysis.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Visualization saved to {plot_path}")

if __name__ == "__main__":
    main()

