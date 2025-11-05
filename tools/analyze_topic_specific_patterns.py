#!/usr/bin/env python3
"""
Analyze topic-specific patterns across models and examine actual posts.
"""

import json
import os
import numpy as np
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

def load_divergence_metrics(json_path: str) -> list:
    with open(json_path, 'r') as f:
        return json.load(f)

def load_topic_json(path: str, topic: str) -> Dict:
    """Load topic JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if "results" in data and topic in data["results"]:
        return data["results"][topic]
    elif "results" in data:
        # Try first result
        first_key = next(iter(data["results"].keys()))
        return data["results"][first_key]
    else:
        return data

def get_posts_and_ratings(data: Dict, timestep: int = 0) -> Tuple[List[str], List]:
    """Extract posts and ratings for a given timestep."""
    posts = []
    ratings = []
    
    if "posts_history" in data and timestep < len(data["posts_history"]):
        posts = data["posts_history"][timestep]
    
    if "ratings_history" in data and timestep < len(data["ratings_history"]):
        ratings = data["ratings_history"][timestep]
    
    return posts, ratings

def analyze_topic_across_models(topic: str, results: list, base_dir: str):
    """Analyze a single topic across all models."""
    
    print("="*80)
    print(f"TOPIC: {topic.upper()}")
    print("="*80)
    
    # Get divergence metrics for this topic
    topic_results = [r for r in results if r["topic"] == topic]
    
    if not topic_results:
        print(f"No data found for topic: {topic}")
        return
    
    print("\n## Divergence Metrics by Model")
    print("-"*80)
    
    for r in sorted(topic_results, key=lambda x: x["experiment"]):
        print(f"{r['experiment']:10s}: final_diff={r['final_diff']:7.4f}, "
              f"rmse={r['rmse']:.4f}, corr={r['correlation']:.4f}, "
              f"pattern={r['pattern']}, bias={r['bias_direction']}")
    
    # Load actual data files
    print("\n## Actual Data Files")
    print("-"*80)
    
    data_files = {}
    
    # Find files for each experiment
    for exp_result in topic_results:
        exp = exp_result["experiment"]
        
        if exp == "nano-a":
            file_path = os.path.join(base_dir, "a_vs_b_nano_degrootsmallworldalltopics_10-14-21-26", f"{topic}.json")
        elif exp == "nano-b":
            file_path = os.path.join(base_dir, "b_vs_a_nano_degrootsmallworldalltopics_10-16-20-33", f"{topic}.json")
        elif exp == "mini":
            # Mini uses long topic names
            long_name = topic.replace("_", " ")
            file_path = os.path.join(base_dir, "a_vs_b_mini_degrootsmallworldalltopicsmini_consolidated_10-25-17-17", 
                                   "degroot", long_name, f"{long_name}.json")
        else:
            continue
        
        if os.path.exists(file_path):
            data = load_topic_json(file_path, topic)
            data_files[exp] = data
            print(f"{exp}: Found {file_path}")
        else:
            print(f"{exp}: File not found: {file_path}")
    
    # Compare posts and interpretations
    if len(data_files) >= 2:
        print("\n## Post and Interpretation Analysis")
        print("-"*80)
        
        # Compare early timestep
        timestep = 0
        print(f"\nTimestep {timestep}:")
        
        for exp, data in data_files.items():
            posts, ratings = get_posts_and_ratings(data, timestep)
            print(f"\n{exp}:")
            print(f"  Posts: {len(posts)}")
            if posts:
                print(f"  Sample posts:")
                for i, post in enumerate(posts[:3]):
                    print(f"    Agent {i}: {post[:100]}...")
            
            if ratings:
                print(f"  Ratings: {len(ratings)} agents rated")
                # Show first agent's ratings
                if ratings[0]:
                    print(f"  Agent 0 rated {len(ratings[0])} neighbors:")
                    for j, (neighbor_id, rating) in enumerate(ratings[0][:3]):
                        print(f"    Neighbor {neighbor_id}: {rating:.3f}")
        
        # Compare final timestep
        print(f"\n{'='*80}")
        print("Final Timestep:")
        print("-"*80)
        
        for exp, data in data_files.items():
            if "posts_history" in data and data["posts_history"]:
                final_timestep = len(data["posts_history"]) - 1
                posts, ratings = get_posts_and_ratings(data, final_timestep)
                
                print(f"\n{exp} (final timestep {final_timestep}):")
                if posts:
                    print(f"  Sample final posts:")
                    for i, post in enumerate(posts[:3]):
                        print(f"    Agent {i}: {post[:100]}...")
                
                # Get opinion history to see final opinions
                if "opinion_history" in data or ("summary_metrics" in data and "opinion_history" in data["summary_metrics"]):
                    if "summary_metrics" in data:
                        opinions = data["summary_metrics"]["opinion_history"]
                    else:
                        opinions = data["opinion_history"]
                    
                    if opinions:
                        final_opinions = opinions[-1]
                        mean_opinion = np.mean(final_opinions)
                        print(f"  Final mean opinion: {mean_opinion:.4f}")
                        print(f"  Opinion range: [{min(final_opinions):.4f}, {max(final_opinions):.4f}]")
    
    # Topic-specific insights
    print("\n## Topic-Specific Patterns")
    print("-"*80)
    
    # Check if there are clear patterns
    final_diffs = [r["final_diff"] for r in topic_results]
    experiments = [r["experiment"] for r in topic_results]
    
    print("\nFinal differences by experiment:")
    for exp, diff in zip(experiments, final_diffs):
        print(f"  {exp}: {diff:7.4f}")
    
    # Check if models agree or disagree
    if len(final_diffs) >= 2:
        if all(d > 0 for d in final_diffs) or all(d < 0 for d in final_diffs):
            print("\n✓ All models agree on bias direction")
        else:
            print("\n⚠️  Models disagree on bias direction!")
            positive = [exp for exp, d in zip(experiments, final_diffs) if d > 0]
            negative = [exp for exp, d in zip(experiments, final_diffs) if d < 0]
            print(f"   Positive bias: {positive}")
            print(f"   Negative bias: {negative}")

def analyze_all_topics_by_model(results: list):
    """Group topics by how they behave across models."""
    
    print("\n" + "="*80)
    print("TOPIC BEHAVIOR CLASSIFICATION")
    print("="*80)
    
    topics = set(r["topic"] for r in results)
    
    # Classify topics by model agreement
    consistent_topics = []  # All models agree on direction
    inconsistent_topics = []  # Models disagree on direction
    high_variance_topics = []  # Large differences between models
    
    for topic in sorted(topics):
        topic_results = [r for r in results if r["topic"] == topic]
        
        if len(topic_results) < 2:
            continue
        
        final_diffs = [r["final_diff"] for r in topic_results]
        experiments = [r["experiment"] for r in topic_results]
        
        # Check agreement
        all_positive = all(d > 0 for d in final_diffs)
        all_negative = all(d < 0 for d in final_diffs)
        
        # Check variance
        variance = np.var(final_diffs)
        max_diff = max(final_diffs) - min(final_diffs)
        
        topic_info = {
            "topic": topic,
            "final_diffs": dict(zip(experiments, final_diffs)),
            "variance": variance,
            "max_diff": max_diff
        }
        
        if all_positive or all_negative:
            consistent_topics.append(topic_info)
        else:
            inconsistent_topics.append(topic_info)
        
        if max_diff > 0.5:
            high_variance_topics.append(topic_info)
    
    print("\n## Topics Where Models Agree (same bias direction)")
    print("-"*80)
    for ti in sorted(consistent_topics, key=lambda x: x["variance"]):
        print(f"{ti['topic']:30s}: {ti['final_diffs']}")
    
    print("\n## Topics Where Models Disagree (opposite bias directions)")
    print("-"*80)
    for ti in sorted(inconsistent_topics, key=lambda x: abs(x['max_diff']), reverse=True):
        print(f"{ti['topic']:30s}: {ti['final_diffs']} (max_diff={ti['max_diff']:.4f})")
    
    print("\n## Topics with High Variance Across Models")
    print("-"*80)
    for ti in sorted(high_variance_topics, key=lambda x: x["max_diff"], reverse=True):
        print(f"{ti['topic']:30s}: max_diff={ti['max_diff']:.4f}, {ti['final_diffs']}")

def main():
    parser = argparse.ArgumentParser(description="Analyze topic-specific patterns")
    parser.add_argument("--metrics", required=True, help="Path to divergence_metrics.json")
    parser.add_argument("--base-dir", default="results", help="Base directory for result files")
    parser.add_argument("--topic", help="Analyze specific topic (otherwise analyze all)")
    args = parser.parse_args()
    
    results = load_divergence_metrics(args.metrics)
    
    if args.topic:
        analyze_topic_across_models(args.topic, results, args.base_dir)
    else:
        # Analyze all topics
        topics = sorted(set(r["topic"] for r in results))
        
        # First, classify all topics
        analyze_all_topics_by_model(results)
        
        # Then do detailed analysis for a few key topics
        print("\n" + "="*80)
        print("DETAILED ANALYSIS OF KEY TOPICS")
        print("="*80)
        
        # Pick topics that show interesting patterns
        key_topics = ["immigration", "environment_economy", "corporate_activism", 
                     "hot_dog_sandwich", "gun_safety"]
        
        for topic in key_topics:
            if topic in topics or any(topic.lower() in t.lower() for t in topics):
                matching_topics = [t for t in topics if topic.lower() in t.lower()]
                if matching_topics:
                    analyze_topic_across_models(matching_topics[0], results, args.base_dir)
                    print("\n")

if __name__ == "__main__":
    main()

