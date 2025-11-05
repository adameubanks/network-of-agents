#!/usr/bin/env python3
"""
Analyze the mismatch between posts, interpretations, and opinions.
This is the key to understanding why models behave differently.
"""

import json
import os
import numpy as np
import argparse
from typing import Dict, List, Tuple

def load_topic_data(file_path: str, topic: str) -> Dict:
    """Load topic JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if "results" in data and topic in data["results"]:
        return data["results"][topic]
    elif "results" in data:
        return next(iter(data["results"].values()))
    else:
        return data

def analyze_interpretation_chain(data: Dict, timestep: int, agent_id: int = 0):
    """Analyze the full chain: opinion → post → interpretation → new opinion."""
    
    if timestep >= len(data.get("opinion_history", [])):
        return None
    
    opinions = data["opinion_history"][timestep]
    agent_opinion = opinions[agent_id]
    
    # Get post
    post = None
    if timestep < len(data.get("posts_history", [])):
        posts = data["posts_history"][timestep]
        if posts and agent_id < len(posts):
            post = posts[agent_id]
    
    # Get how others interpreted this agent's post
    interpretations = []
    if timestep < len(data.get("ratings_history", [])):
        ratings = data["ratings_history"][timestep]
        # Find ratings where others rated this agent
        for rater_id, neighbor_ratings in enumerate(ratings):
            if neighbor_ratings:
                for neighbor_id, rating in neighbor_ratings:
                    if neighbor_id == agent_id:
                        interpretations.append((rater_id, rating))
    
    # Get next opinion
    next_opinion = None
    if timestep + 1 < len(data.get("opinion_history", [])):
        next_opinions = data["opinion_history"][timestep + 1]
        next_opinion = next_opinions[agent_id]
    
    return {
        "timestep": timestep,
        "agent_id": agent_id,
        "opinion": agent_opinion,
        "post": post,
        "interpretations": interpretations,
        "mean_interpretation": np.mean([r for _, r in interpretations]) if interpretations else None,
        "next_opinion": next_opinion,
        "opinion_change": next_opinion - agent_opinion if next_opinion is not None else None
    }

def compare_models_across_topic(topic: str, base_dir: str):
    """Compare how different models interpret the same topic."""
    
    print("="*80)
    print(f"INTERPRETATION CHAIN ANALYSIS: {topic.upper()}")
    print("="*80)
    
    # File paths
    files = {
        "nano-a": os.path.join(base_dir, "a_vs_b_nano_degrootsmallworldalltopics_10-14-21-26", f"{topic}.json"),
        "nano-b": os.path.join(base_dir, "b_vs_a_nano_degrootsmallworldalltopics_10-16-20-33", f"{topic}.json"),
        "mini": os.path.join(base_dir, "a_vs_b_mini_degrootsmallworldalltopicsmini_consolidated_10-25-17-17", 
                           "degroot", topic.replace("_", " "), f"{topic.replace('_', ' ')}.json")
    }
    
    # Load data
    data_by_model = {}
    for model, path in files.items():
        if os.path.exists(path):
            data_by_model[model] = load_topic_data(path, topic)
            print(f"\n✓ Loaded {model}")
        else:
            print(f"\n✗ {model}: File not found: {path}")
    
    if len(data_by_model) < 2:
        print("\nNeed at least 2 models to compare")
        return
    
    # Analyze interpretation chains
    print("\n" + "="*80)
    print("INTERPRETATION CHAIN COMPARISON")
    print("="*80)
    
    # Pick a few timesteps to analyze
    timesteps = [1, 5, 10, 15, 20]
    agent_id = 0
    
    for ts in timesteps:
        print(f"\n{'='*80}")
        print(f"TIMESTEP {ts}")
        print("-"*80)
        
        for model, data in data_by_model.items():
            chain = analyze_interpretation_chain(data, ts, agent_id)
            
            if chain:
                print(f"\n{model.upper()}:")
                print(f"  Opinion: {chain['opinion']:.4f}")
                if chain['post']:
                    print(f"  Post: {chain['post'][:150]}...")
                if chain['interpretations']:
                    print(f"  Interpreted by {len(chain['interpretations'])} neighbors:")
                    print(f"    Mean interpretation: {chain['mean_interpretation']:.4f}")
                    print(f"    Range: [{min(r for _, r in chain['interpretations']):.4f}, "
                          f"{max(r for _, r in chain['interpretations']):.4f}]")
                if chain['next_opinion'] is not None:
                    print(f"  Next opinion: {chain['next_opinion']:.4f} (change: {chain['opinion_change']:+.4f})")
                
                # Check for mismatch
                if chain['post'] and chain['mean_interpretation'] is not None:
                    # Simple heuristic: if post is positive-sounding but interpretation is negative
                    post_lower = chain['post'].lower()
                    positive_words = ['good', 'strengthen', 'benefit', 'positive', 'support', 'believe']
                    negative_words = ['bad', 'harm', 'problem', 'concern', 'worry', 'against']
                    
                    has_positive = any(word in post_lower for word in positive_words)
                    has_negative = any(word in post_lower for word in negative_words)
                    
                    if has_positive and not has_negative and chain['mean_interpretation'] < -0.3:
                        print(f"  ⚠️  MISMATCH: Positive post but negative interpretation!")
                    elif has_negative and not has_positive and chain['mean_interpretation'] > 0.3:
                        print(f"  ⚠️  MISMATCH: Negative post but positive interpretation!")
    
    # Analyze final state
    print("\n" + "="*80)
    print("FINAL STATE ANALYSIS")
    print("="*80)
    
    for model, data in data_by_model.items():
        if "final_opinions" in data:
            final_opinions = data["final_opinions"]
            final_mean = np.mean(final_opinions)
            final_std = np.std(final_opinions)
            
            print(f"\n{model.upper()}:")
            print(f"  Final mean opinion: {final_mean:.4f}")
            print(f"  Final std: {final_std:.4f}")
            print(f"  Range: [{min(final_opinions):.4f}, {max(final_opinions):.4f}]")
            
            # Check final posts
            if "posts_history" in data and data["posts_history"]:
                final_ts = len(data["posts_history"]) - 1
                final_posts = data["posts_history"][final_ts]
                if final_posts:
                    print(f"  Sample final posts:")
                    for i, post in enumerate(final_posts[:3]):
                        print(f"    Agent {i} (opinion={final_opinions[i]:.4f}): {post[:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Analyze post-interpretation mismatch")
    parser.add_argument("--topic", required=True, help="Topic to analyze")
    parser.add_argument("--base-dir", default="results", help="Base directory for result files")
    args = parser.parse_args()
    
    compare_models_across_topic(args.topic, args.base_dir)

if __name__ == "__main__":
    main()

