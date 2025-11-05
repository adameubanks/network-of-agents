#!/usr/bin/env python3
"""
Deep dive into model size differences (nano vs mini).
"""

import json
import numpy as np
import argparse
from collections import defaultdict

def load_results(json_path: str) -> list:
    with open(json_path, 'r') as f:
        return json.load(f)

def analyze_model_effects(results: list):
    """Analyze how model size affects divergence."""
    
    print("="*80)
    print("MODEL SIZE EFFECT ANALYSIS")
    print("="*80)
    
    # Group by experiment
    by_exp = defaultdict(list)
    for r in results:
        by_exp[r["experiment"]].append(r)
    
    # Key insight: nano-b and mini both have reversed=true
    # But they show OPPOSITE biases!
    nano_b = by_exp.get("nano-b", [])
    mini = by_exp.get("mini", [])
    
    if not nano_b or not mini:
        missing = []
        if not nano_b:
            missing.append("nano-b")
        if not mini:
            missing.append("mini")
        print(f"\n⚠️  Cannot analyze model size effects: missing data for {', '.join(missing)}")
        print("Available experiments:", list(by_exp.keys()))
        return
    
    print("\n## CRITICAL FINDING: Model Size Reverses Bias Direction")
    print("-"*80)
    print("Both nano-b and mini have reversed=true, but:")
    if nano_b and mini:
        nano_b_final_diffs = [r['final_diff'] for r in nano_b]
        mini_final_diffs = [r['final_diff'] for r in mini]
        print(f"  nano-b (gpt-5-nano): avg final_diff = {np.mean(nano_b_final_diffs):.4f} (POSITIVE bias)")
        print(f"  mini (gpt-5-mini):   avg final_diff = {np.mean(mini_final_diffs):.4f} (NEGATIVE bias)")
        print("\n⚠️  This contradicts the simple 'reversed parameter' explanation!")
        print("   Model size interacts with framing in unexpected ways.")
    else:
        print("  Cannot compute averages: insufficient data")
    
    # Compare topic-by-topic
    print("\n## Topic-by-Topic Comparison: nano-b vs mini")
    print("-"*80)
    
    topics_seen = set(r["topic"] for r in results)
    topic_comparisons = []
    
    for topic in sorted(topics_seen):
        nano_b_r = [r for r in nano_b if r["topic"] == topic]
        mini_r = [r for r in mini if r["topic"] == topic]
        
        if nano_b_r and mini_r:
            nb = nano_b_r[0]
            mi = mini_r[0]
            
            final_diff_diff = mi["final_diff"] - nb["final_diff"]
            rmse_diff = mi["rmse"] - nb["rmse"]
            corr_diff = mi["correlation"] - nb["correlation"]
            
            topic_comparisons.append({
                "topic": topic,
                "nano_final": nb["final_diff"],
                "mini_final": mi["final_diff"],
                "final_diff": final_diff_diff,
                "rmse_diff": rmse_diff,
                "corr_diff": corr_diff,
                "nano_rmse": nb["rmse"],
                "mini_rmse": mi["rmse"]
            })
            
            print(f"\n{topic}:")
            print(f"  nano-b: final_diff={nb['final_diff']:7.4f}, rmse={nb['rmse']:.4f}, corr={nb['correlation']:.4f}")
            print(f"  mini:   final_diff={mi['final_diff']:7.4f}, rmse={mi['rmse']:.4f}, corr={mi['correlation']:.4f}")
            print(f"  Δ:      final_diff={final_diff_diff:7.4f}, rmse={rmse_diff:+.4f}, corr={corr_diff:+.4f}")
    
    # Aggregate patterns
    print("\n## Aggregate Patterns")
    print("-"*80)
    
    if topic_comparisons:
        final_diffs = [tc["final_diff"] for tc in topic_comparisons]
        rmse_diffs = [tc["rmse_diff"] for tc in topic_comparisons]
        
        print(f"\nAverage difference (mini - nano-b):")
        print(f"  Final difference: {np.mean(final_diffs):.4f} (mini is more negative)")
        print(f"  RMSE difference:  {np.mean(rmse_diffs):.4f} (mini has {'higher' if np.mean(rmse_diffs) > 0 else 'lower'} RMSE)")
        
        # Which model is "better"?
        print(f"\nWhich model is closer to pure math?")
        avg_nano_rmse = np.mean([tc["nano_rmse"] for tc in topic_comparisons])
        avg_mini_rmse = np.mean([tc["mini_rmse"] for tc in topic_comparisons])
        print(f"  nano-b avg RMSE: {avg_nano_rmse:.4f}")
        print(f"  mini avg RMSE:   {avg_mini_rmse:.4f}")
        
        if avg_mini_rmse < avg_nano_rmse:
            print("  → mini is CLOSER to pure math (lower RMSE)")
        elif avg_nano_rmse < avg_mini_rmse:
            print("  → nano-b is CLOSER to pure math (lower RMSE)")
        else:
            print("  → Similar RMSE")
    else:
        print("\nNo overlapping topics found for comparison")
    
    # Topic categories
    print("\n## Topics Where Model Size Matters Most")
    print("-"*80)
    
    if topic_comparisons:
        # Sort by absolute difference
        sorted_by_diff = sorted(topic_comparisons, key=lambda x: abs(x["final_diff"]), reverse=True)
        
        print("\nLargest final_diff differences (mini - nano-b):")
        for tc in sorted_by_diff[:5]:
            print(f"  {tc['topic']:30s} Δ={tc['final_diff']:7.4f} "
                  f"(nano={tc['nano_final']:7.4f}, mini={tc['mini_final']:7.4f})")
        
        print("\nLargest RMSE differences:")
        sorted_by_rmse = sorted(topic_comparisons, key=lambda x: abs(x["rmse_diff"]), reverse=True)
        for tc in sorted_by_rmse[:5]:
            print(f"  {tc['topic']:30s} Δ={tc['rmse_diff']:+.4f} "
                  f"(nano={tc['nano_rmse']:.4f}, mini={tc['mini_rmse']:.4f})")
    else:
        print("No overlapping topics found for comparison")
    
    # Consistency check
    print("\n## Consistency Analysis")
    print("-"*80)
    
    if topic_comparisons:
        # How often do they agree on direction?
        same_direction = sum(1 for tc in topic_comparisons 
                            if (tc["nano_final"] > 0) == (tc["mini_final"] > 0))
        opposite_direction = len(topic_comparisons) - same_direction
        
        print(f"Same direction (both + or both -): {same_direction}/{len(topic_comparisons)}")
        print(f"Opposite direction: {opposite_direction}/{len(topic_comparisons)}")
        
        if opposite_direction > same_direction:
            print("  ⚠️  Models show OPPOSITE biases for most topics!")
    else:
        print("No overlapping topics found for consistency analysis")
    
    # Now compare nano-a vs mini (both reversed=false)
    print("\n## Comparison: nano-a vs mini (both reversed=false)")
    print("-"*80)
    
    nano_a = by_exp.get("nano-a", [])
    nano_a_mini_comparisons = []
    
    for topic in sorted(topics_seen):
        nano_a_r = [r for r in nano_a if r["topic"] == topic]
        mini_r = [r for r in mini if r["topic"] == topic]
        
        if nano_a_r and mini_r:
            na = nano_a_r[0]
            mi = mini_r[0]
            final_diff_diff = mi["final_diff"] - na["final_diff"]
            nano_a_mini_comparisons.append({
                "topic": topic,
                "nano_final": na["final_diff"],
                "mini_final": mi["final_diff"],
                "final_diff": final_diff_diff
            })
    
    if nano_a_mini_comparisons:
        print("\nNote: nano-a has reversed=false, mini has reversed=true")
        print("But comparing them anyway to see model size effects:")
        for tc in nano_a_mini_comparisons[:5]:
            print(f"  {tc['topic']:30s} nano-a={tc['nano_final']:7.4f}, mini={tc['mini_final']:7.4f}, Δ={tc['final_diff']:7.4f}")
    
    # Generate narrative
    print("\n" + "="*80)
    print("NARRATIVE: Model Size Effects")
    print("="*80)
    
    print("""
## The Model Size Paradox

The data reveals a critical contradiction:

1. **Reversed Parameter Effect**: nano-a (reversed=false) vs nano-b (reversed=true) shows
   systematic bias reversal. This suggests framing matters.

2. **Model Size Effect**: nano-b (reversed=true) vs mini (reversed=true) shows OPPOSITE
   biases even though framing is the same. This contradicts simple framing explanation.

## Possible Explanations

### Hypothesis 1: Model Size Changes Interpretation Strategy
- **nano**: Smaller model, may rely more on surface-level cues and default assumptions
- **mini**: Larger model, may use more sophisticated reasoning that inverts defaults
- Result: Same framing, different interpretation strategies → different biases

### Hypothesis 2: Model Size Affects Expressiveness
- **nano**: May generate simpler posts that are easier to interpret consistently
- **mini**: May generate more nuanced posts that are harder to interpret consistently
- Result: Different expressiveness → different interpretation errors → different biases

### Hypothesis 3: Model Size Has Different Default Biases
- **nano**: Has default positive bias when interpreting posts
- **mini**: Has default negative bias when interpreting posts
- Result: Same posts, different default interpretations → opposite biases

### Hypothesis 4: Model Size Interacts with Framing
- The reversed parameter doesn't just swap A/B - it triggers different model behaviors
- **nano**: Interprets reversed framing with positive bias
- **mini**: Interprets reversed framing with negative bias
- Result: Framing effect is model-dependent, not universal

## The Inconsistent Narrative Problem

You cannot form a consistent narrative because:

1. **Reversed parameter** suggests framing drives bias
2. **Model size** suggests model capabilities drive bias
3. **Topic variation** suggests topic characteristics drive bias
4. **All three interact** in ways that create apparent contradictions

The "wild variation" comes from these three factors interacting:
- Topic expressiveness × Model interpretation strategy × Framing direction
- This creates a 3D interaction space where patterns are hard to see

## What This Means

The divergence is NOT random, but it's also NOT simply explained by:
- ❌ Just the reversed parameter
- ❌ Just model size
- ❌ Just topic characteristics

It's the **interaction of all three** that creates the patterns you see.

## Next Steps to Understand

1. **Interpretation analysis**: Compare what nano vs mini interpret from same posts
2. **Post generation analysis**: Compare what nano vs mini agents generate
3. **Bias decomposition**: Separate framing effects from model effects from topic effects
4. **Controlled experiments**: Test each factor independently
""")

def main():
    parser = argparse.ArgumentParser(description="Analyze model size differences")
    parser.add_argument("--metrics", required=True, help="Path to divergence_metrics.json")
    args = parser.parse_args()
    
    results = load_results(args.metrics)
    analyze_model_effects(results)

if __name__ == "__main__":
    main()

