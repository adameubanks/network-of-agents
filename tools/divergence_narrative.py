#!/usr/bin/env python3
"""
Generate a narrative explanation of divergence patterns.
"""

import json
import os
import numpy as np
from collections import defaultdict
import argparse

def load_results(json_path: str) -> list:
    with open(json_path, 'r') as f:
        return json.load(f)

def analyze_patterns(results: list):
    """Identify patterns and generate narrative."""
    
    # Group by experiment
    by_exp = defaultdict(list)
    for r in results:
        by_exp[r["experiment"]].append(r)
    
    # Group by topic across experiments
    by_topic = defaultdict(list)
    for r in results:
        by_topic[r["topic"]].append(r)
    
    print("="*80)
    print("DIVERGENCE PATTERN ANALYSIS")
    print("="*80)
    
    print("\n## KEY FINDINGS\n")
    
    # 1. Reversed parameter effect
    print("### 1. Reversed Parameter Effect (A vs B Framing)")
    print("-" * 80)
    
    nano_a_results = by_exp.get("nano-a", [])
    nano_b_results = by_exp.get("nano-b", [])
    
    if nano_a_results and nano_b_results:
        nano_a_avg = np.mean([r["final_diff"] for r in nano_a_results])
        nano_b_avg = np.mean([r["final_diff"] for r in nano_b_results])
        
        print(f"nano-a (reversed=false): Average final difference = {nano_a_avg:.4f}")
        print(f"nano-b (reversed=true):  Average final difference = {nano_b_avg:.4f}")
        print(f"Difference: {nano_b_avg - nano_a_avg:.4f}")
        
        if abs(nano_b_avg - nano_a_avg) > 0.2:
            print("\n⚠️  STRONG REVERSED EFFECT: Swapping A/B creates systematic bias!")
            print("   This suggests LLM interpretation is biased by topic framing.")
        else:
            print("\n✓ Moderate reversed effect")
    else:
        missing = []
        if not nano_a_results:
            missing.append("nano-a")
        if not nano_b_results:
            missing.append("nano-b")
        print(f"Cannot compute reversed parameter effect: missing data for {', '.join(missing)}")
        print("Available experiments:", list(by_exp.keys()))
    
    # 2. Topic-specific patterns
    print("\n### 2. Topic-Specific Divergence Patterns")
    print("-" * 80)
    
    # Find topics with consistent patterns
    high_divergence = []
    low_divergence = []
    inconsistent = []
    
    for topic, topic_results in by_topic.items():
        if len(topic_results) < 2:
            continue
        
        rmse_vals = [r["rmse"] for r in topic_results]
        avg_rmse = np.mean(rmse_vals)
        std_rmse = np.std(rmse_vals)
        
        if avg_rmse > 0.5:
            high_divergence.append((topic, avg_rmse))
        elif avg_rmse < 0.15:
            low_divergence.append((topic, avg_rmse))
        
        if std_rmse > 0.2:
            inconsistent.append((topic, std_rmse))
    
    print("\nHigh Divergence Topics (RMSE > 0.5):")
    for topic, rmse in sorted(high_divergence, key=lambda x: x[1], reverse=True):
        print(f"  - {topic}: {rmse:.4f}")
        # Show experiment differences
        topic_r = [r for r in results if r["topic"] == topic]
        for r in topic_r:
            print(f"    {r['experiment']}: final_diff={r['final_diff']:.4f}, pattern={r['pattern']}")
    
    print("\nLow Divergence Topics (RMSE < 0.15):")
    for topic, rmse in sorted(low_divergence, key=lambda x: x[1]):
        print(f"  - {topic}: {rmse:.4f}")
    
    print("\nInconsistent Across Experiments (high std):")
    for topic, std_val in sorted(inconsistent, key=lambda x: x[1], reverse=True):
        print(f"  - {topic}: std={std_val:.4f}")
    
    # 3. Bias direction analysis
    print("\n### 3. Systematic Bias Analysis")
    print("-" * 80)
    
    for exp_name, exp_results in by_exp.items():
        positive = sum(1 for r in exp_results if r["bias_direction"] == "positive")
        negative = sum(1 for r in exp_results if r["bias_direction"] == "negative")
        neutral = sum(1 for r in exp_results if r["bias_direction"] == "neutral")
        
        print(f"\n{exp_name}:")
        print(f"  Positive bias: {positive} topics")
        print(f"  Negative bias: {negative} topics")
        print(f"  Neutral: {neutral} topics")
        
        if positive > negative + 2:
            print(f"  → Systematic positive bias (LLM interprets more positively)")
        elif negative > positive + 2:
            print(f"  → Systematic negative bias (LLM interprets more negatively)")
    
    # 4. Correlation analysis
    print("\n### 4. Correlation with Pure Math")
    print("-" * 80)
    
    for exp_name, exp_results in by_exp.items():
        corrs = [r["correlation"] for r in exp_results]
        avg_corr = np.mean(corrs)
        print(f"{exp_name}: Average correlation = {avg_corr:.4f}")
        
        if avg_corr < 0:
            print("  → NEGATIVE correlation! LLM trajectory is inverted from pure math")
        elif avg_corr < 0.3:
            print("  → Low correlation - weak relationship to pure math")
        else:
            print("  → Moderate correlation")
    
    # 5. Pattern types
    print("\n### 5. Divergence Pattern Types")
    print("-" * 80)
    
    pattern_counts = defaultdict(int)
    for r in results:
        pattern_counts[r["pattern"]] += 1
    
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{pattern}: {count} topics")
    
    # 6. Narrative explanation
    print("\n" + "="*80)
    print("NARRATIVE EXPLANATION")
    print("="*80)
    
    print("""
The divergence between pure math and LLM-based simulations stems from several key mechanisms:

1. **LLM Interpretation Bias**: 
   Instead of using agents' actual opinions directly (like pure math), the LLM-based
   simulation uses LLM-interpreted opinions extracted from generated posts. This creates
   a "translation layer" where:
   - Agents express opinions as text posts
   - LLM interprets posts back to numerical opinions
   - Interpretation may not perfectly match original opinions
   
2. **Topic Framing Effects (Reversed Parameter)**:
   The "reversed" parameter swaps which topic option is A vs B. This creates systematic
   bias because:
   - LLM interpretation may have default assumptions about topic framing
   - Some topics may be easier to express clearly in one direction
   - The framing affects how posts are interpreted
   
3. **Topic Expressiveness**:
   Different topics vary in how clearly opinions can be expressed in text:
   - Simple/clear topics (e.g., "hot dog sandwich") → lower divergence
   - Complex/polarized topics (e.g., "immigration", "environment_economy") → higher divergence
   - Some topics may have ambiguous language that LLMs interpret inconsistently
   
4. **Model Differences**:
   - nano vs mini models may have different interpretation biases
   - Model size affects how well it can extract nuanced opinions from text
   
5. **Cascading Effects**:
   Small interpretation errors compound over time through the DeGroot update:
   - Early misinterpretations affect all subsequent updates
   - Final convergence values can be significantly different even if trajectories are similar
   
The wild variation you're seeing is because:
- Each topic has different expressiveness characteristics
- The reversed parameter creates systematic bias that varies by topic
- LLM interpretation introduces noise that compounds differently for different topics
- Some topics may have inherent ambiguity that LLMs resolve differently than humans would
""")
    
    # 7. Specific examples
    print("\n### 7. Illustrative Examples")
    print("-" * 80)
    
    # Find immigration (appears in all experiments)
    imm_results = [r for r in results if "immigration" in r["topic"].lower()]
    if imm_results:
        print("\nImmigration (present in all experiments):")
        for r in imm_results:
            print(f"  {r['experiment']}: final_diff={r['final_diff']:.4f}, "
                  f"corr={r['correlation']:.4f}, rmse={r['rmse']:.4f}")
        print("  → Shows how reversed parameter creates opposite biases!")
    
    # Find topics with extreme differences
    print("\nTopics with extreme divergence:")
    extreme = sorted([r for r in results if r["rmse"] > 0.7], 
                     key=lambda x: x["rmse"], reverse=True)[:5]
    for r in extreme:
        print(f"  {r['topic']} ({r['experiment']}): rmse={r['rmse']:.4f}, "
              f"final_diff={r['final_diff']:.4f}, pattern={r['pattern']}")

def main():
    parser = argparse.ArgumentParser(description="Generate divergence narrative")
    parser.add_argument("--metrics", required=True, help="Path to divergence_metrics.json")
    args = parser.parse_args()
    
    results = load_results(args.metrics)
    analyze_patterns(results)

if __name__ == "__main__":
    main()

