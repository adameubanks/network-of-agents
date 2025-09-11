#!/usr/bin/env python3
"""
Script to add results from additional models (Grok, GPT-5) to the analysis
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd

def add_model_results(model_name, results_dir, output_file="figures/complete_analysis.csv"):
    """
    Add results from a new model to the comprehensive analysis
    
    Args:
        model_name: Name of the model (e.g., "grok", "gpt-5")
        results_dir: Directory containing the new model's results
        output_file: Output file for combined results
    """
    
    # Load existing results
    existing_file = Path("figures/summary_statistics.csv")
    if existing_file.exists():
        existing_df = pd.read_csv(existing_file)
    else:
        existing_df = pd.DataFrame()
    
    # Load new model results
    new_results = {}
    results_path = Path(results_dir)
    
    for topic_dir in results_path.iterdir():
        if topic_dir.is_dir() and topic_dir.name != "test_topic":
            for subtopic_dir in topic_dir.iterdir():
                if subtopic_dir.is_dir():
                    for json_file in subtopic_dir.glob("*.json"):
                        if model_name.lower() in json_file.name.lower():
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                                topic = data['topic']
                                new_results[topic] = data
    
    # Load DeGroot baseline
    degroot_file = Path("results/degroot/test_topic/test_topic_50_50_no-llm_20250910_205014.json")
    with open(degroot_file, 'r') as f:
        degroot_data = json.load(f)
    
    # Calculate metrics for new model
    new_model_data = []
    for topic, llm_data in new_results.items():
        llm_final = np.array(llm_data['final_opinions'])
        degroot_final = np.array(degroot_data['final_opinions'])
        
        # Calculate metrics
        fixed_point_error = np.linalg.norm(llm_final - degroot_final)
        bias = np.mean(llm_final - degroot_final)
        llm_variance = np.var(llm_final)
        degroot_variance = np.var(degroot_final)
        llm_converged = np.std(llm_final) < 0.1
        degroot_converged = np.std(degroot_final) < 0.1
        
        new_model_data.append({
            'Model': model_name,
            'Topic': topic,
            'Bias': f"{bias:.4f}",
            'Fixed-Point Error': f"{fixed_point_error:.4f}",
            'LLM Mean': f"{np.mean(llm_final):.4f}",
            'DeGroot Mean': f"{np.mean(degroot_final):.4f}",
            'LLM Std': f"{np.std(llm_final):.4f}",
            'DeGroot Std': f"{np.std(degroot_final):.4f}",
            'LLM Converged': 'Yes' if llm_converged else 'No',
            'DeGroot Converged': 'Yes' if degroot_converged else 'No'
        })
    
    # Combine with existing results
    new_df = pd.DataFrame(new_model_data)
    
    if not existing_df.empty:
        # Add model column to existing data if it doesn't exist
        if 'Model' not in existing_df.columns:
            existing_df['Model'] = 'GPT-5-nano'
        
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # Save combined results
    combined_df.to_csv(output_file, index=False)
    
    print(f"Added {len(new_results)} topics for {model_name}")
    print(f"Combined results saved to {output_file}")
    
    # Print summary statistics
    print(f"\n{model_name} Summary:")
    print(f"Average Bias: {np.mean([float(row['Bias']) for row in new_model_data]):.4f}")
    print(f"Average Fixed-Point Error: {np.mean([float(row['Fixed-Point Error']) for row in new_model_data]):.4f}")
    
    return combined_df

def create_model_comparison_plot(combined_df, save_dir="figures"):
    """Create comparison plot across all models"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # Convert bias to numeric for plotting
    combined_df['Bias_Numeric'] = combined_df['Bias'].astype(float)
    combined_df['Error_Numeric'] = combined_df['Fixed-Point Error'].astype(float)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bias comparison
    sns.boxplot(data=combined_df, x='Model', y='Bias_Numeric', ax=ax1)
    ax1.set_title('Bias Comparison Across Models')
    ax1.set_ylabel('Bias (LLM - DeGroot)')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Error comparison
    sns.boxplot(data=combined_df, x='Model', y='Error_Numeric', ax=ax2)
    ax2.set_title('Fixed-Point Error Comparison Across Models')
    ax2.set_ylabel('Fixed-Point Error')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Model comparison plot saved to figures/model_comparison.png")

if __name__ == "__main__":
    print("Model Results Addition Script")
    print("=" * 40)
    print("To add results from a new model, run:")
    print("python add_model_results.py --model grok --results-dir results/grok")
    print("python add_model_results.py --model gpt-5 --results-dir results/gpt-5")
    print("\nOr use the functions directly:")
    print("add_model_results('grok', 'results/grok')")
    print("add_model_results('gpt-5', 'results/gpt-5')")
