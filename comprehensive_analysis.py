#!/usr/bin/env python3
"""
Comprehensive Analysis Script for LLM vs DeGroot Opinion Dynamics
Analyzes existing results and generates all figures for the paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OpinionDynamicsAnalyzer:
    def __init__(self, results_dir="results/degroot"):
        self.results_dir = Path(results_dir)
        self.llm_results = {}
        self.degroot_results = {}
        self.load_all_data()
        
    def load_all_data(self):
        """Load all LLM and DeGroot results"""
        print("Loading all results...")
        
        # Load pure DeGroot results
        degroot_file = self.results_dir / "test_topic" / "test_topic_50_50_no-llm_20250910_205014.json"
        with open(degroot_file, 'r') as f:
            self.degroot_results = json.load(f)
        
        # Load all LLM results
        for topic_dir in self.results_dir.iterdir():
            if topic_dir.is_dir() and topic_dir.name != "test_topic":
                for subtopic_dir in topic_dir.iterdir():
                    if subtopic_dir.is_dir():
                        for json_file in subtopic_dir.glob("*.json"):
                            if "no-llm" not in json_file.name:
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                    topic = data['topic']
                                    self.llm_results[topic] = data
                                    
        print(f"Loaded {len(self.llm_results)} LLM experiments")
        print(f"Topics: {list(self.llm_results.keys())}")
    
    def calculate_metrics(self, llm_opinions, degroot_opinions):
        """Calculate all key metrics"""
        llm_final = np.array(llm_opinions)
        degroot_final = np.array(degroot_opinions)
        
        # Fixed-point error (global fidelity)
        fixed_point_error = np.linalg.norm(llm_final - degroot_final)
        
        # Bias (mean signed difference)
        bias = np.mean(llm_final - degroot_final)
        
        # Variance analysis
        llm_variance = np.var(llm_final)
        degroot_variance = np.var(degroot_final)
        
        # Convergence analysis
        llm_converged = np.std(llm_final) < 0.1  # Low std indicates convergence
        degroot_converged = np.std(degroot_final) < 0.1
        
        return {
            'fixed_point_error': fixed_point_error,
            'bias': bias,
            'llm_variance': llm_variance,
            'degroot_variance': degroot_variance,
            'llm_converged': llm_converged,
            'degroot_converged': degroot_converged,
            'llm_mean': np.mean(llm_final),
            'degroot_mean': np.mean(degroot_final),
            'llm_std': np.std(llm_final),
            'degroot_std': np.std(degroot_final)
        }
    
    def analyze_all_topics(self):
        """Analyze all topics and return comprehensive results"""
        results = {}
        
        for topic, llm_data in self.llm_results.items():
            print(f"Analyzing topic: {topic}")
            
            # Get final opinions
            llm_final = llm_data['final_opinions']
            degroot_final = self.degroot_results['final_opinions']
            
            # Calculate metrics
            metrics = self.calculate_metrics(llm_final, degroot_final)
            
            # Add trajectory data if available
            if 'mean_opinions' in llm_data:
                metrics['llm_trajectory'] = llm_data['mean_opinions']
            if 'mean_opinions' in self.degroot_results:
                metrics['degroot_trajectory'] = self.degroot_results['mean_opinions']
            
            results[topic] = metrics
            
        return results
    
    def plot_trajectory_comparison(self, results, save_dir="figures"):
        """Plot trajectory comparisons for all topics"""
        Path(save_dir).mkdir(exist_ok=True)
        
        n_topics = len(results)
        fig, axes = plt.subplots(2, (n_topics + 1) // 2, figsize=(15, 10))
        if n_topics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (topic, data) in enumerate(results.items()):
            if 'llm_trajectory' in data and 'degroot_trajectory' in data:
                ax = axes[i]
                
                timesteps = range(len(data['llm_trajectory']))
                ax.plot(timesteps, data['llm_trajectory'], 'b-', label='LLM (GPT-5-nano)', linewidth=2)
                ax.plot(timesteps, data['degroot_trajectory'], 'r--', label='Pure DeGroot', linewidth=2)
                
                ax.set_title(f'{topic[:30]}...' if len(topic) > 30 else topic)
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Mean Opinion')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/trajectory_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_bias_analysis(self, results, save_dir="figures"):
        """Plot bias analysis across topics"""
        Path(save_dir).mkdir(exist_ok=True)
        
        topics = list(results.keys())
        biases = [results[topic]['bias'] for topic in topics]
        fixed_point_errors = [results[topic]['fixed_point_error'] for topic in topics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bias plot
        colors = ['red' if b > 0.1 else 'orange' if b > 0.05 else 'green' for b in biases]
        bars1 = ax1.bar(range(len(topics)), biases, color=colors, alpha=0.7)
        ax1.set_xlabel('Topic')
        ax1.set_ylabel('Bias (LLM - DeGroot)')
        ax1.set_title('Bias Analysis Across Topics')
        ax1.set_xticks(range(len(topics)))
        ax1.set_xticklabels([t[:15] + '...' if len(t) > 15 else t for t in topics], rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Add bias values on bars
        for i, (bar, bias) in enumerate(zip(bars1, biases)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{bias:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Fixed-point error plot
        bars2 = ax2.bar(range(len(topics)), fixed_point_errors, color='blue', alpha=0.7)
        ax2.set_xlabel('Topic')
        ax2.set_ylabel('Fixed-Point Error')
        ax2.set_title('Algorithmic Fidelity Across Topics')
        ax2.set_xticks(range(len(topics)))
        ax2.set_xticklabels([t[:15] + '...' if len(t) > 15 else t for t in topics], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add error values on bars
        for i, (bar, error) in enumerate(zip(bars2, fixed_point_errors)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{error:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_scatter_comparison(self, results, save_dir="figures"):
        """Plot scatter plot of LLM vs DeGroot final opinions"""
        Path(save_dir).mkdir(exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for topic, data in results.items():
            llm_mean = data['llm_mean']
            degroot_mean = data['degroot_mean']
            
            # Color by bias magnitude
            bias = abs(data['bias'])
            color = 'red' if bias > 0.2 else 'orange' if bias > 0.1 else 'green'
            
            ax.scatter(degroot_mean, llm_mean, s=100, alpha=0.7, 
                      color=color, label=topic[:20] + '...' if len(topic) > 20 else topic)
        
        # Perfect agreement line
        min_val = min(min(data['llm_mean'], data['degroot_mean']) for data in results.values())
        max_val = max(max(data['llm_mean'], data['degroot_mean']) for data in results.values())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Agreement')
        
        ax.set_xlabel('Pure DeGroot Final Opinion')
        ax.set_ylabel('LLM Final Opinion')
        ax.set_title('LLM vs DeGroot: Final Opinion Comparison')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/scatter_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence_analysis(self, results, save_dir="figures"):
        """Plot convergence analysis"""
        Path(save_dir).mkdir(exist_ok=True)
        
        topics = list(results.keys())
        llm_stds = [results[topic]['llm_std'] for topic in topics]
        degroot_stds = [results[topic]['degroot_std'] for topic in topics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(topics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, llm_stds, width, label='LLM (GPT-5-nano)', alpha=0.7)
        bars2 = ax.bar(x + width/2, degroot_stds, width, label='Pure DeGroot', alpha=0.7)
        
        ax.set_xlabel('Topic')
        ax.set_ylabel('Standard Deviation of Final Opinions')
        ax.set_title('Convergence Analysis: Opinion Variance')
        ax.set_xticks(x)
        ax.set_xticklabels([t[:15] + '...' if len(t) > 15 else t for t in topics], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_table(self, results, save_dir="figures"):
        """Generate summary statistics table"""
        Path(save_dir).mkdir(exist_ok=True)
        
        data = []
        for topic, metrics in results.items():
            data.append({
                'Topic': topic,
                'Bias': f"{metrics['bias']:.4f}",
                'Fixed-Point Error': f"{metrics['fixed_point_error']:.4f}",
                'LLM Mean': f"{metrics['llm_mean']:.4f}",
                'DeGroot Mean': f"{metrics['degroot_mean']:.4f}",
                'LLM Std': f"{metrics['llm_std']:.4f}",
                'DeGroot Std': f"{metrics['degroot_std']:.4f}",
                'LLM Converged': 'Yes' if metrics['llm_converged'] else 'No',
                'DeGroot Converged': 'Yes' if metrics['degroot_converged'] else 'No'
            })
        
        df = pd.DataFrame(data)
        df.to_csv(f'{save_dir}/summary_statistics.csv', index=False)
        
        # Create a nice table plot
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.title('Summary Statistics: LLM vs Pure DeGroot', fontsize=16, pad=20)
        plt.savefig(f'{save_dir}/summary_table.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def run_complete_analysis(self):
        """Run complete analysis and generate all figures"""
        print("Running comprehensive analysis...")
        
        # Analyze all topics
        results = self.analyze_all_topics()
        
        # Generate all figures
        print("Generating trajectory comparison...")
        self.plot_trajectory_comparison(results)
        
        print("Generating bias analysis...")
        self.plot_bias_analysis(results)
        
        print("Generating scatter comparison...")
        self.plot_scatter_comparison(results)
        
        print("Generating convergence analysis...")
        self.plot_convergence_analysis(results)
        
        print("Generating summary table...")
        df = self.generate_summary_table(results)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        
        total_topics = len(results)
        high_bias_topics = sum(1 for r in results.values() if abs(r['bias']) > 0.2)
        high_error_topics = sum(1 for r in results.values() if r['fixed_point_error'] > 1.0)
        
        print(f"Total topics analyzed: {total_topics}")
        print(f"Topics with high bias (>0.2): {high_bias_topics}")
        print(f"Topics with high error (>1.0): {high_error_topics}")
        print(f"Average bias: {np.mean([r['bias'] for r in results.values()]):.4f}")
        print(f"Average fixed-point error: {np.mean([r['fixed_point_error'] for r in results.values()]):.4f}")
        
        return results, df

if __name__ == "__main__":
    analyzer = OpinionDynamicsAnalyzer()
    results, summary_df = analyzer.run_complete_analysis()
    
    print("\nAnalysis complete! Check the 'figures' directory for all generated plots.")
    print("Summary statistics saved to 'figures/summary_statistics.csv'")
