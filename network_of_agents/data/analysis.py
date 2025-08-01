"""
Data analysis tools for simulation results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class SimulationAnalyzer:
    """
    Analyzes simulation results and generates insights.
    """
    
    def __init__(self, data_storage):
        """
        Initialize the analyzer with data storage.
        
        Args:
            data_storage: SimulationDataStorage instance
        """
        self.data_storage = data_storage
    
    def analyze_opinion_convergence(self) -> Dict[str, Any]:
        """
        Analyze opinion convergence patterns.
        
        Returns:
            Dictionary containing convergence analysis
        """
        opinion_history = self.data_storage.get_opinion_history()
        
        if not opinion_history or len(opinion_history) < 2:
            return {}
        
        # Calculate opinion variance over time
        variances = []
        for opinions in opinion_history:
            if opinions is not None:
                variance = np.var(opinions)
                variances.append(variance)
            else:
                variances.append(0)
        
        # Calculate convergence metrics
        initial_variance = variances[0]
        final_variance = variances[-1]
        
        # Convergence speed
        if initial_variance > 0:
            convergence_speed = (initial_variance - final_variance) / initial_variance
        else:
            convergence_speed = 0
        
        # Convergence time (when variance drops below 10% of initial)
        threshold = initial_variance * 0.1
        convergence_time = len(variances)
        for i, variance in enumerate(variances):
            if variance <= threshold:
                convergence_time = i
                break
        
        # Stability analysis
        if len(variances) > 10:
            final_stability = np.std(variances[-10:])
        else:
            final_stability = np.std(variances)
        
        return {
            'convergence_speed': convergence_speed,
            'convergence_time': convergence_time,
            'final_stability': final_stability,
            'initial_variance': initial_variance,
            'final_variance': final_variance,
            'variance_history': variances
        }
    
    def analyze_network_evolution(self) -> Dict[str, Any]:
        """
        Analyze network structure evolution.
        
        Returns:
            Dictionary containing network evolution analysis
        """
        metrics_history = self.data_storage.get_metrics_history()
        
        if not metrics_history:
            return {}
        
        # Extract metrics over time
        timesteps = list(range(len(metrics_history)))
        densities = [m.get('density', 0) if m else 0 for m in metrics_history]
        clustering_coeffs = [m.get('clustering_coefficient', 0) if m else 0 for m in metrics_history]
        num_components = [m.get('num_components', 0) if m else 0 for m in metrics_history]
        echo_chambers = [m.get('echo_chambers', 0) if m else 0 for m in metrics_history]
        
        # Calculate network evolution metrics
        density_change = densities[-1] - densities[0]
        clustering_change = clustering_coeffs[-1] - clustering_coeffs[0]
        component_change = num_components[-1] - num_components[0]
        
        # Network stability
        density_stability = np.std(densities[-10:]) if len(densities) >= 10 else np.std(densities)
        
        return {
            'density_evolution': densities,
            'clustering_evolution': clustering_coeffs,
            'component_evolution': num_components,
            'echo_chamber_evolution': echo_chambers,
            'density_change': density_change,
            'clustering_change': clustering_change,
            'component_change': component_change,
            'density_stability': density_stability,
            'timesteps': timesteps
        }
    
    def analyze_agent_behavior(self) -> Dict[str, Any]:
        """
        Analyze individual agent behavior patterns.
        
        Returns:
            Dictionary containing agent behavior analysis
        """
        agent_data_history = self.data_storage.get_agent_data_history()
        
        if not agent_data_history:
            return {}
        
        # Analyze final agent states
        final_agent_data = agent_data_history[-1]
        if not final_agent_data:
            return {}
        
        # Extract agent metrics
        agent_degrees = [agent.get('degree', 0) for agent in final_agent_data]
        agent_opinions = [agent.get('opinions', []) for agent in final_agent_data]
        
        # Calculate agent statistics
        degree_stats = {
            'mean_degree': np.mean(agent_degrees),
            'std_degree': np.std(agent_degrees),
            'max_degree': np.max(agent_degrees),
            'min_degree': np.min(agent_degrees),
            'degree_distribution': agent_degrees
        }
        
        # Analyze opinion patterns
        if agent_opinions and len(agent_opinions[0]) > 0:
            opinion_matrix = np.array(agent_opinions)
            opinion_stats = {
                'mean_opinions': np.mean(opinion_matrix, axis=0).tolist(),
                'std_opinions': np.std(opinion_matrix, axis=0).tolist(),
                'opinion_correlations': np.corrcoef(opinion_matrix.T).tolist()
            }
        else:
            opinion_stats = {}
        
        return {
            'degree_statistics': degree_stats,
            'opinion_statistics': opinion_stats,
            'n_agents': len(final_agent_data)
        }
    
    def detect_echo_chambers(self, similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Detect echo chambers in the final network state.
        
        Args:
            similarity_threshold: Threshold for echo chamber detection
            
        Returns:
            Dictionary containing echo chamber analysis
        """
        adjacency_history = self.data_storage.get_adjacency_history()
        opinion_history = self.data_storage.get_opinion_history()
        
        if not adjacency_history or not opinion_history:
            return {}
        
        # Get final state
        final_adjacency = adjacency_history[-1]
        final_opinions = opinion_history[-1]
        
        if final_adjacency is None or final_opinions is None:
            return {}
        
        # Calculate opinion similarity matrix
        n_agents = len(final_opinions)
        similarity_matrix = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    opinion_diff = np.linalg.norm(final_opinions[i] - final_opinions[j], ord=1)
                    max_possible_diff = len(final_opinions[i])
                    similarity = 1 - (opinion_diff / max_possible_diff)
                    similarity_matrix[i, j] = max(0, similarity)
        
        # Find echo chambers
        import networkx as nx
        echo_chamber_graph = (final_adjacency > 0) & (similarity_matrix > similarity_threshold)
        G = nx.from_numpy_array(echo_chamber_graph.astype(int))
        components = list(nx.connected_components(G))
        
        # Filter out single-node components
        echo_chambers = [list(comp) for comp in components if len(comp) > 1]
        
        # Analyze echo chambers
        chamber_sizes = [len(chamber) for chamber in echo_chambers]
        chamber_opinions = []
        
        for chamber in echo_chambers:
            chamber_opinion = np.mean([final_opinions[i] for i in chamber], axis=0)
            chamber_opinions.append(chamber_opinion)
        
        return {
            'echo_chambers': echo_chambers,
            'chamber_sizes': chamber_sizes,
            'chamber_opinions': [op.tolist() for op in chamber_opinions],
            'num_echo_chambers': len(echo_chambers),
            'total_echo_chamber_agents': sum(chamber_sizes),
            'avg_chamber_size': np.mean(chamber_sizes) if chamber_sizes else 0,
            'max_chamber_size': max(chamber_sizes) if chamber_sizes else 0
        }
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report.
        
        Returns:
            Formatted summary report
        """
        report = "SIMULATION SUMMARY REPORT\n"
        report += "=" * 40 + "\n\n"
        
        # Basic statistics
        summary_stats = self.data_storage.get_summary_statistics()
        report += f"Total timesteps: {summary_stats.get('total_timesteps', 0)}\n"
        report += f"Number of agents: {summary_stats.get('n_agents', 0)}\n"
        report += f"Number of topics: {summary_stats.get('n_topics', 0)}\n\n"
        
        # Convergence analysis
        convergence_analysis = self.analyze_opinion_convergence()
        if convergence_analysis:
            report += "OPINION CONVERGENCE:\n"
            report += f"  Convergence speed: {convergence_analysis.get('convergence_speed', 0):.4f}\n"
            report += f"  Convergence time: {convergence_analysis.get('convergence_time', 0)} timesteps\n"
            report += f"  Final stability: {convergence_analysis.get('final_stability', 0):.4f}\n\n"
        
        # Network evolution
        network_analysis = self.analyze_network_evolution()
        if network_analysis:
            report += "NETWORK EVOLUTION:\n"
            report += f"  Density change: {network_analysis.get('density_change', 0):.4f}\n"
            report += f"  Clustering change: {network_analysis.get('clustering_change', 0):.4f}\n"
            report += f"  Component change: {network_analysis.get('component_change', 0):.0f}\n\n"
        
        # Echo chambers
        echo_chamber_analysis = self.detect_echo_chambers()
        if echo_chamber_analysis:
            report += "ECHO CHAMBERS:\n"
            report += f"  Number of echo chambers: {echo_chamber_analysis.get('num_echo_chambers', 0)}\n"
            report += f"  Total agents in echo chambers: {echo_chamber_analysis.get('total_echo_chamber_agents', 0)}\n"
            report += f"  Average chamber size: {echo_chamber_analysis.get('avg_chamber_size', 0):.2f}\n"
            report += f"  Maximum chamber size: {echo_chamber_analysis.get('max_chamber_size', 0)}\n\n"
        
        # Agent behavior
        agent_analysis = self.analyze_agent_behavior()
        if agent_analysis:
            degree_stats = agent_analysis.get('degree_statistics', {})
            report += "AGENT BEHAVIOR:\n"
            report += f"  Average degree: {degree_stats.get('mean_degree', 0):.2f}\n"
            report += f"  Degree standard deviation: {degree_stats.get('std_degree', 0):.2f}\n"
            report += f"  Maximum degree: {degree_stats.get('max_degree', 0)}\n"
            report += f"  Minimum degree: {degree_stats.get('min_degree', 0)}\n\n"
        
        return report
    
    def create_visualization_plots(self, output_dir: str = "analysis_plots"):
        """
        Create and save visualization plots.
        
        Args:
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Opinion convergence plot
        convergence_analysis = self.analyze_opinion_convergence()
        if convergence_analysis and 'variance_history' in convergence_analysis:
            plt.figure(figsize=(10, 6))
            plt.plot(convergence_analysis['variance_history'])
            plt.title('Opinion Variance Over Time')
            plt.xlabel('Timestep')
            plt.ylabel('Opinion Variance')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'opinion_convergence.png'))
            plt.close()
        
        # Network evolution plot
        network_analysis = self.analyze_network_evolution()
        if network_analysis:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            axes[0, 0].plot(network_analysis['density_evolution'])
            axes[0, 0].set_title('Network Density')
            axes[0, 0].set_xlabel('Timestep')
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(network_analysis['clustering_evolution'])
            axes[0, 1].set_title('Clustering Coefficient')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].grid(True)
            
            axes[1, 0].plot(network_analysis['component_evolution'])
            axes[1, 0].set_title('Number of Components')
            axes[1, 0].set_xlabel('Timestep')
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(network_analysis['echo_chamber_evolution'])
            axes[1, 1].set_title('Number of Echo Chambers')
            axes[1, 1].set_xlabel('Timestep')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'network_evolution.png'))
            plt.close()
        
        # Agent degree distribution
        agent_analysis = self.analyze_agent_behavior()
        if agent_analysis and 'degree_statistics' in agent_analysis:
            degree_dist = agent_analysis['degree_statistics']['degree_distribution']
            plt.figure(figsize=(8, 6))
            plt.hist(degree_dist, bins=20, alpha=0.7, edgecolor='black')
            plt.title('Agent Degree Distribution')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'degree_distribution.png'))
            plt.close()
    
    def export_analysis_data(self, output_dir: str = "analysis_data"):
        """
        Export analysis data to CSV files.
        
        Args:
            output_dir: Directory to save data files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export convergence data
        convergence_analysis = self.analyze_opinion_convergence()
        if convergence_analysis and 'variance_history' in convergence_analysis:
            convergence_df = pd.DataFrame({
                'timestep': list(range(len(convergence_analysis['variance_history']))),
                'opinion_variance': convergence_analysis['variance_history']
            })
            convergence_df.to_csv(os.path.join(output_dir, 'convergence_data.csv'), index=False)
        
        # Export network evolution data
        network_analysis = self.analyze_network_evolution()
        if network_analysis:
            network_df = pd.DataFrame({
                'timestep': network_analysis['timesteps'],
                'density': network_analysis['density_evolution'],
                'clustering_coefficient': network_analysis['clustering_evolution'],
                'num_components': network_analysis['component_evolution'],
                'num_echo_chambers': network_analysis['echo_chamber_evolution']
            })
            network_df.to_csv(os.path.join(output_dir, 'network_evolution.csv'), index=False)
        
        # Export agent data
        agent_analysis = self.analyze_agent_behavior()
        if agent_analysis and 'degree_statistics' in agent_analysis:
            degree_dist = agent_analysis['degree_statistics']['degree_distribution']
            agent_df = pd.DataFrame({
                'agent_id': list(range(len(degree_dist))),
                'degree': degree_dist
            })
            agent_df.to_csv(os.path.join(output_dir, 'agent_degrees.csv'), index=False) 