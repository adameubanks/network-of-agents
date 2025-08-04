"""
Data analysis for simulation results.
"""

import numpy as np
from typing import Dict, List, Any
from .storage import DataStorage


class SimulationAnalyzer:
    """
    Analyzes simulation results and generates insights.
    """
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize analyzer.
        
        Args:
            data_storage: Data storage containing simulation results
        """
        self.data_storage = data_storage
    
    def analyze_network_evolution(self) -> Dict[str, Any]:
        """
        Analyze network evolution over time.
        
        Returns:
            Dictionary containing network evolution analysis
        """
        adjacency_history = self.data_storage.get_adjacency_history()
        metrics_history = self.data_storage.get_metrics_history()
        
        if not adjacency_history or not metrics_history:
            return {}
        
        # Calculate network evolution metrics
        density_history = []
        clustering_history = []
        components_history = []
        
        for metrics in metrics_history:
            if metrics is not None:
                density_history.append(metrics.get('density', 0.0))
                clustering_history.append(metrics.get('clustering_coefficient', 0.0))
                components_history.append(metrics.get('num_components', 1))
            else:
                density_history.append(0.0)
                clustering_history.append(0.0)
                components_history.append(1)
        
        return {
            'density_history': density_history,
            'clustering_history': clustering_history,
            'components_history': components_history,
            'final_density': density_history[-1] if density_history else 0.0,
            'final_clustering': clustering_history[-1] if clustering_history else 0.0,
            'final_components': components_history[-1] if components_history else 1
        }
    
    def analyze_opinion_evolution(self) -> Dict[str, Any]:
        """
        Analyze opinion evolution over time.
        
        Returns:
            Dictionary containing opinion evolution analysis
        """
        opinion_history = self.data_storage.get_opinion_history()
        
        if not opinion_history or len(opinion_history) < 2:
            return {}
        
        # Calculate opinion evolution metrics
        mean_opinions = []
        std_opinions = []
        variance_history = []
        
        for opinions in opinion_history:
            if opinions is not None:
                mean_opinions.append(np.mean(opinions))
                std_opinions.append(np.std(opinions))
                variance_history.append(np.var(opinions))
            else:
                mean_opinions.append(0.0)
                std_opinions.append(0.0)
                variance_history.append(0.0)
        
        # Calculate convergence metrics
        initial_variance = variance_history[0] if variance_history else 0.0
        final_variance = variance_history[-1] if variance_history else 0.0
        convergence_speed = (initial_variance - final_variance) / len(variance_history) if len(variance_history) > 1 else 0.0
        
        return {
            'mean_opinions': mean_opinions,
            'std_opinions': std_opinions,
            'variance_history': variance_history,
            'initial_variance': initial_variance,
            'final_variance': final_variance,
            'convergence_speed': convergence_speed,
            'convergence_achieved': final_variance < 0.01
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
            opinion_array = np.array([op[0] for op in agent_opinions])  # Single topic
            opinion_stats = {
                'mean_opinion': np.mean(opinion_array),
                'std_opinion': np.std(opinion_array),
                'opinion_range': [np.min(opinion_array), np.max(opinion_array)]
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
        Detect echo chambers in the network.
        
        Args:
            similarity_threshold: Threshold for considering agents similar
            
        Returns:
            Dictionary containing echo chamber analysis
        """
        opinion_history = self.data_storage.get_opinion_history()
        adjacency_history = self.data_storage.get_adjacency_history()
        
        if not opinion_history or not adjacency_history:
            return {}
        
        # Analyze final state for echo chambers
        final_opinions = opinion_history[-1]
        final_adjacency = adjacency_history[-1]
        
        if final_opinions is None or final_adjacency is None:
            return {}
        
        # Find connected components
        n_agents = len(final_opinions)
        visited = [False] * n_agents
        components = []
        
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in range(n_agents):
                if final_adjacency[node, neighbor] == 1 and not visited[neighbor]:
                    dfs(neighbor, component)
        
        for i in range(n_agents):
            if not visited[i]:
                component = []
                dfs(i, component)
                if len(component) > 1:  # Only consider components with multiple agents
                    components.append(component)
        
        # Analyze each component for opinion homogeneity
        echo_chambers = []
        for component in components:
            component_opinions = [final_opinions[i] for i in component]
            opinion_variance = np.var(component_opinions)
            
            # Consider it an echo chamber if opinions are very similar
            if opinion_variance < 0.01:  # Low variance indicates echo chamber
                echo_chambers.append({
                    'agents': component,
                    'size': len(component),
                    'mean_opinion': np.mean(component_opinions),
                    'opinion_variance': opinion_variance
                })
        
        return {
            'echo_chambers': echo_chambers,
            'num_echo_chambers': len(echo_chambers),
            'total_agents_in_echo_chambers': sum(ec['size'] for ec in echo_chambers),
            'components': components,
            'num_components': len(components)
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Dictionary containing comprehensive analysis
        """
        network_analysis = self.analyze_network_evolution()
        opinion_analysis = self.analyze_opinion_evolution()
        agent_analysis = self.analyze_agent_behavior()
        echo_chamber_analysis = self.detect_echo_chambers()
        
        # Get summary statistics
        summary_stats = self.data_storage.get_summary_statistics()
        
        # Combine all analyses
        report = {
            'summary_statistics': summary_stats,
            'network_evolution': network_analysis,
            'opinion_evolution': opinion_analysis,
            'agent_behavior': agent_analysis,
            'echo_chambers': echo_chamber_analysis
        }
        
        # Add overall assessment
        convergence_achieved = opinion_analysis.get('convergence_achieved', False)
        final_variance = opinion_analysis.get('final_variance', 1.0)
        num_echo_chambers = echo_chamber_analysis.get('num_echo_chambers', 0)
        
        report['overall_assessment'] = {
            'opinion_convergence': convergence_achieved,
            'opinion_diversity': 'high' if final_variance > 0.1 else 'medium' if final_variance > 0.01 else 'low',
            'echo_chamber_formation': num_echo_chambers > 0,
            'network_stability': network_analysis.get('final_density', 0.0) > 0.1
        }
        
        return report
    
    def export_analysis_report(self, filename: str):
        """
        Export analysis report to file.
        
        Args:
            filename: Output filename
        """
        report = self.generate_comprehensive_report()
        
        import json
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
    
    def print_summary(self):
        """Print a summary of the analysis."""
        report = self.generate_comprehensive_report()
        
        print("=== SIMULATION ANALYSIS SUMMARY ===")
        print(f"Number of agents: {report['summary_statistics'].get('n_agents', 0)}")
        print(f"Number of timesteps: {report['summary_statistics'].get('num_timesteps', 0)}")
        print(f"Initial mean opinion: {report['summary_statistics'].get('initial_mean_opinion', 0):.4f}")
        print(f"Final mean opinion: {report['summary_statistics'].get('final_mean_opinion', 0):.4f}")
        print(f"Opinion convergence: {report['overall_assessment'].get('opinion_convergence', False)}")
        print(f"Echo chambers detected: {report['echo_chambers'].get('num_echo_chambers', 0)}")
        print(f"Final network density: {report['network_evolution'].get('final_density', 0):.4f}")
        print("==================================") 