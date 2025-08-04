"""
Bias testing scenarios for the simulation.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from ..simulation.controller import SimulationController
from ..llm.litellm_client import LiteLLMClient
from .analyzer import BiasAnalyzer


class BiasTestingScenarios:
    """
    Manages bias testing scenarios for the simulation.
    """
    
    def __init__(self, llm_client: Optional[LiteLLMClient] = None):
        """
        Initialize bias testing scenarios.
        
        Args:
            llm_client: LLM client for opinion generation
        """
        self.llm_client = llm_client
        self.analyzer = BiasAnalyzer()
    
    def run_single_topic_test(self, topic: str, n_agents: int, num_timesteps: int, use_llm: bool) -> Dict[str, Any]:
        """
        Run a single topic bias test.
        
        Args:
            topic: Topic to test
            n_agents: Number of agents
            num_timesteps: Number of timesteps
            use_llm: Whether to use LLM for opinion generation
            
        Returns:
            Dictionary containing test results
        """
        # Create simulation controller
        controller = SimulationController(
            n_agents=n_agents,
            num_timesteps=num_timesteps,
            llm_client=self.llm_client if use_llm else None,
            topics=[topic],
            initial_opinion_diversity=0.8
        )
        
        # Run simulation
        results = controller.run_simulation(progress_bar=False)
        
        # Extract metrics
        metrics = self._extract_simulation_metrics(controller)
        metrics['topic'] = topic
        
        return metrics
    
    def _extract_simulation_metrics(self, controller: SimulationController) -> Dict[str, Any]:
        """
        Extract metrics from simulation controller.
        
        Args:
            controller: Simulation controller
            
        Returns:
            Dictionary containing simulation metrics
        """
        # Get final opinion matrix
        final_opinions = controller._get_opinion_matrix()
        
        # Calculate metrics
        avg_opinion = np.mean(final_opinions)
        opinion_variance = np.var(final_opinions)
        
        # Calculate convergence speed
        opinion_history = controller.data_storage.get_opinion_history()
        convergence_speed = self._calculate_convergence_speed(opinion_history)
        
        # Get network metrics
        final_adjacency = controller.network.get_adjacency_matrix()
        network_density = np.sum(final_adjacency) / (len(final_adjacency) ** 2)
        
        return {
            'average_opinion': avg_opinion,
            'opinion_variance': opinion_variance,
            'convergence_speed': convergence_speed,
            'network_density': network_density,
            'final_opinions': final_opinions.tolist(),
            'opinion_history': [op.tolist() for op in opinion_history]
        }
    
    def _calculate_convergence_speed(self, opinion_history: List[np.ndarray]) -> float:
        """
        Calculate convergence speed from opinion history.
        
        Args:
            opinion_history: List of opinion matrices over time
            
        Returns:
            Convergence speed metric
        """
        if len(opinion_history) < 2:
            return 0.0
        
        # Calculate variance over time
        variances = []
        for opinions in opinion_history:
            if opinions is not None and len(opinions) > 0:
                variance = np.var(opinions)
                variances.append(variance)
            else:
                variances.append(0.0)
        
        if len(variances) < 2:
            return 0.0
        
        # Calculate convergence speed
        initial_variance = variances[0]
        final_variance = variances[-1]
        
        if initial_variance > 0:
            return (initial_variance - final_variance) / initial_variance
        else:
            return 0.0
    
    def run_bias_comparison(self, term1: str, term2: str, n_agents: int = 30, 
                           num_timesteps: int = 100, use_llm: bool = True,
                           use_consistent_seeds: bool = True) -> Dict[str, Any]:
        """
        Run bias comparison between two terms.
        
        Args:
            term1: First term to test
            term2: Second term to test
            n_agents: Number of agents
            num_timesteps: Number of timesteps
            use_llm: Whether to use LLM for opinion generation
            use_consistent_seeds: Whether to use consistent random seeds
            
        Returns:
            Dictionary containing comparison results
        """
        # Set random seeds for reproducibility
        if use_consistent_seeds:
            np.random.seed(42)
            seed1 = 42
            seed2 = 43
        else:
            seed1 = None
            seed2 = None
        
        # Run simulations for both terms
        result1 = self.run_single_topic_test(term1, n_agents, num_timesteps, use_llm)
        result2 = self.run_single_topic_test(term2, n_agents, num_timesteps, use_llm)
        
        # Calculate differences
        opinion_diff = abs(result1['average_opinion'] - result2['average_opinion'])
        variance_diff = abs(result1['opinion_variance'] - result2['opinion_variance'])
        convergence_diff = abs(result1['convergence_speed'] - result2['convergence_speed'])
        density_diff = abs(result1['network_density'] - result2['network_density'])
        
        # Determine if bias is detected
        bias_threshold = 0.1
        bias_indicators = [
            opinion_diff > bias_threshold,
            variance_diff > bias_threshold,
            convergence_diff > bias_threshold,
            density_diff > bias_threshold
        ]
        
        bias_detected = any(bias_indicators)
        bias_confidence = sum(bias_indicators) / len(bias_indicators)
        
        return {
            'term1': term1,
            'term2': term2,
            'result1': result1,
            'result2': result2,
            'differences': {
                'opinion_difference': opinion_diff,
                'variance_difference': variance_diff,
                'convergence_difference': convergence_diff,
                'density_difference': density_diff
            },
            'bias_detected': bias_detected,
            'bias_confidence': bias_confidence,
            'bias_threshold': bias_threshold,
            'use_consistent_seeds': use_consistent_seeds
        }
    
    def run_comprehensive_bias_test(self, topic_pairs: List[List[str]], n_agents: int = 30,
                                  num_timesteps: int = 100, use_llm: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive bias testing across multiple topic pairs.
        
        Args:
            topic_pairs: List of topic pairs to test
            n_agents: Number of agents
            num_timesteps: Number of timesteps
            use_llm: Whether to use LLM for opinion generation
            
        Returns:
            Dictionary containing comprehensive test results
        """
        results = []
        total_bias_detected = 0
        
        for term1, term2 in topic_pairs:
            comparison = self.run_bias_comparison(
                term1, term2, n_agents, num_timesteps, use_llm, use_consistent_seeds=True
            )
            results.append(comparison)
            
            if comparison['bias_detected']:
                total_bias_detected += 1
        
        # Calculate overall statistics
        total_tests = len(results)
        bias_rate = total_bias_detected / total_tests if total_tests > 0 else 0
        
        avg_confidence = np.mean([r['bias_confidence'] for r in results]) if results else 0
        
        return {
            'topic_pairs': topic_pairs,
            'individual_results': results,
            'summary': {
                'total_tests': total_tests,
                'bias_detected_count': total_bias_detected,
                'bias_rate': bias_rate,
                'average_confidence': avg_confidence
            },
            'parameters': {
                'n_agents': n_agents,
                'num_timesteps': num_timesteps,
                'use_llm': use_llm
            }
        }
    
    def generate_bias_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable bias report.
        
        Args:
            comparison_results: Results from bias comparison
            
        Returns:
            Formatted report string
        """
        report = "BIAS TESTING REPORT\n"
        report += "=" * 30 + "\n\n"
        
        # Basic information
        report += f"Term 1: {comparison_results['term1']}\n"
        report += f"Term 2: {comparison_results['term2']}\n"
        report += f"Bias detected: {comparison_results['bias_detected']}\n"
        report += f"Bias confidence: {comparison_results['bias_confidence']:.2f}\n\n"
        
        # Detailed differences
        differences = comparison_results['differences']
        report += "DIFFERENCES:\n"
        report += f"  Opinion difference: {differences['opinion_difference']:.4f}\n"
        report += f"  Variance difference: {differences['variance_difference']:.4f}\n"
        report += f"  Convergence difference: {differences['convergence_difference']:.4f}\n"
        report += f"  Density difference: {differences['density_difference']:.4f}\n\n"
        
        # Individual results
        result1 = comparison_results['result1']
        result2 = comparison_results['result2']
        
        report += f"RESULTS FOR '{comparison_results['term1']}':\n"
        report += f"  Average opinion: {result1['average_opinion']:.4f}\n"
        report += f"  Opinion variance: {result1['opinion_variance']:.4f}\n"
        report += f"  Convergence speed: {result1['convergence_speed']:.4f}\n"
        report += f"  Network density: {result1['network_density']:.4f}\n\n"
        
        report += f"RESULTS FOR '{comparison_results['term2']}':\n"
        report += f"  Average opinion: {result2['average_opinion']:.4f}\n"
        report += f"  Opinion variance: {result2['opinion_variance']:.4f}\n"
        report += f"  Convergence speed: {result2['convergence_speed']:.4f}\n"
        report += f"  Network density: {result2['network_density']:.4f}\n\n"
        
        return report 