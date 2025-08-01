"""
Bias testing scenarios for detecting bias patterns in opinion convergence.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ..simulation.controller import SimulationController
from ..llm.litellm_client import LiteLLMClient
from ..config.config_manager import ConfigManager


class BiasTestingScenarios:
    """
    Manages bias testing scenarios for the simulation.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize bias testing scenarios.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.bias_config = config_manager.get_bias_testing_config()
    
    def run_all_tests(self, n_agents: int = 50, num_timesteps: int = 180, use_llm: bool = True) -> Dict[str, Any]:
        """
        Run all bias testing scenarios.
        
        Args:
            n_agents: Number of agents for each test
            num_timesteps: Number of timesteps for each test
            use_llm: Whether to use LLM integration
            
        Returns:
            Dictionary containing results for all tests
        """
        results = {}
        
        # Run language bias tests
        language_bias_results = self.run_language_bias_tests(n_agents, num_timesteps, use_llm)
        results.update(language_bias_results)
        
        # Run framing bias tests
        framing_bias_results = self.run_framing_bias_tests(n_agents, num_timesteps, use_llm)
        results.update(framing_bias_results)
        
        # Run neutral vs controversial tests
        neutral_results = self.run_neutral_vs_controversial_tests(n_agents, num_timesteps, use_llm)
        results.update(neutral_results)
        
        return results
    
    def run_language_bias_tests(self, n_agents: int, num_timesteps: int, use_llm: bool) -> Dict[str, Any]:
        """
        Run language bias tests comparing different word choices.
        
        Args:
            n_agents: Number of agents
            num_timesteps: Number of timesteps
            use_llm: Whether to use LLM integration
            
        Returns:
            Dictionary containing language bias test results
        """
        results = {}
        topic_pairs = self.config_manager.get_topic_pairs("language_bias")
        
        for i, (term1, term2) in enumerate(topic_pairs):
            test_name = f"language_bias_{i+1}_{term1}_vs_{term2}"
            print(f"Running language bias test: {term1} vs {term2}")
            
            # Run simulation with term1
            result1 = self._run_single_topic_test(term1, n_agents, num_timesteps, use_llm)
            
            # Run simulation with term2
            result2 = self._run_single_topic_test(term2, n_agents, num_timesteps, use_llm)
            
            # Compare results
            comparison = self._compare_simulation_results(result1, result2)
            comparison['term1'] = term1
            comparison['term2'] = term2
            
            results[test_name] = comparison
        
        return results
    
    def run_framing_bias_tests(self, n_agents: int, num_timesteps: int, use_llm: bool) -> Dict[str, Any]:
        """
        Run framing bias tests comparing different issue framings.
        
        Args:
            n_agents: Number of agents
            num_timesteps: Number of timesteps
            use_llm: Whether to use LLM integration
            
        Returns:
            Dictionary containing framing bias test results
        """
        results = {}
        topic_pairs = self.config_manager.get_topic_pairs("framing_bias")
        
        for i, (frame1, frame2) in enumerate(topic_pairs):
            test_name = f"framing_bias_{i+1}_{frame1}_vs_{frame2}"
            print(f"Running framing bias test: {frame1} vs {frame2}")
            
            # Run simulation with frame1
            result1 = self._run_single_topic_test(frame1, n_agents, num_timesteps, use_llm)
            
            # Run simulation with frame2
            result2 = self._run_single_topic_test(frame2, n_agents, num_timesteps, use_llm)
            
            # Compare results
            comparison = self._compare_simulation_results(result1, result2)
            comparison['frame1'] = frame1
            comparison['frame2'] = frame2
            
            results[test_name] = comparison
        
        return results
    
    def run_neutral_vs_controversial_tests(self, n_agents: int, num_timesteps: int, use_llm: bool) -> Dict[str, Any]:
        """
        Run tests comparing neutral topics vs controversial topics.
        
        Args:
            n_agents: Number of agents
            num_timesteps: Number of timesteps
            use_llm: Whether to use LLM integration
            
        Returns:
            Dictionary containing neutral vs controversial test results
        """
        results = {}
        
        # Get neutral and controversial topics
        neutral_topics = self.config_manager.get_neutral_topics()
        political_topics = self.config_manager.get_political_topics()
        
        # Test neutral topics
        neutral_results = {}
        for i, topic in enumerate(neutral_topics[:3]):  # Test first 3 neutral topics
            test_name = f"neutral_topic_{i+1}_{topic}"
            print(f"Running neutral topic test: {topic}")
            
            result = self._run_single_topic_test(topic, n_agents, num_timesteps, use_llm)
            neutral_results[test_name] = result
        
        # Test controversial topics
        controversial_results = {}
        for i, topic in enumerate(political_topics[:3]):  # Test first 3 political topics
            test_name = f"controversial_topic_{i+1}_{topic}"
            print(f"Running controversial topic test: {topic}")
            
            result = self._run_single_topic_test(topic, n_agents, num_timesteps, use_llm)
            controversial_results[test_name] = result
        
        # Compare neutral vs controversial
        neutral_avg = self._calculate_average_metrics(neutral_results)
        controversial_avg = self._calculate_average_metrics(controversial_results)
        
        comparison = {
            'neutral_metrics': neutral_avg,
            'controversial_metrics': controversial_avg,
            'convergence_speed_diff': neutral_avg['convergence_speed'] - controversial_avg['convergence_speed'],
            'final_opinion_variance_diff': neutral_avg['final_opinion_variance'] - controversial_avg['final_opinion_variance'],
            'echo_chamber_diff': controversial_avg['echo_chambers'] - neutral_avg['echo_chambers'],
            'bias_detected': abs(neutral_avg['convergence_speed'] - controversial_avg['convergence_speed']) > 0.1
        }
        
        results['neutral_vs_controversial'] = comparison
        results.update(neutral_results)
        results.update(controversial_results)
        
        return results
    
    def _run_single_topic_test(self, topic: str, n_agents: int, num_timesteps: int, use_llm: bool) -> Dict[str, Any]:
        """
        Run a single topic test.
        
        Args:
            topic: Topic to test
            n_agents: Number of agents
            num_timesteps: Number of timesteps
            use_llm: Whether to use LLM integration
            
        Returns:
            Dictionary containing test results
        """
        # Initialize LLM client if needed
        llm_client = None
        if use_llm:
            try:
                llm_config = self.config_manager.get_llm_config()
                llm_client = LiteLLMClient(
                    model_name=llm_config.get("model_name", "gpt-4"),
                    api_key=llm_config.get("api_key_env_var", "OPENAI_API_KEY")
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLM client: {e}")
                use_llm = False
        
        # Get simulation configuration
        sim_config = self.config_manager.get_simulation_config()
        
        # Create simulation controller
        controller = SimulationController(
            n_agents=n_agents,
            n_topics=1,  # Single topic test
            epsilon=sim_config.get("epsilon", 1e-6),
            theta=sim_config.get("theta", 7),
            num_timesteps=num_timesteps,
            initial_connection_probability=sim_config.get("initial_connection_probability", 0.2),
            llm_client=llm_client,
            topics=[topic]
        )
        
        # Run simulation
        results = controller.run_simulation(progress_bar=False)
        
        # Extract metrics
        metrics = self._extract_simulation_metrics(controller)
        metrics['topic'] = topic
        metrics['use_llm'] = use_llm
        
        return metrics
    
    def _extract_simulation_metrics(self, controller: SimulationController) -> Dict[str, Any]:
        """
        Extract metrics from simulation results.
        
        Args:
            controller: Simulation controller instance
            
        Returns:
            Dictionary containing extracted metrics
        """
        # Get opinion history
        opinion_history = controller.data_storage.get_opinion_history()
        
        if not opinion_history or len(opinion_history) < 2:
            return {}
        
        # Calculate convergence speed
        convergence_speed = self._calculate_convergence_speed(opinion_history)
        
        # Calculate final opinion variance
        final_opinions = opinion_history[-1]
        final_opinion_variance = np.var(final_opinions) if final_opinions is not None else 0
        
        # Get network metrics
        metrics_history = controller.data_storage.get_metrics_history()
        final_metrics = metrics_history[-1] if metrics_history else {}
        
        # Get echo chambers
        echo_chambers = controller.network.get_echo_chambers()
        num_echo_chambers = len(echo_chambers)
        
        return {
            'convergence_speed': convergence_speed,
            'final_opinion_variance': final_opinion_variance,
            'final_network_density': final_metrics.get('density', 0),
            'final_clustering_coefficient': final_metrics.get('clustering_coefficient', 0),
            'echo_chambers': num_echo_chambers,
            'total_timesteps': len(opinion_history)
        }
    
    def _calculate_convergence_speed(self, opinion_history: List[np.ndarray]) -> float:
        """
        Calculate the speed of opinion convergence.
        
        Args:
            opinion_history: List of opinion matrices over time
            
        Returns:
            Convergence speed metric
        """
        if len(opinion_history) < 2:
            return 0.0
        
        # Calculate opinion variance over time
        variances = []
        for opinions in opinion_history:
            if opinions is not None:
                variance = np.var(opinions)
                variances.append(variance)
            else:
                variances.append(0)
        
        # Calculate rate of variance reduction
        if len(variances) > 1:
            initial_variance = variances[0]
            final_variance = variances[-1]
            
            if initial_variance > 0:
                # Normalized convergence speed (0 to 1)
                convergence_speed = (initial_variance - final_variance) / initial_variance
                return max(0.0, min(1.0, convergence_speed))
        
        return 0.0
    
    def _compare_simulation_results(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two simulation results.
        
        Args:
            result1: First simulation result
            result2: Second simulation result
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison = {}
        
        # Compare convergence speed
        convergence_speed_diff = result1.get('convergence_speed', 0) - result2.get('convergence_speed', 0)
        comparison['convergence_speed_diff'] = convergence_speed_diff
        
        # Compare final opinion variance
        final_opinion_diff = result1.get('final_opinion_variance', 0) - result2.get('final_opinion_variance', 0)
        comparison['final_opinion_diff'] = final_opinion_diff
        
        # Compare echo chambers
        echo_chamber_diff = result1.get('echo_chambers', 0) - result2.get('echo_chambers', 0)
        comparison['echo_chamber_diff'] = echo_chamber_diff
        
        # Determine if bias is detected
        bias_threshold = self.bias_config.get('convergence_threshold', 0.01)
        bias_detected = abs(convergence_speed_diff) > bias_threshold
        
        comparison['bias_detected'] = bias_detected
        comparison['bias_magnitude'] = abs(convergence_speed_diff)
        
        # Store individual results
        comparison['result1'] = result1
        comparison['result2'] = result2
        
        return comparison
    
    def _calculate_average_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate average metrics across multiple test results.
        
        Args:
            results: Dictionary of test results
            
        Returns:
            Dictionary containing average metrics
        """
        metrics = ['convergence_speed', 'final_opinion_variance', 'echo_chambers']
        averages = {}
        
        for metric in metrics:
            values = [result.get(metric, 0) for result in results.values() if isinstance(result, dict)]
            if values:
                averages[metric] = np.mean(values)
            else:
                averages[metric] = 0.0
        
        return averages
    
    def save_results(self, filename: str, results: Dict[str, Any]):
        """
        Save bias testing results to a file.
        
        Args:
            filename: Name of the file to save to
            results: Results to save
        """
        # Create directory if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """
        Load bias testing results from a file.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            Loaded results
        """
        with open(filename, 'r') as f:
            return json.load(f)
    
    def generate_bias_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable bias report.
        
        Args:
            results: Bias testing results
            
        Returns:
            Formatted report string
        """
        report = "BIAS TESTING REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Language bias results
        language_tests = {k: v for k, v in results.items() if k.startswith('language_bias')}
        if language_tests:
            report += "LANGUAGE BIAS TESTS:\n"
            report += "-" * 20 + "\n"
            for test_name, result in language_tests.items():
                report += f"Test: {result.get('term1', '')} vs {result.get('term2', '')}\n"
                report += f"  Convergence speed difference: {result.get('convergence_speed_diff', 0):.4f}\n"
                report += f"  Bias detected: {result.get('bias_detected', False)}\n"
                report += f"  Bias magnitude: {result.get('bias_magnitude', 0):.4f}\n\n"
        
        # Framing bias results
        framing_tests = {k: v for k, v in results.items() if k.startswith('framing_bias')}
        if framing_tests:
            report += "FRAMING BIAS TESTS:\n"
            report += "-" * 18 + "\n"
            for test_name, result in framing_tests.items():
                report += f"Test: {result.get('frame1', '')} vs {result.get('frame2', '')}\n"
                report += f"  Convergence speed difference: {result.get('convergence_speed_diff', 0):.4f}\n"
                report += f"  Bias detected: {result.get('bias_detected', False)}\n"
                report += f"  Bias magnitude: {result.get('bias_magnitude', 0):.4f}\n\n"
        
        # Neutral vs controversial results
        if 'neutral_vs_controversial' in results:
            result = results['neutral_vs_controversial']
            report += "NEUTRAL VS CONTROVERSIAL TOPICS:\n"
            report += "-" * 32 + "\n"
            report += f"Convergence speed difference: {result.get('convergence_speed_diff', 0):.4f}\n"
            report += f"Echo chamber difference: {result.get('echo_chamber_diff', 0):.4f}\n"
            report += f"Bias detected: {result.get('bias_detected', False)}\n\n"
        
        # Summary
        total_tests = len([k for k in results.keys() if k.startswith(('language_bias', 'framing_bias'))])
        bias_detected_count = len([v for v in results.values() if isinstance(v, dict) and v.get('bias_detected', False)])
        
        report += "SUMMARY:\n"
        report += "-" * 8 + "\n"
        report += f"Total tests run: {total_tests}\n"
        report += f"Tests with bias detected: {bias_detected_count}\n"
        report += f"Bias detection rate: {bias_detected_count/total_tests*100:.1f}%\n"
        
        return report 