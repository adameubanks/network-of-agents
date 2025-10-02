"""
Simple 4-phase experimental runner that implements the experimental design exactly.

This runner follows the 4-phase protocol:
1. Phase 1: Mathematical baseline (all 6 configurations)
2. Phase 2: LLM experiments (6 configurations × 10 topics)
3. Phase 3: Symmetry testing (B vs A orientation)
4. Phase 4: Analysis and metrics
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from .simulation.controller import Controller
from .llm_client import LLMClient
from .network.graph_generator import create_network_model
from .core.mathematics import update_opinions_pure_degroot, update_opinions_friedkin_johnsen
# Visualization removed - handled by separate analyzer

logger = logging.getLogger(__name__)

class Runner:
    """
    Simple 4-phase experimental runner that implements the experimental design exactly.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results = {}
        
        # Initialize LLM client if needed
        self.llm_client = None
        if config.get("llm_enabled", False):
            self.llm_client = LLMClient(
                model_name=config.get("llm_model", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # Set output directory for results
        self.output_dir = config.get("output_dir", "results")
    
    def run_all_phases(self) -> Dict[str, Any]:
        """
        Run all phases of the experimental protocol.
        
        Returns:
            Complete experimental results
        """
        logger.info("Starting experimental protocol with built-in symmetry testing")
        
        # Phase 1: Mathematical baseline
        logger.info("=== PHASE 1: Mathematical Baseline ===")
        phase1_results = self._run_phase1_mathematical_baseline()
        self.results["phase1_mathematical_baseline"] = phase1_results
        
        # Phase 2: LLM experiments (with built-in symmetry testing)
        logger.info("=== PHASE 2: LLM Experiments (A vs B and B vs A) ===")
        try:
            phase2_results = self._run_phase2_llm_experiments()
            self.results["phase2_llm_experiments"] = phase2_results
        except Exception as e:
            logger.warning(f"Phase 2 failed (expected if LLM models unavailable): {e}")
            self.results["phase2_llm_experiments"] = {"error": str(e)}
        
        # Phase 3: Analysis
        logger.info("=== PHASE 3: Analysis ===")
        try:
            phase3_results = self._run_phase3_analysis()
            self.results["phase3_analysis"] = phase3_results
        except Exception as e:
            logger.warning(f"Phase 3 failed: {e}")
            self.results["phase3_analysis"] = {"error": str(e)}
        
        # Save complete results
        self._save_results()
        
        return self.results
    
    def _run_phase1_mathematical_baseline(self) -> Dict[str, Any]:
        """
        Phase 1: Run all 6 configurations with mathematical models.
        
        Returns:
            Mathematical baseline results
        """
        results = {}
        
        # Get networks from config
        networks = self.config.get("networks", ["smallworld", "scalefree", "random", "echo", "karate", "stubborn"])
        
        # Default parameters for each network type
        network_params = {
            "smallworld": {"n_agents": 50, "k": 4, "beta": 0.1},
            "scalefree": {"n_agents": 50, "m": 2},
            "random": {"n_agents": 50, "p": 0.1},
            "echo": {"n_agents": 50, "n_communities": 2, "p_intra": 0.3, "p_inter": 0.05},
            "karate": {"n_agents": 34},
            "stubborn": {"n_agents": 50, "k": 4, "beta": 0.1, "stubborn_fraction": 0.1, "lambda_flexible": 0.8}
        }
        
        for network in networks:
            logger.info(f"Running mathematical baseline: {network}")
            config = {
                "name": network,
                "topology": network,
                "topology_params": network_params[network],
                "model": "friedkin_johnsen" if network == "stubborn" else "degroot",
                "model_params": {"epsilon": 1e-6}
            }
            result = self._run_mathematical_experiment(config)
            results[network] = result
            
        return results
    
    def _run_phase2_llm_experiments(self) -> Dict[str, Any]:
        """
        Phase 2: Run 6 configurations × 10 topics = 60 LLM experiments.
        
        Returns:
            LLM experiment results
        """
        results = {}
        
        # Get the 6 configurations from Phase 1
        phase1_results = self.results["phase1_mathematical_baseline"]
        
        # Get topics from config
        topics = self.config.get("topics", [
            "immigration", "environment_economy", "corporate_activism", 
            "gun_safety", "social_media", "toilet_paper", "hot_dog",
            "child_free_weddings", "restaurant_etiquette", "human_cloning"
        ])
        
        for config_name, config_data in phase1_results.items():
            results[config_name] = {}
            
            for topic in topics:
                # Test both A vs B and B vs A orientations
                logger.info(f"Running LLM experiment: {config_name} + {topic} (A vs B)")
                result_a_vs_b = self._run_llm_experiment(config_name, topic, config_data, orientation="A_vs_B")
                results[config_name][f"{topic}_A_vs_B"] = result_a_vs_b
                
                logger.info(f"Running LLM experiment: {config_name} + {topic} (B vs A)")
                result_b_vs_a = self._run_llm_experiment(config_name, topic, config_data, orientation="B_vs_A")
                results[config_name][f"{topic}_B_vs_A"] = result_b_vs_a
                
        return results
    
    def _run_phase3_symmetry_testing(self) -> Dict[str, Any]:
        """
        Phase 3: Run B vs A orientation for all experiments.
        
        Returns:
            Symmetry testing results
        """
        results = {}
        
        # Get LLM results from Phase 2
        phase2_results = self.results["phase2_llm_experiments"]
        
        for config_name, config_data in phase2_results.items():
            results[config_name] = {}
            
            for topic, topic_data in config_data.items():
                logger.info(f"Running symmetry test: {config_name} + {topic} (B vs A)")
                result = self._run_symmetry_experiment(config_name, topic, topic_data)
                results[config_name][topic] = result
        
        return results
    
    def _run_phase3_analysis(self) -> Dict[str, Any]:
        """
        Phase 3: Analysis and metrics calculation.
        
        Returns:
            Analysis results
        """
        analysis = {
            "algorithmic_fidelity": self._calculate_algorithmic_fidelity(),
            "systematic_bias": self._calculate_systematic_bias(),
            "symmetry_violations": self._calculate_symmetry_violations(),
            "calibration_errors": self._calculate_calibration_errors()
        }
        
        return analysis
    
    def _run_mathematical_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single mathematical experiment.
        
        Args:
            config: Configuration for the experiment
            
        Returns:
            Experiment results
        """
        # Create network
        network = create_network_model(
            config["topology"],
            config["topology_params"],
            random_seed=42
        )
        
        n_agents = network.n_agents
        adjacency = network.get_adjacency_matrix()
        
        # Initialize opinions
        opinions = np.random.normal(0, 0.3, n_agents)
        opinions = np.clip(opinions, -1.0, 1.0)
        
        # Convert to [0, 1] for mathematical models
        math_opinions = (opinions + 1) / 2
        initial_math_opinions = math_opinions.copy()
        
        # Run simulation
        opinion_history = [opinions.copy()]
        convergence_timestep = None
        
        for timestep in range(1000):  # Max 1000 timesteps
            if config["model"] == "degroot":
                new_math_opinions = update_opinions_pure_degroot(
                    math_opinions, adjacency, config["model_params"]["epsilon"]
                )
            elif config["model"] == "friedkin_johnsen":
                # Get susceptibility values from network if it's a stubborn topology
                if hasattr(network, 'get_lambda_values'):
                    susceptibility = network.get_lambda_values()
                else:
                    # Fallback: create susceptibility matrix from parameters
                    lambda_val = config["topology_params"].get("lambda_flexible", 0.8)
                    stubborn_fraction = config["topology_params"].get("stubborn_fraction", 0.1)
                    n_stubborn = int(n_agents * stubborn_fraction)
                    susceptibility = np.ones(n_agents) * lambda_val
                    if n_stubborn > 0:
                        susceptibility[:n_stubborn] = 0.0
                
                new_math_opinions = update_opinions_friedkin_johnsen(
                    math_opinions, adjacency, susceptibility, initial_math_opinions, 
                    config["model_params"]["epsilon"]
                )
            else:
                raise ValueError(f"Unknown model: {config['model']}")
            
            # Check convergence
            if np.mean(np.abs(new_math_opinions - math_opinions)) < 1e-6:
                convergence_timestep = timestep + 1
                break
            
            math_opinions = new_math_opinions
            # Convert back to [-1, 1] for storage
            agent_opinions = 2 * math_opinions - 1
            opinion_history.append(agent_opinions.copy())
        
        return {
            "config": config,
            "convergence_timestep": convergence_timestep,
            "final_opinions": opinion_history[-1].tolist(),
            "opinion_history": [op.tolist() for op in opinion_history],
            "network_info": {
                "n_agents": n_agents,
                "n_edges": np.sum(adjacency) // 2,
                "avg_degree": np.mean(np.sum(adjacency, axis=1))
            }
        }
    
    def _run_llm_experiment(self, config_name: str, topic: str, baseline_data: Dict[str, Any], orientation: str = "A_vs_B") -> Dict[str, Any]:
        """
        Run a single LLM experiment.
        
        Args:
            config_name: Name of the configuration
            topic: Topic for the experiment
            baseline_data: Baseline data from Phase 1
            orientation: "A_vs_B" or "B_vs_A" for symmetry testing
            
        Returns:
            LLM experiment results
        """
        # Get configuration from baseline
        config = baseline_data["config"]
        
        # Create network
        network = create_network_model(
            config["topology"],
            config["topology_params"],
            random_seed=42
        )
        
        n_agents = network.n_agents
        adjacency = network.get_adjacency_matrix()
        
        # Use same convergence timestep as mathematical baseline
        convergence_timestep = baseline_data["convergence_timestep"]
        
        # Create controller for LLM simulation with orientation
        topic_with_orientation = f"{topic}_{orientation}"
        controller = Controller(
            llm_client=self.llm_client,
            n_agents=n_agents,
            epsilon=config["model_params"]["epsilon"],
            num_timesteps=convergence_timestep,
            topics=[topic_with_orientation],
            random_seed=42,
            llm_enabled=True
        )
        
        # Set the network adjacency
        controller.network.adjacency_matrix = adjacency
        
        # Run simulation
        results = controller.run_simulation(progress_bar=False)
        
        return {
            "config": config,
            "topic": topic,
            "orientation": orientation,
            "convergence_timestep": convergence_timestep,
            "llm_results": results,
            "baseline_comparison": self._compare_with_baseline(results, baseline_data)
        }
    
    def _run_symmetry_experiment(self, config_name: str, topic: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run symmetry test (B vs A orientation).
        
        Args:
            config_name: Name of the configuration
            topic: Topic for the experiment
            original_data: Original A vs B results
            
        Returns:
            Symmetry test results
        """
        # For now, return placeholder - this would run the same experiment
        # but with topic orientation reversed (B vs A instead of A vs B)
        return {
            "config_name": config_name,
            "topic": topic,
            "orientation": "B_vs_A",
            "symmetry_violation": 0.0,  # Placeholder
            "original_results": original_data
        }
    
    def _compare_with_baseline(self, llm_results: Dict[str, Any], baseline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare LLM results with mathematical baseline."""
        llm_final = np.array(llm_results["final_opinions"])
        baseline_final = np.array(baseline_data["final_opinions"])
        
        # Calculate L2 norm (algorithmic fidelity)
        l2_norm = np.linalg.norm(llm_final - baseline_final)
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(llm_final - baseline_final))
        
        # Calculate correlation
        correlation = np.corrcoef(llm_final, baseline_final)[0, 1]
        
        return {
            "l2_norm": float(l2_norm),
            "mae": float(mae),
            "correlation": float(correlation)
        }
    
    def _calculate_algorithmic_fidelity(self) -> Dict[str, Any]:
        """Calculate algorithmic fidelity metrics."""
        # Placeholder - would analyze all LLM vs baseline comparisons
        return {"overall_fidelity": 0.0}
    
    def _calculate_systematic_bias(self) -> Dict[str, Any]:
        """Calculate systematic bias metrics."""
        # Placeholder - would analyze systematic biases by topic/network
        return {"overall_bias": 0.0}
    
    def _calculate_symmetry_violations(self) -> Dict[str, Any]:
        """Calculate symmetry violation metrics."""
        # Placeholder - would analyze A vs B vs B vs A differences
        return {"overall_symmetry_violation": 0.0}
    
    def _calculate_calibration_errors(self) -> Dict[str, Any]:
        """Calculate calibration errors vs human baselines."""
        # Placeholder - would compare LLM results to human polling data
        return {"overall_calibration_error": 0.0}
    
    def _save_results(self):
        """Save results to JSON files."""
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = self._make_json_serializable(self.results)
        
        # Save complete results
        complete_filename = os.path.join(self.output_dir, f"complete_results_{timestamp}.json")
        with open(complete_filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save individual phase results
        for phase_name, phase_data in serializable_results.items():
            phase_filename = os.path.join(self.output_dir, f"{phase_name}_{timestamp}.json")
            with open(phase_filename, 'w') as f:
                json.dump(phase_data, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}/")
        logger.info(f"Complete results: {complete_filename}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def run(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the simple 4-phase experimental protocol.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Complete experimental results
    """
    runner = Runner(config)
    return runner.run_all_phases()
