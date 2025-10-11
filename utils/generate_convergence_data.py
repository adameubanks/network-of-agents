#!/usr/bin/env python3
"""
Script to generate convergence data for pure math models with configurable parameters.
This script runs experiments to find actual convergence timesteps and generates a JSON file.
"""

import json
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
from datetime import datetime
from network_of_agents.core.mathematics import update_opinions_pure_degroot, update_opinions_friedkin_johnsen
from network_of_agents.network.generator import create_network_model, get_network_params
from network_of_agents.core.mathematics import initialize_opinions_normal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConvergenceExperiment:
    """Class to run convergence experiments with configurable parameters."""
    
    def __init__(self, 
                 epsilon: float = 1e-4,
                 max_steps: int = 1000,
                 friedkin_johnsen_max_steps: int = 50,
                 n_agents: int = 50,
                 random_seed: int = 42,
                 lambda_value: float = 0.1):
        """
        Initialize convergence experiment with configurable parameters.
        
        Args:
            epsilon: Convergence threshold (opinion variance must be < epsilon)
            max_steps: Maximum steps for DeGroot model
            friedkin_johnsen_max_steps: Maximum steps for Friedkin-Johnsen model
            n_agents: Number of agents in the network
            random_seed: Random seed for reproducibility
            lambda_value: Stubbornness parameter for Friedkin-Johnsen model
        """
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.friedkin_johnsen_max_steps = friedkin_johnsen_max_steps
        self.n_agents = n_agents
        self.random_seed = random_seed
        self.lambda_value = lambda_value
        
        # Set random seed
        np.random.seed(random_seed)
    
    def run_single_experiment(self, network_type: str, model: str) -> Dict[str, Any]:
        """
        Run a single convergence experiment.
        
        Returns:
            Dictionary with convergence results
        """
        logger.info(f"Running {model} on {network_type} network with {self.n_agents} agents...")
        
        try:
            # Create network
            network_params = get_network_params(network_type, self.n_agents)
            network = create_network_model(network_type, network_params, random_seed=self.random_seed)
            adjacency = network.get_adjacency_matrix()
            
            # Use actual number of agents from the network (important for karate)
            actual_n_agents = adjacency.shape[0]
            
            # Initialize opinions
            opinions = initialize_opinions_normal(actual_n_agents, random_seed=self.random_seed)
            opinion_history = [opinions.copy()]
            
            # Store initial opinions for Friedkin-Johnsen model
            initial_opinions = opinions.copy()
            
            # Determine max steps based on model
            effective_max_steps = self.friedkin_johnsen_max_steps if model == "friedkin_johnsen" else self.max_steps
            
            # Run simulation until convergence
            converged = False
            final_variance = None
            
            for timestep in range(effective_max_steps):
                # Update opinions based on model
                if model == "degroot":
                    opinions_next = update_opinions_pure_degroot(opinions, adjacency, self.epsilon)
                elif model == "friedkin_johnsen":
                    # For Friedkin-Johnsen, use lambda for all agents
                    lambda_values = np.full(actual_n_agents, self.lambda_value)
                    opinions_next = update_opinions_friedkin_johnsen(opinions, adjacency, lambda_values, initial_opinions, self.epsilon)
                else:
                    raise ValueError(f"Unknown model: {model}")
                
                # Check for convergence (opinion variance < epsilon)
                opinion_variance = np.var(opinions_next)
                final_variance = opinion_variance
                
                if opinion_variance < self.epsilon:
                    logger.info(f"  ✅ Converged at timestep {timestep + 1}: variance = {opinion_variance:.2e}")
                    converged = True
                    break
                    
                opinions = opinions_next
                opinion_history.append(opinions.copy())
            else:
                # If we exit the loop without breaking, we didn't converge
                if model == "friedkin_johnsen":
                    logger.info(f"  ℹ️  Friedkin-Johnsen: Using {effective_max_steps} steps (did not converge, final variance: {final_variance:.2e})")
                else:
                    logger.warning(f"  ⚠️  Did not converge within {effective_max_steps} steps. Final variance: {final_variance:.2e}")
            
            # Calculate final statistics
            final_mean_opinion = np.mean(opinions)
            final_std_opinion = np.std(opinions)
            
            # For Friedkin-Johnsen, always return the max steps if it didn't converge
            if model == "friedkin_johnsen" and not converged:
                convergence_steps = effective_max_steps
            else:
                convergence_steps = timestep + 1
            
            return {
                "convergence_steps": convergence_steps,
                "converged": converged,
                "final_mean_opinion": float(final_mean_opinion),
                "final_std_opinion": float(final_std_opinion),
                "final_variance": float(final_variance) if final_variance is not None else None,
                "n_agents": actual_n_agents,
                "notes": f"{model.title()} model - {'converged' if converged else 'did not converge'} in {convergence_steps} steps"
            }
            
        except Exception as e:
            logger.error(f"  ❌ Experiment failed: {e}")
            return {
                "convergence_steps": effective_max_steps,
                "converged": False,
                "final_mean_opinion": 0.0,
                "final_std_opinion": 1.0,
                "final_variance": None,
                "n_agents": self.n_agents,
                "notes": f"{model.title()} model - failed with error: {str(e)}"
            }
    
    def run_all_experiments(self, topologies: List[str], models: List[str]) -> Dict[str, Any]:
        """Run all convergence experiments and return structured results."""
        
        logger.info("🧮 Starting convergence experiments...")
        logger.info(f"Parameters: epsilon={self.epsilon}, max_steps={self.max_steps}, "
                   f"friedkin_johnsen_max_steps={self.friedkin_johnsen_max_steps}")
        
        results = {
            "convergence_reference": {
                "description": "Convergence steps for opinion dynamics across different network topologies",
                "source": "Pure mathematical experiments (no LLM)",
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "experiment_details": {
                    "n_agents": self.n_agents,
                    "epsilon": self.epsilon,
                    "convergence_threshold": f"Opinion variance < {self.epsilon}",
                    "random_seed": self.random_seed,
                    "max_steps": self.max_steps,
                    "friedkin_johnsen_max_steps": self.friedkin_johnsen_max_steps,
                    "lambda_value": self.lambda_value
                },
                "convergence_threshold": f"Opinion variance < {self.epsilon}"
            },
            "topology_convergence": {},
            "convergence_patterns": {
                "fastest_convergence": [],
                "slowest_convergence": [],
                "convergence_range": {
                    "min_steps": float('inf'),
                    "max_steps": 0,
                    "average_steps": 0.0
                },
                "network_characteristics": {
                    "high_clustering": ["smallworld", "echo"],
                    "hub_structure": ["scalefree"],
                    "uniform_connectivity": ["random"],
                    "real_world": ["karate"],
                    "stubborn_agents": ["stubborn"]
                }
            },
            "experimental_notes": {
                "methodology": "Pure opinion dynamics with connected network initialization",
                "convergence_criteria": f"Opinion variance across all agents < {self.epsilon}",
                "network_generation": f"Each topology generated with consistent parameters for {self.n_agents} agents",
                "special_cases": {
                    "karate": "Uses real network structure from Zachary's karate club dataset",
                    "stubborn": f"Uses Friedkin-Johnsen model with lambda={self.lambda_value} for all agents"
                }
            }
        }
        
        # Run experiments for each topology and model
        all_convergence_steps = []
        
        for topology in topologies:
            logger.info(f"\n📊 Testing topology: {topology}")
            results["topology_convergence"][topology] = {}
            
            for model in models:
                logger.info(f"  Testing model: {model}")
                experiment_result = self.run_single_experiment(topology, model)
                
                # Store results with model-specific keys
                results["topology_convergence"][topology].update({
                    f"{model}_convergence_steps": experiment_result["convergence_steps"],
                    f"{model}_converged": experiment_result["converged"],
                    f"{model}_final_mean_opinion": experiment_result["final_mean_opinion"],
                    f"{model}_final_std_opinion": experiment_result["final_std_opinion"],
                    f"{model}_final_variance": experiment_result["final_variance"],
                    f"{model}_notes": experiment_result["notes"]
                })
                
                # Collect convergence steps for pattern analysis
                if experiment_result["converged"]:
                    all_convergence_steps.append(experiment_result["convergence_steps"])
        
        # Analyze convergence patterns
        if all_convergence_steps:
            results["convergence_patterns"]["convergence_range"] = {
                "min_steps": min(all_convergence_steps),
                "max_steps": max(all_convergence_steps),
                "average_steps": np.mean(all_convergence_steps)
            }
            
            # Sort topologies by convergence speed (only for converged experiments)
            topology_speeds = []
            for topology in topologies:
                if topology in results["topology_convergence"]:
                    # Use degroot convergence steps for ranking
                    degroot_steps = results["topology_convergence"][topology].get("degroot_convergence_steps", float('inf'))
                    if degroot_steps != float('inf'):
                        topology_speeds.append((topology, degroot_steps))
            
            topology_speeds.sort(key=lambda x: x[1])
            results["convergence_patterns"]["fastest_convergence"] = [t[0] for t in topology_speeds[:3]]
            results["convergence_patterns"]["slowest_convergence"] = [t[0] for t in topology_speeds[-3:]]
        
        return results

def main():
    """Main function to run convergence experiments and generate JSON."""
    
    # Load configuration from file
    config_file = "convergence_config.json"
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"📁 Loaded configuration from {config_file}")
    except FileNotFoundError:
        logger.warning(f"Configuration file {config_file} not found, using defaults")
        # Default configuration
        config = {
            "experiment_parameters": {
                "epsilon": 1e-4,
                "max_steps": 1000,
                "friedkin_johnsen_max_steps": 50,
                "n_agents": 50,
                "random_seed": 42,
                "lambda_value": 0.1
            },
            "topologies": ["smallworld", "scalefree", "random", "echo", "karate", "stubborn"],
            "models": ["degroot", "friedkin_johnsen"],
            "output_file": "configs/convergence_data.json"
        }
    
    # Extract parameters
    params = config["experiment_parameters"]
    topologies = config["topologies"]
    models = config["models"]
    output_file = config["output_file"]
    
    # Create experiment instance
    experiment = ConvergenceExperiment(
        epsilon=params["epsilon"],
        max_steps=params["max_steps"],
        friedkin_johnsen_max_steps=params["friedkin_johnsen_max_steps"],
        n_agents=params["n_agents"],
        random_seed=params["random_seed"],
        lambda_value=params["lambda_value"]
    )
    
    # Run all experiments
    results = experiment.run_all_experiments(topologies, models)
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Convergence experiments completed!")
    logger.info(f"📁 Results saved to {output_file}")
    
    # Print summary
    logger.info("\n📊 Convergence Summary:")
    for topology in topologies:
        if topology in results["topology_convergence"]:
            logger.info(f"  {topology}:")
            for model in models:
                steps = results["topology_convergence"][topology].get(f"{model}_convergence_steps", "N/A")
                converged = results["topology_convergence"][topology].get(f"{model}_converged", False)
                status = "✅" if converged else "❌"
                logger.info(f"    {model}: {steps} steps {status}")

if __name__ == "__main__":
    main()
