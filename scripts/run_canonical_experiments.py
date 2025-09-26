#!/usr/bin/env python3
"""
Canonical experiments runner for LLM algorithmic fidelity study.

This script runs all 6 canonical experimental configurations across
10 topics, testing both A vs B and B vs A orientations for symmetry analysis.
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

from network_of_agents.canonical_configs import (
    CANONICAL_EXPERIMENTS, 
    CANONICAL_TOPICS, 
    create_experiment_config
)
from network_of_agents.simulation.canonical_controller import CanonicalController
from network_of_agents.llm_client import LLMClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_single_experiment(experiment_name: str, topic: Dict[str, Any], 
                         llm_client: LLMClient, results_dir: str) -> Dict[str, Any]:
    """
    Run a single experiment configuration.
    
    Args:
        experiment_name: Name of the canonical experiment
        topic: Topic configuration
        llm_client: LLM client for API calls
        results_dir: Directory to save results
        
    Returns:
        Experiment results
    """
    logger.info(f"Running {experiment_name} with topic: {topic['name']}")
    
    # Create experiment configuration
    config = create_experiment_config(experiment_name, topic)
    
    # Create controller
    controller = CanonicalController(config, llm_client)
    
    # Run simulation
    start_time = time.time()
    results = controller.run_simulation()
    end_time = time.time()
    
    # Add metadata
    results["metadata"] = {
        "experiment_name": experiment_name,
        "topic": topic,
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    filename = f"{experiment_name}_{topic['name'].replace(' ', '_').lower()}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {filepath}")
    return results

def run_symmetry_test(experiment_name: str, topic: Dict[str, Any], 
                     llm_client: LLMClient, results_dir: str) -> Dict[str, Any]:
    """
    Run symmetry test (A vs B and B vs A orientations).
    
    Args:
        experiment_name: Name of the canonical experiment
        topic: Topic configuration
        llm_client: LLM client for API calls
        results_dir: Directory to save results
        
    Returns:
        Symmetry test results
    """
    logger.info(f"Running symmetry test for {experiment_name} with topic: {topic['name']}")
    
    # Test A vs B orientation
    topic_a_vs_b = topic.copy()
    results_a_vs_b = run_single_experiment(experiment_name, topic_a_vs_b, llm_client, results_dir)
    
    # Test B vs A orientation
    topic_b_vs_a = topic.copy()
    topic_b_vs_a["a"], topic_b_vs_a["b"] = topic_b_vs_a["b"], topic_b_vs_a["a"]
    results_b_vs_a = run_single_experiment(experiment_name, topic_b_vs_a, llm_client, results_dir)
    
    # Calculate symmetry violation
    mean_a_vs_b = results_a_vs_b["convergence_info"]["final_mean"]
    mean_b_vs_a = results_b_vs_a["convergence_info"]["final_mean"]
    symmetry_violation = abs(mean_a_vs_b + mean_b_vs_a)  # Should be 0 for perfect symmetry
    
    symmetry_results = {
        "experiment_name": experiment_name,
        "topic": topic["name"],
        "a_vs_b_mean": mean_a_vs_b,
        "b_vs_a_mean": mean_b_vs_a,
        "symmetry_violation": symmetry_violation,
        "results_a_vs_b": results_a_vs_b,
        "results_b_vs_a": results_b_vs_a
    }
    
    # Save symmetry results
    filename = f"{experiment_name}_{topic['name'].replace(' ', '_').lower()}_symmetry.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(symmetry_results, f, indent=2)
    
    logger.info(f"Symmetry violation: {symmetry_violation:.4f}")
    return symmetry_results

def run_all_experiments(llm_model: str = "gpt-4o-mini", 
                       api_key_env: str = "OPENAI_API_KEY",
                       results_dir: str = "results/llm_experiments",
                       run_symmetry_tests: bool = True) -> Dict[str, Any]:
    """
    Run all canonical experiments.
    
    Args:
        llm_model: LLM model to use
        api_key_env: Environment variable name for API key
        results_dir: Directory to save results
        run_symmetry_tests: Whether to run symmetry tests
        
    Returns:
        Summary of all results
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize LLM client
    llm_client = LLMClient(
        model_name=llm_model,
        api_key_env=api_key_env
    )
    
    # Get all experiments and topics
    experiments = list(CANONICAL_EXPERIMENTS.keys())
    topics = CANONICAL_TOPICS
    
    logger.info(f"Running {len(experiments)} experiments across {len(topics)} topics")
    logger.info(f"Total runs: {len(experiments) * len(topics) * (2 if run_symmetry_tests else 1)}")
    
    # Run experiments
    all_results = {}
    symmetry_results = {}
    
    for experiment_name in experiments:
        logger.info(f"Starting experiment: {experiment_name}")
        experiment_results = {}
        
        for topic in topics:
            try:
                if run_symmetry_tests:
                    # Run symmetry test
                    symmetry_result = run_symmetry_test(experiment_name, topic, llm_client, results_dir)
                    symmetry_results[f"{experiment_name}_{topic['name']}"] = symmetry_result
                    experiment_results[topic["name"]] = symmetry_result
                else:
                    # Run single orientation
                    result = run_single_experiment(experiment_name, topic, llm_client, results_dir)
                    experiment_results[topic["name"]] = result
                
            except Exception as e:
                logger.error(f"Failed to run {experiment_name} with {topic['name']}: {e}")
                experiment_results[topic["name"]] = {"error": str(e)}
        
        all_results[experiment_name] = experiment_results
    
    # Calculate summary statistics
    summary = calculate_summary_statistics(all_results, symmetry_results)
    
    # Save summary
    summary_file = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"All experiments completed. Summary saved to {summary_file}")
    return summary

def calculate_summary_statistics(all_results: Dict[str, Any], 
                               symmetry_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics from all results."""
    summary = {
        "total_experiments": len(all_results),
        "total_topics": len(CANONICAL_TOPICS),
        "symmetry_tests_run": len(symmetry_results) > 0,
        "experiment_summaries": {},
        "symmetry_analysis": {},
        "overall_metrics": {}
    }
    
    # Analyze each experiment
    for experiment_name, experiment_results in all_results.items():
        successful_runs = [r for r in experiment_results.values() if "error" not in r]
        failed_runs = [r for r in experiment_results.values() if "error" in r]
        
        if successful_runs:
            final_means = [r.get("convergence_info", {}).get("final_mean", 0) for r in successful_runs]
            api_calls = [r.get("convergence_info", {}).get("api_calls", 0) for r in successful_runs]
            
            summary["experiment_summaries"][experiment_name] = {
                "successful_runs": len(successful_runs),
                "failed_runs": len(failed_runs),
                "mean_final_opinion": np.mean(final_means) if final_means else 0,
                "std_final_opinion": np.std(final_means) if final_means else 0,
                "total_api_calls": sum(api_calls),
                "avg_api_calls_per_run": np.mean(api_calls) if api_calls else 0
            }
    
    # Analyze symmetry violations
    if symmetry_results:
        symmetry_violations = [r["symmetry_violation"] for r in symmetry_results.values()]
        summary["symmetry_analysis"] = {
            "total_symmetry_tests": len(symmetry_violations),
            "mean_symmetry_violation": np.mean(symmetry_violations),
            "std_symmetry_violation": np.std(symmetry_violations),
            "max_symmetry_violation": np.max(symmetry_violations),
            "min_symmetry_violation": np.min(symmetry_violations)
        }
    
    return summary

def main():
    """Main function to run all experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run canonical opinion dynamics experiments")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable for API key")
    parser.add_argument("--results-dir", default="results/llm_experiments", help="Results directory")
    parser.add_argument("--no-symmetry", action="store_true", help="Skip symmetry tests")
    parser.add_argument("--experiment", help="Run specific experiment only")
    parser.add_argument("--topic", help="Run specific topic only")
    
    args = parser.parse_args()
    
    if args.experiment and args.topic:
        # Run single experiment with specific topic
        llm_client = LLMClient(model_name=args.model, api_key_env=args.api_key_env)
        
        # Find the topic
        topic = next((t for t in CANONICAL_TOPICS if t["name"] == args.topic), None)
        if not topic:
            logger.error(f"Topic not found: {args.topic}")
            return
        
        if args.no_symmetry:
            result = run_single_experiment(args.experiment, topic, llm_client, args.results_dir)
        else:
            result = run_symmetry_test(args.experiment, topic, llm_client, args.results_dir)
        
        print(f"Results: {json.dumps(result, indent=2)}")
    else:
        # Run all experiments
        summary = run_all_experiments(
            llm_model=args.model,
            api_key_env=args.api_key_env,
            results_dir=args.results_dir,
            run_symmetry_tests=not args.no_symmetry
        )
        
        print(f"Experiment summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()
