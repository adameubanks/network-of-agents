#!/usr/bin/env python3
"""
Analyze experimental results and create visualizations.

This script takes JSON results from the runner and creates:
- Network visualizations
- Opinion trajectory plots
- Symmetry analysis
- Statistical comparisons
"""

import json
import argparse
import logging
import os
import numpy as np
from typing import Dict, Any, List
from network_of_agents.visualization import NetworkOpinionVisualizer

def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument("results_dir", help="Directory containing JSON result files")
    parser.add_argument("--output-dir", default="analysis", help="Output directory for analysis")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Create analyzer
    analyzer = ResultsAnalyzer(args.results_dir, args.output_dir)
    
    try:
        analyzer.analyze_all()
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to analyze results: {e}")
        raise

class ResultsAnalyzer:
    """Analyze experimental results and create visualizations."""
    
    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.visualizer = NetworkOpinionVisualizer(output_dir)
        
        # Load results
        self.results = self._load_results()
    
    def _load_results(self) -> Dict[str, Any]:
        """Load all JSON result files."""
        results = {}
        
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.results_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results[filename.replace('.json', '')] = data
        
        # Debug: print(f"Loaded files: {list(results.keys())}")
        return results
    
    def analyze_all(self):
        """Run complete analysis."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Find the latest phase files
        phase1_key = None
        phase2_key = None
        
        for key in self.results.keys():
            if key.startswith('phase1_mathematical_baseline'):
                phase1_key = key
            elif key.startswith('phase2_llm_experiments'):
                phase2_key = key
        
        # Analyze mathematical baseline
        if phase1_key:
            self._analyze_mathematical_baseline(phase1_key)
        
        # Analyze LLM experiments
        if phase2_key:
            self._analyze_llm_experiments(phase2_key)
        
        # Analyze symmetry
        self._analyze_symmetry()
        
        # Create summary report
        self._create_summary_report()
    
    def _analyze_mathematical_baseline(self, phase1_key):
        """Analyze mathematical baseline results."""
        logger = logging.getLogger(__name__)
        logger.info("Analyzing mathematical baseline...")
        
        baseline_data = self.results[phase1_key]
        
        for network_name, network_data in baseline_data.items():
            logger.info(f"  Creating visualizations for {network_name}...")
            
            # Convert opinion history back to numpy arrays
            opinion_history = [np.array(op) for op in network_data['opinion_history']]
            
            # Create visualizations
            viz_paths = self.visualizer.create_comprehensive_visualization(
                config_name=f"mathematical_{network_name}",
                config=network_data['config'],
                opinion_history=opinion_history,
                adjacency_matrix=self._reconstruct_adjacency(network_data),
                convergence_timestep=network_data['convergence_timestep'],
                network_info=network_data['network_info']
            )
            
            # Save visualization paths
            network_data['visualization_paths'] = viz_paths
    
    def _analyze_llm_experiments(self, phase2_key):
        """Analyze LLM experiment results."""
        logger = logging.getLogger(__name__)
        logger.info("Analyzing LLM experiments...")
        
        llm_data = self.results[phase2_key]
        
        # Check if LLM experiments failed
        if isinstance(llm_data, dict) and 'error' in llm_data:
            logger.warning(f"LLM experiments failed: {llm_data['error']}")
            return
        
        for network_name, network_experiments in llm_data.items():
            for experiment_name, experiment_data in network_experiments.items():
                logger.info(f"  Creating visualizations for {network_name} + {experiment_name}...")
                
                # Convert opinion history back to numpy arrays
                opinion_history = [np.array(op) for op in experiment_data['llm_results']['opinion_history']]
                
                # Create visualizations
                viz_paths = self.visualizer.create_comprehensive_visualization(
                    config_name=f"llm_{network_name}_{experiment_name}",
                    config=experiment_data['config'],
                    opinion_history=opinion_history,
                    adjacency_matrix=self._reconstruct_adjacency(experiment_data),
                    convergence_timestep=experiment_data['convergence_timestep'],
                    network_info=experiment_data['config'].get('network_info', {})
                )
                
                # Save visualization paths
                experiment_data['visualization_paths'] = viz_paths
    
    def _analyze_symmetry(self):
        """Analyze symmetry violations between A vs B and B vs A orientations."""
        logger = logging.getLogger(__name__)
        logger.info("Analyzing symmetry violations...")
        
        if 'phase2_llm_experiments' not in self.results:
            return
        
        symmetry_results = {}
        llm_data = self.results['phase2_llm_experiments']
        
        for network_name, network_experiments in llm_data.items():
            symmetry_results[network_name] = {}
            
            # Group experiments by topic (ignoring orientation)
            topic_groups = {}
            for exp_name, exp_data in network_experiments.items():
                if '_A_vs_B' in exp_name:
                    topic = exp_name.replace('_A_vs_B', '')
                    topic_groups[topic] = {'A_vs_B': exp_data}
                elif '_B_vs_A' in exp_name:
                    topic = exp_name.replace('_B_vs_A', '')
                    if topic not in topic_groups:
                        topic_groups[topic] = {}
                    topic_groups[topic]['B_vs_A'] = exp_data
            
            # Calculate symmetry violations for each topic
            for topic, orientations in topic_groups.items():
                if 'A_vs_B' in orientations and 'B_vs_A' in orientations:
                    violation = self._calculate_symmetry_violation(
                        orientations['A_vs_B'],
                        orientations['B_vs_A']
                    )
                    symmetry_results[network_name][topic] = violation
        
        # Save symmetry analysis
        symmetry_file = os.path.join(self.output_dir, 'symmetry_analysis.json')
        with open(symmetry_file, 'w') as f:
            json.dump(symmetry_results, f, indent=2)
        
        logger.info(f"Symmetry analysis saved to {symmetry_file}")
    
    def _calculate_symmetry_violation(self, a_vs_b_data: Dict, b_vs_a_data: Dict) -> float:
        """Calculate symmetry violation between A vs B and B vs A results."""
        # Get final opinions
        a_vs_b_opinions = np.array(a_vs_b_data['llm_results']['final_opinions'])
        b_vs_a_opinions = np.array(b_vs_a_data['llm_results']['final_opinions'])
        
        # For perfect symmetry, B vs A should be the negative of A vs B
        expected_b_vs_a = -a_vs_b_opinions
        
        # Calculate mean absolute error
        violation = np.mean(np.abs(b_vs_a_opinions - expected_b_vs_a))
        
        return float(violation)
    
    def _reconstruct_adjacency(self, data: Dict) -> np.ndarray:
        """Reconstruct adjacency matrix from network info."""
        # This is a placeholder - in practice, you'd need to store the adjacency matrix
        # or reconstruct it from the network parameters
        n_agents = data['network_info']['n_agents']
        return np.zeros((n_agents, n_agents))  # Placeholder
    
    def _create_summary_report(self):
        """Create a summary report of all analyses."""
        logger = logging.getLogger(__name__)
        logger.info("Creating summary report...")
        
        summary = {
            "analysis_timestamp": str(np.datetime64('now')),
            "total_networks": len(self.results.get('phase1_mathematical_baseline', {})),
            "total_llm_experiments": sum(
                len(network_experiments) 
                for network_experiments in self.results.get('phase2_llm_experiments', {}).values()
            ),
            "symmetry_analysis_completed": 'phase2_llm_experiments' in self.results
        }
        
        # Save summary
        summary_file = os.path.join(self.output_dir, 'analysis_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_file}")

if __name__ == "__main__":
    main()
