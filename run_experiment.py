#!/usr/bin/env python3
"""
Simple script to run opinion dynamics experiments.
"""

import json
import argparse
import logging
from network_of_agents.runner import Runner

def main():
    parser = argparse.ArgumentParser(description="Run opinion dynamics experiments")
    parser.add_argument("config", help="Path to configuration file (.json)")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Add output directory to config
    config["output_dir"] = args.output_dir
    
    # Run experiments
    try:
        runner = Runner(config)
        results = runner.run_all_phases()
        logger.info("Experiments completed successfully!")
        logger.info(f"Raw results saved to {args.output_dir}")
        logger.info("Run 'python analyze_results.py <results_dir>' to create visualizations and analysis")
    except Exception as e:
        logger.error(f"Failed to run experiments: {e}")
        raise

if __name__ == "__main__":
    main()
