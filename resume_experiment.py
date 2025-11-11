#!/usr/bin/env python3
"""
Resume an incomplete experiment by running only the missing/incomplete topics.
"""

import json
import argparse
import logging
import os
from typing import Dict, Any, List, Set
from network_of_agents.runner import Runner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_completed_topics(experiment_path: str, model: str = "degroot") -> Set[str]:
    """Check which topics are completed in the experiment directory."""
    completed = set()
    model_dir = os.path.join(experiment_path, model)
    
    if not os.path.exists(model_dir):
        return completed
    
    for topic_dir in os.listdir(model_dir):
        topic_path = os.path.join(model_dir, topic_dir)
        if not os.path.isdir(topic_path):
            continue
        
        streaming_file = os.path.join(topic_path, f"{topic_dir}_streaming.json")
        final_file = os.path.join(topic_path, f"{topic_dir}.json")
        
        if os.path.exists(final_file):
            topic = topic_dir
            completed.add(topic)
            logger.info(f"✓ {topic} is completed (final JSON exists)")
        elif os.path.exists(streaming_file):
            try:
                with open(streaming_file, 'r') as f:
                    data = json.load(f)
                    metadata = data.get("experiment_metadata", {})
                    if metadata.get("completed", False):
                        topic = metadata.get("topics", [topic_dir])[0]
                        completed.add(topic)
                        logger.info(f"✓ {topic} is completed")
                    else:
                        topic = metadata.get("topics", [topic_dir])[0]
                        logger.info(f"⚠ {topic} is incomplete (will re-run)")
            except Exception as e:
                logger.warning(f"Could not read {streaming_file}: {e}")
    
    return completed

def load_config_from_experiment(experiment_path: str) -> Dict[str, Any]:
    """Try to load config from experiment metadata, or use default."""
    metadata_file = os.path.join(experiment_path, "metadata", "experiment_metadata.json")
    
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                config = metadata.get("config", {})
                if config:
                    logger.info("Loaded config from experiment metadata")
                    return config
        except Exception as e:
            logger.warning(f"Could not load config from metadata: {e}")
    
    logger.warning("No config found in metadata, you'll need to provide --config")
    return {}

def resume_experiment(experiment_path: str, config: Dict[str, Any]):
    """Resume an experiment by running only missing topics."""
    experiment_path = os.path.abspath(experiment_path)
    
    if not os.path.isdir(experiment_path):
        raise ValueError(f"Experiment directory does not exist: {experiment_path}")
    
    model = config.get("models", ["degroot"])[0]
    all_topics = config.get("topics", [])
    network_types = config.get("topologies", ["smallworld"])
    n_agents = config.get("n_agents", 50)
    reversed_flag = config.get("reversed", False)
    
    completed_topics = check_completed_topics(experiment_path, model)
    missing_topics = [t for t in all_topics if t not in completed_topics]
    
    if not missing_topics:
        logger.info("🎉 All topics are already completed!")
        return
    
    logger.info(f"📋 Found {len(completed_topics)} completed topics")
    logger.info(f"🔄 Need to run {len(missing_topics)} missing topics: {missing_topics}")
    
    output_dir = os.path.dirname(experiment_path)
    exp_name_parts = os.path.basename(experiment_path).split("_")
    if len(exp_name_parts) > 2:
        config["experiment"]["name"] = "_".join(exp_name_parts[:-2])
    else:
        config["experiment"]["name"] = exp_name_parts[0] if exp_name_parts else "experiment"
    
    config["output_dir"] = output_dir
    
    runner = Runner(config)
    runner.experiment_path = experiment_path
    runner.experiment_dir = os.path.basename(experiment_path)
    
    results = []
    for i, topic in enumerate(missing_topics):
        try:
            logger.info(f"🔄 [{i+1}/{len(missing_topics)}] Running {topic}...")
            
            topic_dir = os.path.join(experiment_path, model, topic)
            streaming_file = os.path.join(topic_dir, f"{topic}_streaming.json")
            if os.path.exists(streaming_file):
                logger.info(f"  Removing incomplete streaming file for {topic}")
                os.remove(streaming_file)
            
            result = runner.run_experiment(
                network_types[0],
                topic,
                n_agents,
                model,
                reversed_flag
            )
            results.append(result)
            logger.info(f"✅ [{i+1}/{len(missing_topics)}] {topic} completed")
        except Exception as e:
            logger.error(f"❌ [{i+1}/{len(missing_topics)}] {topic} failed: {e}")
            results.append({"error": str(e), "config": {"topic": topic}})
    
    if results:
        runner.save_results(results)
        logger.info(f"✅ Resume completed! Results saved to: {experiment_path}")

def main():
    parser = argparse.ArgumentParser(description="Resume an incomplete experiment")
    parser.add_argument("experiment_path", help="Path to experiment directory (e.g., results/b_vs_a_mini_degrootsmallworldalltopicsmini_11-04-21-05)")
    parser.add_argument("--config", help="Config file path (optional if metadata exists)")
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = load_config_from_experiment(args.experiment_path)
        if not config:
            logger.error("❌ No config provided and none found in experiment metadata. Use --config")
            return
    
    resume_experiment(args.experiment_path, config)

if __name__ == "__main__":
    main()

