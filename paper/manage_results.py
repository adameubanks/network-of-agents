#!/usr/bin/env python3
"""
Simple script to manage simulation results.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List

def build_index() -> List[Dict[str, Any]]:
    """Build runs index from simplified structure."""
    results_dir = Path('results')
    if not results_dir.exists():
        print("No results directory found")
        return []
    
    runs_index = {}
    
    for file_path in results_dir.iterdir():
        if not file_path.is_file() or file_path.name == 'runs_index.json':
            continue
            
        # Extract run info from filename
        run_info = extract_run_info(file_path.name)
        if not run_info:
            continue
        
        run_id = f"{run_info['topic']}_{run_info['n_agents']}_{run_info['timesteps']}_{run_info['model']}_{run_info['timestamp']}"
        
        if run_id not in runs_index:
            runs_index[run_id] = {
                'run_id': run_id,
                'topic': run_info['topic'],
                'n_agents': run_info['n_agents'],
                'timesteps': run_info['timesteps'],
                'model': run_info['model'],
                'timestamp': run_info['timestamp'],
                'date': run_info['timestamp'][:8],
                'files': {}
            }
        
        runs_index[run_id]['files'][run_info['file_type']] = str(file_path)
    
    return list(runs_index.values())

def extract_run_info(filename: str) -> dict:
    """Extract run information from filename."""
    timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
    if not timestamp_match:
        return {}
    
    timestamp = timestamp_match.group(1)
    parts = filename.split(timestamp)
    if len(parts) != 2:
        return {}
    
    before_timestamp = parts[0].rstrip('_')
    before_parts = before_timestamp.split('_')
    
    n, T, model, topic = 0, 0, 'unknown', 'unknown'
    # Format: {topic_safe}_{n_agents}_{num_timesteps}_{model_name}_{timestamp}
    # So we need the last 3 parts before timestamp: model, num_timesteps, n_agents
    # Everything before that is the topic
    if len(before_parts) >= 3:
        try:
            model = before_parts[-1]  # Last part is model
            T = int(before_parts[-2])  # Second to last is num_timesteps
            n = int(before_parts[-3])  # Third to last is n_agents
            topic = '_'.join(before_parts[:-3])  # Everything before is topic
        except ValueError:
            pass
    
    file_type = 'unknown'
    if filename.endswith('.json'):
        file_type = 'data'
    elif filename.endswith('.png'):
        if 'mean_std' in filename:
            file_type = 'mean_std_plot'
        elif 'individuals' in filename:
            file_type = 'individuals_plot'
    elif filename.endswith('.mp4'):
        file_type = 'video'
    
    return {'topic': topic, 'n_agents': n, 'timesteps': T, 'model': model, 'timestamp': timestamp, 'file_type': file_type}

def list_topics(runs: List[Dict[str, Any]]):
    """List all topics."""
    topics = set(r['topic'] for r in runs)
    print("Topics:")
    for topic in sorted(topics):
        count = len([r for r in runs if r['topic'] == topic])
        print(f"  {topic} ({count} runs)")

def list_models(runs: List[Dict[str, Any]]):
    """List all models."""
    models = set(r['model'] for r in runs)
    print("Models:")
    for model in sorted(models):
        count = len([r for r in runs if r['model'] == model])
        print(f"  {model} ({count} runs)")

def search_runs(runs: List[Dict[str, Any]], **filters) -> List[Dict[str, Any]]:
    """Search runs by criteria."""
    filtered = runs
    
    if 'topic' in filters:
        topic_filter = filters['topic'].lower()
        filtered = [r for r in filtered if topic_filter in r['topic'].lower()]
    
    if 'model' in filters:
        model_filter = filters['model'].lower()
        filtered = [r for r in filtered if model_filter in r['model'].lower()]
    
    if 'n_agents' in filters:
        filtered = [r for r in filtered if r['n_agents'] == filters['n_agents']]
    
    return filtered

def print_run(run: Dict[str, Any]):
    """Print run details."""
    print(f"Run: {run['run_id']}")
    print(f"  Topic: {run['topic']}")
    print(f"  Model: {run['model']}")
    print(f"  Agents: {run['n_agents']}, Timesteps: {run['timesteps']}")
    print(f"  Date: {run['date']} {run['timestamp'][9:15]}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Manage simulation results")
    parser.add_argument("--list-topics", action="store_true", help="List all topics")
    parser.add_argument("--list-models", action="store_true", help="List all models")
    parser.add_argument("--topic", help="Filter by topic")
    parser.add_argument("--model", help="Filter by model")
    parser.add_argument("--n-agents", type=int, help="Filter by number of agents")
    parser.add_argument("--build-index", action="store_true", help="Build and save index")
    
    args = parser.parse_args()
    
    runs = build_index()
    if not runs:
        return
    
    if args.build_index:
        with open('results/runs_index.json', 'w') as f:
            json.dump(runs, f, indent=2)
        print(f"Built index for {len(runs)} runs")
        return
    
    if args.list_topics:
        list_topics(runs)
        return
    
    if args.list_models:
        list_models(runs)
        return
    
    # Search and display
    filters = {}
    if args.topic:
        filters['topic'] = args.topic
    if args.model:
        filters['model'] = args.model
    if args.n_agents:
        filters['n_agents'] = args.n_agents
    
    if filters:
        filtered_runs = search_runs(runs, **filters)
        print(f"Found {len(filtered_runs)} runs:")
        for run in filtered_runs:
            print_run(run)
    else:
        # Show recent runs
        recent_runs = sorted(runs, key=lambda x: x['timestamp'], reverse=True)[:5]
        print("Recent 5 runs:")
        for run in recent_runs:
            print_run(run)

if __name__ == "__main__":
    main()
