#!/usr/bin/env python3
"""
Simple opinion dynamics experiment - easy to understand and debug.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import os

from network_of_agents.network.graph_generator import create_network_model
from network_of_agents.core.mathematics import update_opinions_pure_degroot, update_opinions_friedkin_johnsen

def main():
    print("🧪 Simple Opinion Dynamics Experiment")
    print("=" * 50)
    
    # Simple configuration
    networks = ["smallworld", "scalefree", "random", "echo", "karate", "stubborn"]
    n_agents = 30  # Smaller for faster testing
    max_steps = 50  # Fixed number of steps
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"simple_experiment_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Results will be saved to: {output_dir}/")
    
    # Run experiments
    results = {}
    
    for network_name in networks:
        print(f"\n🔬 Testing {network_name}...")
        
        try:
            result = run_network_experiment(network_name, n_agents, max_steps)
            results[network_name] = result
            print(f"  ✅ {network_name} completed - final mean: {result['final_mean']:.4f}")
        except Exception as e:
            print(f"  ❌ {network_name} failed: {e}")
            results[network_name] = {"error": str(e)}
    
    # Save results
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    create_visualizations(results, output_dir)
    
    print(f"\n✅ Experiment complete!")
    print(f"📁 Check {output_dir}/ for results and visualizations")

def run_network_experiment(network_name, n_agents, max_steps):
    """Run a single network experiment."""
    
    # Create network
    network = create_network_model(network_name, get_network_params(network_name, n_agents))
    adjacency = network.get_adjacency_matrix()
    
    # Initialize random opinions [-1, 1]
    opinions = np.random.uniform(-1, 1, n_agents)
    opinion_history = [opinions.copy()]
    
    # Run simulation for fixed number of steps
    for t in range(max_steps):
        # Convert to [0, 1] for mathematical models
        math_opinions = (opinions + 1) / 2
        
        # Update opinions
        if network_name == "stubborn":
            # Simple stubbornness: some agents resist change
            lambda_values = np.random.uniform(0.1, 0.9, n_agents)
            X_0 = math_opinions.copy()
            new_math_opinions = update_opinions_friedkin_johnsen(
                math_opinions, adjacency, lambda_values, X_0, epsilon=1e-6
            )
        else:
            new_math_opinions = update_opinions_pure_degroot(
                math_opinions, adjacency, epsilon=1e-6
            )
        
        # Convert back to [-1, 1]
        opinions = 2 * new_math_opinions - 1
        opinion_history.append(opinions.copy())
        
        # Show progress every 10 steps
        if t % 10 == 0:
            print(f"    Step {t}: mean = {np.mean(opinions):.4f}")
    
    return {
        "network_type": network_name,
        "n_agents": n_agents,
        "steps": max_steps,
        "final_opinions": opinions.tolist(),
        "final_mean": float(np.mean(opinions)),
        "opinion_history": [op.tolist() for op in opinion_history],
        "adjacency_matrix": adjacency.tolist()
    }

def create_visualizations(results, output_dir):
    """Create simple visualizations."""
    
    for network_name, data in results.items():
        if "error" in data:
            continue
            
        print(f"  Creating plot for {network_name}...")
        
        # Create a simple 2-panel plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Network: {network_name.title()}", fontsize=14)
        
        # Convert data back to numpy
        opinion_history = [np.array(op) for op in data["opinion_history"]]
        adjacency = np.array(data["adjacency_matrix"])
        
        # Plot 1: Opinion trajectories (first 10 agents)
        for i in range(min(10, len(opinion_history[0]))):
            ax1.plot([op[i] for op in opinion_history], alpha=0.7, linewidth=1)
        ax1.set_title("Opinion Trajectories")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Opinion")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean opinion evolution
        mean_opinions = [np.mean(op) for op in opinion_history]
        ax2.plot(mean_opinions, linewidth=2, color='red')
        ax2.set_title("Mean Opinion Evolution")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Mean Opinion")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{network_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

def get_network_params(network_name, n_agents):
    """Get parameters for network creation."""
    params = {"n_agents": n_agents}
    
    if network_name == "smallworld":
        params.update({"k": 4, "beta": 0.1})
    elif network_name == "scalefree":
        params.update({"m": 2})
    elif network_name == "random":
        params.update({"p": 0.1})
    elif network_name == "echo":
        params.update({"n_communities": 2, "p_intra": 0.3, "p_inter": 0.05})
    elif network_name == "karate":
        params = {"n_agents": 34}  # Override for karate club
        return params  # Return early to avoid n_agents conflict
    elif network_name == "stubborn":
        params.update({"k": 4, "beta": 0.1, "stubborn_fraction": 0.1, "lambda_flexible": 0.8})
    
    return params

if __name__ == "__main__":
    main()
