#!/usr/bin/env python3
"""
Simple opinion dynamics experiment runner.

This script follows the 4-phase experimental protocol:
1. Mathematical baseline (DeGroot/Friedkin-Johnsen)
2. LLM experiments (A vs B and B vs A orientations)
3. Symmetry analysis
4. Results visualization

Usage:
    python run_simple_experiment.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import os

# Import our modules
from network_of_agents.network.graph_generator import create_network_model
from network_of_agents.core.mathematics import update_opinions_pure_degroot, update_opinions_friedkin_johnsen
from network_of_agents.simulation.controller import Controller
from network_of_agents.llm_client import LLMClient

def main():
    """Run the complete experimental protocol."""
    print("🧪 Starting Opinion Dynamics Experiment")
    print("=" * 50)
    
    # Configuration
    config = {
        "networks": ["smallworld", "scalefree", "random", "echo", "karate", "stubborn"],
        "topics": ["immigration", "environment_economy", "corporate_activism", "gun_safety"],
        "n_agents": 50,
        "max_timesteps": 200,
        "epsilon": 1e-6,
        "random_seed": 42
    }
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"experiment_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 1: Mathematical Baseline
    print("\n📊 PHASE 1: Mathematical Baseline")
    print("-" * 30)
    phase1_results = run_mathematical_baseline(config)
    save_results(phase1_results, f"{output_dir}/phase1_mathematical_baseline.json")
    
    # Phase 2: LLM Experiments
    print("\n🤖 PHASE 2: LLM Experiments")
    print("-" * 30)
    phase2_results = run_llm_experiments(config, phase1_results)
    save_results(phase2_results, f"{output_dir}/phase2_llm_experiments.json")
    
    # Phase 3: Symmetry Analysis
    print("\n🔄 PHASE 3: Symmetry Analysis")
    print("-" * 30)
    symmetry_results = analyze_symmetry(phase2_results)
    save_results(symmetry_results, f"{output_dir}/phase3_symmetry_analysis.json")
    
    # Phase 4: Visualization
    print("\n📈 PHASE 4: Visualization")
    print("-" * 30)
    create_all_visualizations(phase1_results, phase2_results, output_dir)
    
    print(f"\n✅ Experiment complete! Results saved to {output_dir}/")
    print(f"📁 Check the PNG files for visualizations")

def run_mathematical_baseline(config):
    """Phase 1: Run mathematical models on all networks."""
    results = {}
    
    for network_name in config["networks"]:
        print(f"  Running {network_name}...")
        
        # Create network
        network = create_network_model(network_name, get_network_params(network_name, config))
        adjacency = network.get_adjacency_matrix()
        n_agents = network.n_agents
        
        # Initialize random opinions [-1, 1]
        opinions = np.random.uniform(-1, 1, n_agents)
        opinion_history = [opinions.copy()]
        
        # Run simulation
        for t in range(config["max_timesteps"]):
            # Convert to [0, 1] for mathematical models
            math_opinions = (opinions + 1) / 2
            
            # Update opinions
            if network_name == "stubborn":
                # For stubborn agents, use simple DeGroot with some agents more resistant
                lambda_values = np.random.uniform(0.1, 0.9, n_agents)  # Random stubbornness
                X_0 = math_opinions.copy()  # Initial opinions
                new_math_opinions = update_opinions_friedkin_johnsen(
                    math_opinions, adjacency, lambda_values, X_0, epsilon=config["epsilon"]
                )
            else:
                new_math_opinions = update_opinions_pure_degroot(
                    math_opinions, adjacency, epsilon=config["epsilon"]
                )
            
            # Convert back to [-1, 1]
            new_opinions = 2 * new_math_opinions - 1
            
            # Check convergence (compare with previous step)
            if t > 0:
                max_change = np.max(np.abs(new_opinions - opinions))
                if max_change < config["epsilon"]:
                    print(f"    Converged at timestep {t}")
                    break
            
            opinions = new_opinions
            opinion_history.append(opinions.copy())
            
            # Progress indicator
            if t % 100 == 0 and t > 0:
                print(f"    Timestep {t}, max change: {max_change:.6f}")
        
        results[network_name] = {
            "network_type": network_name,
            "n_agents": n_agents,
            "convergence_timestep": len(opinion_history) - 1,
            "final_opinions": opinions.tolist(),
            "opinion_history": [op.tolist() for op in opinion_history],
            "adjacency_matrix": adjacency.tolist()
        }
    
    return results

def run_llm_experiments(config, baseline_results):
    """Phase 2: Run LLM experiments with symmetry testing."""
    results = {}
    
    # Try to initialize LLM client
    try:
        llm_client = LLMClient(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        print("  LLM client initialized successfully")
    except Exception as e:
        print(f"  ⚠️  LLM client failed: {e}")
        print("  Skipping LLM experiments (expected if no API key)")
        return {"error": str(e)}
    
    for network_name in config["networks"]:
        results[network_name] = {}
        baseline_data = baseline_results[network_name]
        
        for topic in config["topics"]:
            print(f"  Running {network_name} + {topic}...")
            
            # Test both A vs B and B vs A orientations
            for orientation in ["A_vs_B", "B_vs_A"]:
                try:
                    result = run_single_llm_experiment(
                        network_name, topic, orientation, baseline_data, config, llm_client
                    )
                    results[network_name][f"{topic}_{orientation}"] = result
                except Exception as e:
                    print(f"    ⚠️  {orientation} failed: {e}")
                    results[network_name][f"{topic}_{orientation}"] = {"error": str(e)}
    
    return results

def run_single_llm_experiment(network_name, topic, orientation, baseline_data, config, llm_client):
    """Run a single LLM experiment."""
    # Create network (same as baseline)
    network = create_network_model(network_name, get_network_params(network_name, config))
    adjacency = network.get_adjacency_matrix()
    
    # Create controller
    controller = Controller(
        llm_client=llm_client,
        n_agents=network.n_agents,
        epsilon=config["epsilon"],
        num_timesteps=baseline_data["convergence_timestep"],
        topics=[f"{topic}_{orientation}"],
        random_seed=config["random_seed"],
        llm_enabled=True
    )
    
    # Set network adjacency
    controller.network.adjacency_matrix = adjacency
    
    # Run simulation
    results = controller.run_simulation(progress_bar=False)
    
    return {
        "topic": topic,
        "orientation": orientation,
        "final_opinions": results["final_opinions"],
        "opinion_history": results["opinion_history"],
        "convergence_timestep": baseline_data["convergence_timestep"]
    }

def analyze_symmetry(phase2_results):
    """Phase 3: Analyze symmetry violations."""
    if "error" in phase2_results:
        return {"error": "No LLM results to analyze"}
    
    symmetry_results = {}
    
    for network_name, network_data in phase2_results.items():
        symmetry_results[network_name] = {}
        
        # Group experiments by topic
        topic_groups = {}
        for exp_name, exp_data in network_data.items():
            if "_A_vs_B" in exp_name:
                topic = exp_name.replace("_A_vs_B", "")
                topic_groups[topic] = {"A_vs_B": exp_data}
            elif "_B_vs_A" in exp_name:
                topic = exp_name.replace("_B_vs_A", "")
                if topic not in topic_groups:
                    topic_groups[topic] = {}
                topic_groups[topic]["B_vs_A"] = exp_data
        
        # Calculate symmetry violations
        for topic, orientations in topic_groups.items():
            if "A_vs_B" in orientations and "B_vs_A" in orientations:
                violation = calculate_symmetry_violation(
                    orientations["A_vs_B"], orientations["B_vs_A"]
                )
                symmetry_results[network_name][topic] = violation
    
    return symmetry_results

def calculate_symmetry_violation(a_vs_b_data, b_vs_a_data):
    """Calculate symmetry violation between A vs B and B vs A."""
    if "error" in a_vs_b_data or "error" in b_vs_a_data:
        return None
    
    a_opinions = np.array(a_vs_b_data["final_opinions"])
    b_opinions = np.array(b_vs_a_data["final_opinions"])
    
    # For perfect symmetry, B vs A should be negative of A vs B
    expected_b = -a_opinions
    violation = np.mean(np.abs(b_opinions - expected_b))
    
    return float(violation)

def create_all_visualizations(phase1_results, phase2_results, output_dir):
    """Phase 4: Create all visualizations."""
    print("  Creating network visualizations...")
    
    for network_name, data in phase1_results.items():
        create_network_visualization(network_name, data, output_dir)
    
    if "error" not in phase2_results:
        print("  Creating LLM experiment visualizations...")
        for network_name, network_data in phase2_results.items():
            for exp_name, exp_data in network_data.items():
                if "error" not in exp_data:
                    create_llm_visualization(network_name, exp_name, exp_data, output_dir)

def create_network_visualization(network_name, data, output_dir):
    """Create visualization for a mathematical baseline network."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Mathematical Baseline: {network_name.title()}", fontsize=16)
    
    # Convert data back to numpy
    opinion_history = [np.array(op) for op in data["opinion_history"]]
    adjacency = np.array(data["adjacency_matrix"])
    
    # 1. Initial network
    G = nx.from_numpy_array(adjacency)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax1, node_color=opinion_history[0], cmap='RdBu_r', 
            vmin=-1, vmax=1, node_size=50)
    ax1.set_title("Initial Network")
    
    # 2. Final network
    nx.draw(G, pos, ax=ax2, node_color=opinion_history[-1], cmap='RdBu_r',
            vmin=-1, vmax=1, node_size=50)
    ax2.set_title("Final Network")
    
    # 3. Opinion trajectories
    for i in range(min(20, len(opinion_history[0]))):  # Show first 20 agents
        ax3.plot([op[i] for op in opinion_history], alpha=0.7)
    ax3.set_title("Opinion Trajectories")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Opinion")
    
    # 4. Mean opinion evolution
    mean_opinions = [np.mean(op) for op in opinion_history]
    ax4.plot(mean_opinions)
    ax4.set_title("Mean Opinion Evolution")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Mean Opinion")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mathematical_{network_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_llm_visualization(network_name, exp_name, data, output_dir):
    """Create visualization for an LLM experiment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"LLM Experiment: {network_name.title()} + {exp_name}", fontsize=14)
    
    # Convert data back to numpy
    opinion_history = [np.array(op) for op in data["opinion_history"]]
    
    # 1. Opinion trajectories
    for i in range(min(20, len(opinion_history[0]))):
        ax1.plot([op[i] for op in opinion_history], alpha=0.7)
    ax1.set_title("Opinion Trajectories")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Opinion")
    
    # 2. Mean opinion evolution
    mean_opinions = [np.mean(op) for op in opinion_history]
    ax2.plot(mean_opinions)
    ax2.set_title("Mean Opinion Evolution")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Mean Opinion")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/llm_{network_name}_{exp_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

def get_network_params(network_name, config):
    """Get parameters for network creation."""
    params = {"n_agents": config["n_agents"]}
    
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
    elif network_name == "stubborn":
        params.update({"k": 4, "beta": 0.1, "stubborn_fraction": 0.1, "lambda_flexible": 0.8})
    
    return params

def save_results(results, filename):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
