#!/usr/bin/env python3
"""
Ultra-simple test to verify the basic functionality works.
"""

import numpy as np
import matplotlib.pyplot as plt
from network_of_agents.network.graph_generator import create_network_model
from network_of_agents.core.mathematics import update_opinions_pure_degroot

def main():
    print("🧪 Simple Opinion Dynamics Test")
    print("=" * 40)
    
    # Create a simple small-world network
    print("Creating small-world network...")
    network = create_network_model("smallworld", {"n_agents": 20, "k": 4, "beta": 0.1})
    adjacency = network.get_adjacency_matrix()
    n_agents = network.n_agents
    
    print(f"Network has {n_agents} agents")
    
    # Initialize random opinions
    opinions = np.random.uniform(-1, 1, n_agents)
    print(f"Initial opinions: {opinions[:5]}...")
    
    # Run a few iterations
    print("Running simulation...")
    for t in range(10):
        # Convert to [0, 1] for DeGroot
        math_opinions = (opinions + 1) / 2
        
        # Update using DeGroot
        new_math_opinions = update_opinions_pure_degroot(
            math_opinions, adjacency, epsilon=1e-6
        )
        
        # Convert back to [-1, 1]
        opinions = 2 * new_math_opinions - 1
        
        print(f"  Step {t+1}: mean opinion = {np.mean(opinions):.4f}")
    
    print("✅ Test completed successfully!")
    print(f"Final mean opinion: {np.mean(opinions):.4f}")

if __name__ == "__main__":
    main()
