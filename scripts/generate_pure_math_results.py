#!/usr/bin/env python3
"""
Generate pure mathematical results and visualizations.

This script runs the pure mathematical models and generates all plots
for the canonical experimental configurations.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def main():
    """Main function to generate all pure math results."""
    print("🚀 Generating Pure Mathematical Results and Visualizations")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("network_of_agents"):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Generate network visualizations
    print("\n📊 Step 1: Generating network topology visualizations...")
    success = run_command(
        "python scripts/visualize_networks.py --comparison",
        "Network topology visualization"
    )
    
    if not success:
        print("❌ Failed to generate network visualizations")
        sys.exit(1)
    
    # Step 2: Run pure mathematical analysis
    print("\n🧮 Step 2: Running pure mathematical simulations...")
    success = run_command(
        "python scripts/run_pure_math_analysis.py",
        "Pure mathematical analysis"
    )
    
    if not success:
        print("❌ Failed to run pure mathematical analysis")
        sys.exit(1)
    
    # Step 3: Check results
    print("\n📁 Step 3: Checking generated files...")
    
    # Check for network plots
    if os.path.exists("results/visualizations"):
        plot_files = [f for f in os.listdir("results/visualizations") if f.endswith('.png')]
        print(f"✅ Generated {len(plot_files)} network plots in results/visualizations/")
    else:
        print("❌ Network plots directory not found")
    
    # Check for pure math results
    if os.path.exists("results/pure_math"):
        result_files = [f for f in os.listdir("results/pure_math") if f.endswith('.png')]
        print(f"✅ Generated {len(result_files)} result plots in results/pure_math/")
        
        if os.path.exists("results/pure_math/pure_math_results.json"):
            print("✅ Generated pure_math_results.json with simulation data")
        else:
            print("❌ pure_math_results.json not found")
    else:
        print("❌ Pure math results directory not found")
    
    print("\n🎉 Pure mathematical analysis complete!")
    print("\nGenerated files:")
    print("📁 results/visualizations/ - Network topology visualizations")
    print("📁 results/pure_math/ - Simulation results and convergence plots")
    print("📄 results/pure_math/pure_math_results.json - Complete simulation data")
    print("📄 results/pure_math/summary.json - Summary statistics")

if __name__ == "__main__":
    main()
