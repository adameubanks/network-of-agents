#!/usr/bin/env python3
"""
Main application interface for the Network of Agents simulation.
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from network_of_agents.simulation.controller import SimulationController
from network_of_agents.llm.litellm_client import LiteLLMClient
from network_of_agents.visualization.dashboard import Dashboard
from network_of_agents.config.config_manager import ConfigManager
from network_of_agents.bias_testing.scenarios import BiasTestingScenarios


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Network of Agents Simulation - Study opinion convergence, biases, and fairness"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["simulation", "dashboard", "bias-test", "interactive"],
        default="simulation",
        help="Run mode"
    )
    
    parser.add_argument(
        "--topics", "-t",
        nargs="+",
        help="Topics for the simulation"
    )
    
    parser.add_argument(
        "--agents", "-a",
        type=int,
        default=50,
        help="Number of agents"
    )
    
    parser.add_argument(
        "--timesteps", "-s",
        type=int,
        default=180,
        help="Number of timesteps"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="simulation_results",
        help="Output directory"
    )
    
    parser.add_argument(
        "--llm", "-l",
        action="store_true",
        help="Enable LLM integration"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Dashboard host address"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Dashboard port number"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    
    if args.mode == "simulation":
        run_simulation(args, config_manager)
    elif args.mode == "dashboard":
        run_dashboard(args, config_manager)
    elif args.mode == "bias-test":
        run_bias_testing(args, config_manager)
    elif args.mode == "interactive":
        run_interactive(args, config_manager)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


def run_simulation(args, config_manager):
    """Run a single simulation."""
    print("Starting Network of Agents Simulation...")
    
    # Get configuration
    sim_config = config_manager.get_simulation_config()
    
    # Override with command line arguments
    n_agents = args.agents
    n_topics = len(args.topics) if args.topics else sim_config.get("n_topics", 3)
    num_timesteps = args.timesteps
    
    # Initialize LLM client if requested
    llm_client = None
    if args.llm:
        llm_config = config_manager.get_llm_config()
        try:
            llm_client = LiteLLMClient(
                model_name=llm_config.get("model_name", "gpt-4"),
                api_key=os.getenv(llm_config.get("api_key_env_var", "OPENAI_API_KEY"))
            )
            print("LLM integration enabled")
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            print("Continuing without LLM integration...")
    
    # Create simulation controller
    topics = args.topics or config_manager.get_political_topics()[:n_topics]
    
    controller = SimulationController(
        n_agents=n_agents,
        n_topics=n_topics,
        epsilon=sim_config.get("epsilon", 1e-6),
        theta=sim_config.get("theta", 7),
        num_timesteps=num_timesteps,
        initial_connection_probability=sim_config.get("initial_connection_probability", 0.2),
        llm_client=llm_client,
        topics=topics
    )
    
    print(f"Running simulation with {n_agents} agents, {n_topics} topics, {num_timesteps} timesteps")
    print(f"Topics: {', '.join(topics)}")
    
    # Run simulation
    results = controller.run_simulation(progress_bar=True)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    controller.save_simulation(str(output_dir / "simulation_results.json"))
    controller.data_storage.export_to_csv(str(output_dir), "simulation")
    
    # Print summary
    summary = controller.data_storage.get_summary_statistics()
    print("\nSimulation Summary:")
    print(f"Total timesteps: {summary.get('total_timesteps', 0)}")
    print(f"Average opinion change: {summary.get('average_opinion_change', 0):.4f}")
    print(f"Final network density: {summary.get('final_network_density', 0):.4f}")
    print(f"Final clustering coefficient: {summary.get('final_clustering_coefficient', 0):.4f}")
    print(f"Final echo chambers: {summary.get('final_echo_chambers', 0)}")
    
    print(f"\nResults saved to: {output_dir}")


def run_dashboard(args, config_manager):
    """Run the interactive dashboard."""
    print("Starting Network of Agents Dashboard...")
    
    # Get configuration
    sim_config = config_manager.get_simulation_config()
    viz_config = config_manager.get_visualization_config()
    
    # Initialize LLM client if requested
    llm_client = None
    if args.llm:
        llm_config = config_manager.get_llm_config()
        try:
            llm_client = LiteLLMClient(
                model_name=llm_config.get("model_name", "gpt-4"),
                api_key=os.getenv(llm_config.get("api_key_env_var", "OPENAI_API_KEY"))
            )
            print("LLM integration enabled")
        except Exception as e:
            print(f"Warning: Could not initialize LLM client: {e}")
            print("Continuing without LLM integration...")
    
    # Create simulation controller
    n_agents = args.agents
    n_topics = len(args.topics) if args.topics else sim_config.get("n_topics", 3)
    topics = args.topics or config_manager.get_political_topics()[:n_topics]
    
    controller = SimulationController(
        n_agents=n_agents,
        n_topics=n_topics,
        epsilon=sim_config.get("epsilon", 1e-6),
        theta=sim_config.get("theta", 7),
        num_timesteps=args.timesteps,
        initial_connection_probability=sim_config.get("initial_connection_probability", 0.2),
        llm_client=llm_client,
        topics=topics
    )
    
    # Create and run dashboard
    dashboard = Dashboard(controller, viz_config)
    dashboard.run(host=args.host, port=args.port, debug=args.debug)


def run_bias_testing(args, config_manager):
    """Run bias testing scenarios."""
    print("Starting Bias Testing...")
    
    # Initialize bias testing scenarios
    bias_scenarios = BiasTestingScenarios(config_manager)
    
    # Run bias tests
    results = bias_scenarios.run_all_tests(
        n_agents=args.agents,
        num_timesteps=args.timesteps,
        use_llm=args.llm
    )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    bias_scenarios.save_results(str(output_dir / "bias_test_results.json"), results)
    
    # Print summary
    print("\nBias Testing Summary:")
    for test_name, result in results.items():
        print(f"\n{test_name}:")
        print(f"  Convergence speed difference: {result.get('convergence_speed_diff', 0):.4f}")
        print(f"  Final opinion difference: {result.get('final_opinion_diff', 0):.4f}")
        print(f"  Bias detected: {result.get('bias_detected', False)}")
    
    print(f"\nBias testing results saved to: {output_dir}")


def run_interactive(args, config_manager):
    """Run interactive mode."""
    print("Starting Interactive Mode...")
    print("This mode allows you to interactively configure and run simulations.")
    
    # Interactive configuration
    print("\nConfiguration Options:")
    print("1. Basic simulation")
    print("2. LLM-enabled simulation")
    print("3. Bias testing")
    print("4. Dashboard")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        run_basic_interactive(args, config_manager)
    elif choice == "2":
        run_llm_interactive(args, config_manager)
    elif choice == "3":
        run_bias_interactive(args, config_manager)
    elif choice == "4":
        run_dashboard_interactive(args, config_manager)
    else:
        print("Invalid choice. Exiting.")


def run_basic_interactive(args, config_manager):
    """Run basic interactive simulation."""
    print("\nBasic Simulation Configuration:")
    
    n_agents = int(input("Number of agents (default 50): ") or "50")
    n_topics = int(input("Number of topics (default 3): ") or "3")
    num_timesteps = int(input("Number of timesteps (default 180): ") or "180")
    
    topics = []
    for i in range(n_topics):
        topic = input(f"Topic {i+1}: ")
        topics.append(topic)
    
    # Run simulation
    args.agents = n_agents
    args.topics = topics
    args.timesteps = num_timesteps
    args.llm = False
    
    run_simulation(args, config_manager)


def run_llm_interactive(args, config_manager):
    """Run LLM-enabled interactive simulation."""
    print("\nLLM-Enabled Simulation Configuration:")
    
    n_agents = int(input("Number of agents (default 50): ") or "50")
    n_topics = int(input("Number of topics (default 3): ") or "3")
    num_timesteps = int(input("Number of timesteps (default 180): ") or "180")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        api_key = input("Enter your OpenAI API key: ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    topics = []
    for i in range(n_topics):
        topic = input(f"Topic {i+1}: ")
        topics.append(topic)
    
    # Run simulation
    args.agents = n_agents
    args.topics = topics
    args.timesteps = num_timesteps
    args.llm = True
    
    run_simulation(args, config_manager)


def run_bias_interactive(args, config_manager):
    """Run bias testing interactive mode."""
    print("\nBias Testing Configuration:")
    
    print("Available topic pairs:")
    topic_pairs = config_manager.get_topic_pairs("language_bias")
    for i, pair in enumerate(topic_pairs):
        print(f"{i+1}. {pair[0]} vs {pair[1]}")
    
    choice = input("\nSelect topic pair (1-{}): ".format(len(topic_pairs)))
    try:
        selected_pair = topic_pairs[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice. Using first pair.")
        selected_pair = topic_pairs[0]
    
    n_agents = int(input("Number of agents (default 50): ") or "50")
    num_timesteps = int(input("Number of timesteps (default 180): ") or "180")
    
    # Run bias testing
    args.agents = n_agents
    args.timesteps = num_timesteps
    args.llm = True
    
    run_bias_testing(args, config_manager)


def run_dashboard_interactive(args, config_manager):
    """Run dashboard interactive mode."""
    print("\nDashboard Configuration:")
    
    n_agents = int(input("Number of agents (default 50): ") or "50")
    n_topics = int(input("Number of topics (default 3): ") or "3")
    num_timesteps = int(input("Number of timesteps (default 180): ") or "180")
    use_llm = input("Enable LLM integration? (y/n, default n): ").lower().strip() == 'y'
    
    host = input("Host address (default localhost): ") or "localhost"
    port = int(input("Port number (default 8050): ") or "8050")
    
    # Run dashboard
    args.agents = n_agents
    args.topics = None  # Will use default topics
    args.timesteps = num_timesteps
    args.llm = use_llm
    args.host = host
    args.port = port
    
    run_dashboard(args, config_manager)


if __name__ == "__main__":
    main() 