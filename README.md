# Network of Agents: Social Network Simulation

A comprehensive framework for studying opinion convergence, biases, and fairness in social networks where users are represented by LLM agents. This project implements a mathematical model of opinion dynamics and network evolution, with full LLM integration for realistic agent behavior and bias detection.

## Features

- **Mathematical Framework**: Implements the French-DeGroot opinion dynamics model with network evolution
- **LLM Integration**: Full integration with GPT-4 via LiteLLM for realistic agent behavior
- **Real-time Visualization**: Interactive web-based dashboard with live network graphs and metrics
- **Bias Testing**: Comprehensive framework for detecting bias patterns in opinion convergence
- **Hot-button Topics**: Pre-configured controversial topics for bias analysis
- **Modular Architecture**: Clean, extensible codebase with separate modules for different components

## Installation

```bash
pip install -r requirements.txt
pip install -e .
# Edit .env with your OpenAI API key
```

## Quick Start

```bash
# Basic simulation
python main.py --mode simulation --agents 50 --timesteps 180

# With GPT-4 integration
python main.py --mode simulation --agents 50 --timesteps 180 --gpt4

# Interactive dashboard
python main.py --mode dashboard --agents 50 --timesteps 180 --gpt4

# Bias testing
python main.py --mode bias-test --agents 50 --timesteps 180 --gpt4
```

## Usage

```bash
# Custom topics
python main.py --mode simulation --topics "gun control" "abortion rights" "climate change" --gpt4

# Different network sizes
python main.py --mode simulation --agents 100 --timesteps 300 --gpt4

# Save results
python main.py --mode simulation --output "my_results" --gpt4
```

## Configuration

Config files: `config/default_config.yaml`, `config/topics_config.yaml`

Key parameters: `n_agents`, `n_topics`, `epsilon`, `theta`, `num_timesteps`, `initial_connection_probability`

## Bias Testing

Tests language bias ("queers" vs "gays"), framing bias ("gun control" vs "gun rights"), and neutral vs controversial topics.

## Architecture

Core: `core/mathematics.py`, `llm/`, `network/graph_model.py`, `simulation/controller.py`, `visualization/dashboard.py`, `bias_testing/`, `data/storage.py`

Key classes: `SimulationController`, `GPT4Agent`, `NetworkModel`, `Dashboard`, `BiasTestingScenarios`

## Mathematical Framework

French-DeGroot model with co-evolving opinions and network topology:

Opinion dynamics: `X[k+1] = W(X[k], A[k])X[k]`
Network evolution: `A[k+1] = f(Ŝ[k], ε, θ)`

## Output

Results: JSON, CSV, interactive plots. Analysis: convergence speed, echo chambers, opinion distribution, bias patterns, network evolution.



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{network_of_agents,
  title={Network of Agents: Social Network Simulation for Opinion Dynamics and Bias Detection},
  author={Your Name},
  year={2024}
}
```



## Acknowledgments

Research areas: Opinion dynamics, GPT-4 integration, network science, fairness in AI. 