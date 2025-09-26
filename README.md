# LLM Algorithmic Fidelity in Opinion Dynamics

This repository implements a comprehensive experimental framework for evaluating the algorithmic fidelity of Large Language Models (LLMs) in multi-agent opinion dynamics simulations.

## Overview

We introduce **algorithmic fidelity** as a framework for evaluating whether LLM-based multi-agent systems faithfully reproduce classical opinion dynamics models like DeGroot and Friedkin-Johnsen. Our experimental design tests 6 canonical configurations across 10 topics to assess systematic biases and symmetry violations.

## Experimental Design

### 6 Canonical Configurations

1. **DeGroot Small-World Consensus** - Watts-Strogatz network (N=50, k=4, β=0.1)
2. **DeGroot Scale-Free Influence** - Barabási-Albert network (N=50, m=2)
3. **DeGroot Random Baseline** - Erdős-Rényi graph (N=50, p=0.1)
4. **DeGroot Echo Chambers** - Stochastic Block Model (N=50, 2 communities)
5. **DeGroot Zachary's Karate Club** - Empirical network (34 nodes)
6. **Friedkin-Johnsen Small-World** - With stubborn agents (λ=0.8, 10% stubborn)

### 10 Topics

Political and apolitical topics with human polling baselines:
- Immigration Impact (79% favorable)
- Environment vs Economy (52% environment)
- Corporate Activism (50/50 split)
- Gun Safety (49% increases safety)
- Social Media Democracy (34% good)
- Toilet Paper Orientation (59% over)
- Hot Dog Sandwich (41% yes)
- Child-Free Weddings (45% appropriate)
- Restaurant Etiquette (11% acceptable)
- Human Cloning (8% acceptable)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Pure Mathematical Results (Recommended First)

```bash
python scripts/generate_pure_math_results.py
```

### Run LLM Experiments

```bash
python scripts/run_canonical_experiments.py
```

### Run Specific Experiment

```bash
python scripts/run_canonical_experiments.py --experiment degroot_smallworld --topic "Immigration Impact"
```

### Visualize Networks Only

```bash
python scripts/visualize_networks.py --comparison
```

## Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Results

Results are organized in the `results/` directory:

- **`results/pure_math/`** - Pure mathematical simulation results and plots
- **`results/llm_experiments/`** - LLM experiment results and symmetry tests  
- **`results/visualizations/`** - Network topology visualizations

## Key Findings

Our experiments reveal:
- **Systematic negativity bias** across all configurations
- **456:1 ratio** of extreme negative to positive ratings
- **Symmetry violations** ranging from 0.019 to 1.814
- **Algorithmic fidelity failures** in both DeGroot and Friedkin-Johnsen models

## File Structure

```
network-of-agents/
├── 📁 network_of_agents/         # Core Python package
│   ├── canonical_configs.py      # 6 canonical experimental configurations
│   ├── simulation/
│   │   └── canonical_controller.py  # Main simulation controller
│   ├── network/
│   │   ├── graph_model.py       # Network model
│   │   └── graph_generator.py   # Network topology generators
│   ├── agent.py                 # Agent implementation
│   ├── llm_client.py           # LLM API client
│   └── core/
│       └── mathematics.py      # Mathematical models (DeGroot, Friedkin-Johnsen)
├── 📁 scripts/                  # Executable scripts
│   ├── run_canonical_experiments.py
│   ├── run_pure_math_analysis.py
│   ├── visualize_networks.py
│   └── generate_pure_math_results.py
├── 📁 results/                  # All experimental results
│   ├── llm_experiments/         # LLM experiment results
│   ├── pure_math/              # Pure mathematical results
│   └── visualizations/         # Network topology plots
└── 📁 paper/                   # Research paper
```

See `PROJECT_STRUCTURE.md` for detailed organization.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{eubanks2024algorithmic,
  title={Algorithmic Fidelity of Large Language Models in Multi-Agent Systems},
  author={Eubanks, Adam and Miller, Caelen and Warnick, Sean},
  journal={AAMAS 2026},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.
