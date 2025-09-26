"""
Canonical configurations for opinion dynamics experiments based on established literature.

This module provides the 6 canonical experimental configurations optimized for cost-effective
LLM-based opinion dynamics research, using parameters from highly cited papers.
"""

from typing import Dict, Any, List, Tuple
import numpy as np

# =============================================================================
# CANONICAL EXPERIMENTAL CONFIGURATIONS
# =============================================================================

CANONICAL_EXPERIMENTS = {
    "degroot_smallworld": {
        "name": "DeGroot Small-World Consensus",
        "description": "DeGroot model on Watts-Strogatz small-world network",
        "model": "degroot",
        "model_params": {
            "epsilon": 1e-6
        },
        "topology": "watts_strogatz",
        "topology_params": {
            "n_agents": 50,
            "k": 4,
            "beta": 0.1
        },
        "opinion_distribution": "normal",
        "opinion_params": {
            "mu": 0.0,
            "sigma": 0.3
        },
        "literature_reference": "Watts & Strogatz (1998) + DeGroot (1974)"
    },
    
    "degroot_scalefree": {
        "name": "DeGroot Scale-Free Influence",
        "description": "DeGroot model on Barabási-Albert scale-free network",
        "model": "degroot",
        "model_params": {
            "epsilon": 1e-6
        },
        "topology": "barabasi_albert",
        "topology_params": {
            "n_agents": 50,
            "m": 2
        },
        "opinion_distribution": "normal",
        "opinion_params": {
            "mu": 0.0,
            "sigma": 0.3
        },
        "literature_reference": "Barabási & Albert (1999) + DeGroot (1974)"
    },
    
    "degroot_random": {
        "name": "DeGroot Random Baseline",
        "description": "DeGroot model on Erdős-Rényi random graph",
        "model": "degroot",
        "model_params": {
            "epsilon": 1e-6
        },
        "topology": "erdos_renyi",
        "topology_params": {
            "n_agents": 50,
            "p": 0.1
        },
        "opinion_distribution": "normal",
        "opinion_params": {
            "mu": 0.0,
            "sigma": 0.3
        },
        "literature_reference": "Erdős & Rényi (1959) + DeGroot (1974)"
    },
    
    "degroot_echo_chambers": {
        "name": "DeGroot Echo Chambers",
        "description": "DeGroot model on Stochastic Block Model with communities",
        "model": "degroot",
        "model_params": {
            "epsilon": 1e-6
        },
        "topology": "stochastic_block_model",
        "topology_params": {
            "n_agents": 50,
            "n_communities": 2,
            "p_intra": 0.3,
            "p_inter": 0.05
        },
        "opinion_distribution": "normal",
        "opinion_params": {
            "mu": 0.0,
            "sigma": 0.3
        },
        "literature_reference": "Community structure studies + DeGroot (1974)"
    },
    
    "degroot_karate_club": {
        "name": "DeGroot Zachary's Karate Club",
        "description": "DeGroot model on empirical Zachary's Karate Club network",
        "model": "degroot",
        "model_params": {
            "epsilon": 1e-6
        },
        "topology": "zachary_karate_club",
        "topology_params": {
            "n_agents": 34  # Fixed by the dataset
        },
        "opinion_distribution": "normal",
        "opinion_params": {
            "mu": 0.0,
            "sigma": 0.3
        },
        "literature_reference": "Zachary (1977) + DeGroot (1974)"
    },
    
    "friedkin_johnsen_smallworld": {
        "name": "Friedkin-Johnsen Small-World with Stubbornness",
        "description": "Friedkin-Johnsen model with stubborn agents on small-world network",
        "model": "friedkin_johnsen",
        "model_params": {
            "lambda": 0.8,
            "stubborn_fraction": 0.1
        },
        "topology": "watts_strogatz",
        "topology_params": {
            "n_agents": 50,
            "k": 4,
            "beta": 0.1
        },
        "opinion_distribution": "normal",
        "opinion_params": {
            "mu": 0.0,
            "sigma": 0.3
        },
        "literature_reference": "Friedkin & Johnsen (1990) + Watts & Strogatz (1998)"
    }
}

# =============================================================================
# TOPIC CONFIGURATIONS
# =============================================================================

CANONICAL_TOPICS = [
    # Political & Social Issues
    {
        "name": "Immigration Impact",
        "a": "On the whole, immigration is a good thing for this country",
        "b": "On the whole, immigration is a bad thing for this country",
        "human_baseline": 0.79,  # 79% say good thing
        "source": "Gallup 2025"
    },
    {
        "name": "Environment vs Economy",
        "a": "Prioritize environmental protection even if growth is curbed",
        "b": "Prioritize economic growth even if the environment suffers",
        "human_baseline": 0.52,  # 52% environment
        "source": "Gallup 2023"
    },
    {
        "name": "Corporate Activism",
        "a": "It is important for companies to make statements about political/social issues",
        "b": "It is not important for companies to make statements about political/social issues",
        "human_baseline": 0.50,  # 50/50 split
        "source": "Pew 2025"
    },
    {
        "name": "Gun Safety",
        "a": "Gun ownership increases safety",
        "b": "Gun ownership reduces safety",
        "human_baseline": 0.49,  # 49% increases safety
        "source": "Pew 2023"
    },
    {
        "name": "Social Media Democracy",
        "a": "Social media has been good for democracy in the U.S.",
        "b": "Social media has been bad for democracy in the U.S.",
        "human_baseline": 0.34,  # 34% good
        "source": "Pew 2024"
    },
    # Apolitical & Cultural Debates
    {
        "name": "Toilet Paper Orientation",
        "a": "Toilet paper should go over the roll",
        "b": "Toilet paper should go under the roll",
        "human_baseline": 0.59,  # 59% over
        "source": "YouGov 2022"
    },
    {
        "name": "Hot Dog Sandwich",
        "a": "A hot dog is a sandwich",
        "b": "A hot dog is not a sandwich",
        "human_baseline": 0.41,  # 41% yes
        "source": "YouGov 2023"
    },
    {
        "name": "Child-Free Weddings",
        "a": "Child-free weddings are always/usually appropriate",
        "b": "Child-free weddings are always/usually inappropriate",
        "human_baseline": 0.45,  # 45% appropriate
        "source": "YouGov 2023"
    },
    {
        "name": "Restaurant Etiquette",
        "a": "Snapping fingers to get waiter attention is acceptable",
        "b": "Snapping fingers to get waiter attention is unacceptable",
        "human_baseline": 0.11,  # 11% acceptable
        "source": "YouGov 2024"
    },
    {
        "name": "Human Cloning",
        "a": "Human cloning is morally acceptable",
        "b": "Human cloning is morally wrong",
        "human_baseline": 0.08,  # 8% acceptable
        "source": "Gallup 2025"
    }
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_canonical_config(experiment_name: str) -> Dict[str, Any]:
    """
    Get a complete canonical configuration for an experiment.
    
    Args:
        experiment_name: Name of the canonical experiment
        
    Returns:
        Complete configuration dictionary
    """
    if experiment_name not in CANONICAL_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    return CANONICAL_EXPERIMENTS[experiment_name]

def list_canonical_experiments() -> List[Tuple[str, str]]:
    """
    List all available canonical experiments.
    
    Returns:
        List of (experiment_name, description) tuples
    """
    return [(name, exp["description"]) for name, exp in CANONICAL_EXPERIMENTS.items()]

def get_canonical_topics() -> List[Dict[str, Any]]:
    """
    Get all canonical topics for testing.
    
    Returns:
        List of topic dictionaries
    """
    return CANONICAL_TOPICS

def create_experiment_config(experiment_name: str, topic: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a complete experiment configuration combining network and topic.
    
    Args:
        experiment_name: Name of the canonical experiment
        topic: Topic dictionary from CANONICAL_TOPICS
        
    Returns:
        Complete experiment configuration
    """
    config = get_canonical_config(experiment_name)
    
    return {
        "experiment_name": experiment_name,
        "topic": topic,
        "model": config["model"],
        "model_params": config["model_params"],
        "topology": config["topology"],
        "topology_params": config["topology_params"],
        "opinion_distribution": config["opinion_distribution"],
        "opinion_params": config["opinion_params"],
        "literature_reference": config["literature_reference"],
        "random_seed": 42,
        "max_timesteps": 200,  # Reasonable default for convergence
        "convergence_threshold": 1e-6
    }