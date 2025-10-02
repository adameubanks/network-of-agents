"""
Network graph generator for creating various canonical network topologies.

This module provides functions to generate the 6 canonical network topologies
used in the experimental design, based on parameters from highly cited papers.
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, Tuple
from .graph_model import NetworkModel

def generate_watts_strogatz(n_agents: int, k: int, beta: float, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a Watts-Strogatz small-world network.
    
    Args:
        n_agents: Number of agents
        k: Each node connects to k neighbors on each side
        beta: Rewiring probability
        random_seed: Random seed for reproducibility
        
    Returns:
        Adjacency matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate using NetworkX
    G = nx.watts_strogatz_graph(n_agents, k, beta, seed=random_seed)
    return nx.adjacency_matrix(G).toarray().astype(float)

def generate_barabasi_albert(n_agents: int, m: int, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a Barabási-Albert scale-free network.
    
    Args:
        n_agents: Number of agents
        m: Each new node connects to m existing nodes
        random_seed: Random seed for reproducibility
        
    Returns:
        Adjacency matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate using NetworkX
    G = nx.barabasi_albert_graph(n_agents, m, seed=random_seed)
    return nx.adjacency_matrix(G).toarray().astype(float)

def generate_erdos_renyi(n_agents: int, p: float, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate an Erdős-Rényi random graph.
    
    Args:
        n_agents: Number of agents
        p: Edge probability
        random_seed: Random seed for reproducibility
        
    Returns:
        Adjacency matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate using NetworkX
    G = nx.erdos_renyi_graph(n_agents, p, seed=random_seed)
    return nx.adjacency_matrix(G).toarray().astype(float)

def generate_stochastic_block_model(n_agents: int, n_communities: int, p_intra: float, 
                                  p_inter: float, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a Stochastic Block Model network.
    
    Args:
        n_agents: Number of agents
        n_communities: Number of communities
        p_intra: Within-community connection probability
        p_inter: Between-community connection probability
        random_seed: Random seed for reproducibility
        
    Returns:
        Adjacency matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculate community sizes
    community_size = n_agents // n_communities
    community_sizes = [community_size] * n_communities
    # Add remaining agents to first community
    community_sizes[0] += n_agents - sum(community_sizes)
    
    # Create probability matrix
    probs = np.full((n_communities, n_communities), p_inter)
    np.fill_diagonal(probs, p_intra)
    
    # Generate using NetworkX
    G = nx.stochastic_block_model(community_sizes, probs, seed=random_seed)
    return nx.adjacency_matrix(G).toarray().astype(float)

def generate_zachary_karate_club() -> np.ndarray:
    """
    Generate Zachary's Karate Club network (binary version for opinion dynamics).
    
    Returns:
        Binary adjacency matrix (34x34)
    """
    # Load the classic Zachary's Karate Club dataset
    G = nx.karate_club_graph()
    
    # Convert to binary (unweighted) for opinion dynamics
    G_binary = nx.Graph()
    G_binary.add_edges_from(G.edges())
    
    return nx.adjacency_matrix(G_binary).toarray().astype(float)

def generate_network(topology: str, topology_params: Dict[str, Any], 
                    random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a network based on topology type and parameters.
    
    Args:
        topology: Type of network topology
        topology_params: Parameters for the topology
        random_seed: Random seed for reproducibility
        
    Returns:
        Adjacency matrix
    """
    if topology == "watts_strogatz":
        return generate_watts_strogatz(
            n_agents=topology_params["n_agents"],
            k=topology_params["k"],
            beta=topology_params["beta"],
            random_seed=random_seed
        )
    
    elif topology == "barabasi_albert":
        return generate_barabasi_albert(
            n_agents=topology_params["n_agents"],
            m=topology_params["m"],
            random_seed=random_seed
        )
    
    elif topology == "erdos_renyi":
        return generate_erdos_renyi(
            n_agents=topology_params["n_agents"],
            p=topology_params["p"],
            random_seed=random_seed
        )
    
    elif topology == "stochastic_block_model":
        return generate_stochastic_block_model(
            n_agents=topology_params["n_agents"],
            n_communities=topology_params["n_communities"],
            p_intra=topology_params["p_intra"],
            p_inter=topology_params["p_inter"],
            random_seed=random_seed
        )
    
    elif topology == "zachary_karate_club":
        return generate_zachary_karate_club()
    
    else:
        raise ValueError(f"Unknown topology: {topology}")

def create_network_model(topology: str, topology_params: Dict[str, Any], 
                        random_seed: Optional[int] = None) -> NetworkModel:
    """
    Create a NetworkModel with the specified topology.
    
    Args:
        topology: Type of network topology
        topology_params: Parameters for the topology
        random_seed: Random seed for reproducibility
        
    Returns:
        NetworkModel instance with the generated network
    """
    # Generate adjacency matrix
    adjacency_matrix = generate_network(topology, topology_params, random_seed)
    
    # Create network model
    n_agents = adjacency_matrix.shape[0]
    network_model = NetworkModel(n_agents, random_seed)
    network_model.update_adjacency_matrix(adjacency_matrix)
    
    return network_model

def get_network_info(adjacency_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Get information about a network.
    
    Args:
        adjacency_matrix: Network adjacency matrix
        
    Returns:
        Dictionary with network statistics
    """
    n_agents = adjacency_matrix.shape[0]
    total_edges = np.sum(adjacency_matrix) / 2  # Undirected graph
    max_possible_edges = n_agents * (n_agents - 1) / 2
    density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
    average_degree = np.mean(np.sum(adjacency_matrix, axis=1))
    
    return {
        "n_agents": n_agents,
        "total_edges": int(total_edges),
        "density": density,
        "average_degree": average_degree,
        "max_degree": int(np.max(np.sum(adjacency_matrix, axis=1))),
        "min_degree": int(np.min(np.sum(adjacency_matrix, axis=1)))
    }
