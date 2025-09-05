"""
Social dynamics implementation based on IEEE CCTA 2023 paper.
Implements follow/unfollow mechanism, trust/distrust thresholds, and opinion dynamics.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


def initialize_susceptibility_matrix(n_agents: int, 
                                   base_susceptibility: float = 0.1) -> np.ndarray:
    """
    Initialize susceptibility matrix A with non-homogeneous agent susceptibilities.
    
    Args:
        n_agents: Number of agents in the network
        base_susceptibility: Base susceptibility between agents
        
    Returns:
        Susceptibility matrix A where A[i,j] is agent i's susceptibility to agent j
    """
    A = np.full((n_agents, n_agents), base_susceptibility)
    
    # Agents are not influenced by themselves
    np.fill_diagonal(A, 0.0)
    
    # Row normalize so each agent's total susceptibility sums to 1
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    A = A / row_sums
    
    return A


def apply_social_dynamics(opinions: np.ndarray,
                         posts: np.ndarray,
                         adjacency: np.ndarray,
                         trust_threshold: float,
                         distrust_threshold: float,
                         susceptibility_matrix: np.ndarray,
                         diffusion_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply social dynamics from the IEEE paper.
    
    Implements:
    1. Opinion update: x_k(t+1) = Σ(a_kj * x_j(t)) + (1 - α_k(t)) * x_k(t)
    2. Unfollowing when posts exceed distrust threshold
    
    Args:
        opinions: Current opinion vector (n_agents,)
        posts: Posted content opinion vector (n_agents,)
        adjacency: Current adjacency matrix
        trust_threshold: Trust threshold θr
        distrust_threshold: Distrust threshold θd
        susceptibility_matrix: Susceptibility matrix A
        diffusion_rate: Maximum diffusion rate αk ≤ diffusion_rate
        
    Returns:
        Tuple of (updated_opinions, updated_adjacency)
    """
    n_agents = len(opinions)
    new_opinions = opinions.copy()
    new_adjacency = adjacency.copy()
    
    # Process each agent
    for k in range(n_agents):
        # Find neighbors of agent k
        neighbors = np.where(adjacency[k, :] == 1)[0]
        
        if len(neighbors) == 0:
            continue
        
        # Find influencing neighbors (within trust threshold)
        influencing_neighbors = []
        for j in neighbors:
            if abs(posts[j] - opinions[k]) <= trust_threshold:
                influencing_neighbors.append(j)
        
        # Calculate active susceptibility
        alpha_k = 0.0
        for j in influencing_neighbors:
            alpha_k += susceptibility_matrix[k, j]
        
        # Limit by diffusion rate
        alpha_k = min(alpha_k, diffusion_rate)
        
        # Update opinion using convex combination (Equation 1)
        if influencing_neighbors:
            weighted_sum = 0.0
            for j in influencing_neighbors:
                weighted_sum += susceptibility_matrix[k, j] * opinions[j]
            
            new_opinions[k] = weighted_sum + (1 - alpha_k) * opinions[k]
        
        # Check for unfollowing due to distrust
        for j in neighbors:
            if abs(posts[j] - opinions[k]) >= distrust_threshold:
                new_adjacency[k, j] = 0
                new_adjacency[j, k] = 0  # Undirected graph
    
    return new_opinions, new_adjacency


def random_solicitation(current_adjacency: np.ndarray,
                       solicitation_rate: int,
                       n_agents: int) -> List[Tuple[int, int]]:
    """
    Random solicitation mechanism to balance unfollowing.
    
    Args:
        current_adjacency: Current adjacency matrix
        solicitation_rate: Number of solicitations per timestep
        n_agents: Total number of agents
        
    Returns:
        List of (solicitor, target) pairs for new connections
    """
    solicitations = []
    
    # Randomly select agents to solicit
    solicitors = np.random.choice(n_agents, min(solicitation_rate, n_agents), replace=False)
    
    for solicitor in solicitors:
        # Find potential targets (not already connected)
        potential_targets = []
        for target in range(n_agents):
            if target != solicitor and current_adjacency[solicitor, target] == 0:
                potential_targets.append(target)
        
        if potential_targets:
            target = np.random.choice(potential_targets)
            solicitations.append((solicitor, target))
    
    return solicitations


def update_network_with_solicitations(adjacency: np.ndarray,
                                    solicitations: List[Tuple[int, int]],
                                    acceptance_rate: float = 1.0) -> np.ndarray:
    """
    Update adjacency matrix with new solicitations.
    
    Args:
        adjacency: Current adjacency matrix
        solicitations: List of (solicitor, target) pairs
        acceptance_rate: Probability of accepting solicitation
        
    Returns:
        Updated adjacency matrix
    """
    new_adjacency = adjacency.copy()
    
    for solicitor, target in solicitations:
        if np.random.rand() < acceptance_rate:
            new_adjacency[solicitor, target] = 1
            new_adjacency[target, solicitor] = 1  # Undirected
    
    return new_adjacency


def detect_echo_chambers(adjacency: np.ndarray, 
                        opinions: np.ndarray,
                        min_cluster_size: int = 3,
                        opinion_threshold: float = 0.3) -> List[List[int]]:
    """
    Detect echo chambers in the network based on opinion clustering.
    
    Args:
        adjacency: Current adjacency matrix
        opinions: Opinion vector (n_agents,)
        min_cluster_size: Minimum size for an echo chamber
        opinion_threshold: Maximum opinion difference within a chamber
        
    Returns:
        List of echo chamber clusters
    """
    import networkx as nx
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adjacency)
    
    # Find connected components
    connected_components = list(nx.connected_components(G))
    
    echo_chambers = []
    
    for component in connected_components:
        if len(component) < min_cluster_size:
            continue
        
        component_list = list(component)
        
        # Check if opinions are similar within the component
        component_opinions = opinions[component_list]
        opinion_std = np.std(component_opinions)
        
        # If standard deviation is low, it's an echo chamber
        if opinion_std <= opinion_threshold:
            echo_chambers.append(component_list)
    
    return echo_chambers


def calculate_network_polarization(opinions: np.ndarray) -> float:
    """
    Calculate network polarization as the standard deviation of opinions.
    
    Args:
        opinions: Opinion vector (n_agents,)
        
    Returns:
        Polarization measure (higher = more polarized)
    """
    return np.std(opinions)


def generate_post_content(agent_opinion: float, 
                         noise_level: float = 0.1) -> float:
    """
    Generate post content based on agent opinion with some noise.
    
    Args:
        agent_opinion: Agent's opinion value
        noise_level: Amount of noise to add to post content
        
    Returns:
        Post content opinion value
    """
    # Add noise to opinion for post content
    noise = np.random.normal(0, noise_level)
    post_content = agent_opinion + noise
    
    # Clamp to [-1, 1] range
    post_content = np.clip(post_content, -1.0, 1.0)
    
    return post_content
