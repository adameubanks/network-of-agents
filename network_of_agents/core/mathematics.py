"""
Core mathematical functions for opinion dynamics simulation.

This module provides all mathematical operations needed for the research,
including opinion dynamics models, network generation, and analysis functions.
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Optional, Dict, Any

# ============================================================================
# CORE MATHEMATICAL FUNCTIONS
# ============================================================================

def row_normalize(M: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Row-normalize a matrix: R(M) = diag(M1 + ε1))^(-1) M"""
    return M / (M.sum(axis=1, keepdims=True) + epsilon)

def calculate_DN(M: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Calculate row-wise difference matrix DN(M) := R(D(M))"""
    M_2d = M.reshape(-1, 1) if M.ndim == 1 else M
    D_M = np.linalg.norm(M_2d[:, np.newaxis, :] - M_2d[np.newaxis, :, :], ord=1, axis=2)
    np.fill_diagonal(D_M, 0)
    return row_normalize(D_M, epsilon)

def calculate_SN(M: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Calculate row-wise similarity matrix SN(M) := R(1 - (I + DN(M)))"""
    M_2d = M.reshape(-1, 1) if M.ndim == 1 else M
    DN_M = calculate_DN(M_2d, epsilon)
    n = DN_M.shape[0]
    return row_normalize(np.ones_like(DN_M) - (np.eye(n) + DN_M), epsilon)

def calculate_S_hat(X_k: np.ndarray, theta: int, epsilon: float = 1e-6) -> np.ndarray:
    """Calculate probabilistic similarity matrix Ŝ[k] := R(SN^◦θ(X[k]))"""
    X_k_2d = X_k.reshape(-1, 1) if X_k.ndim == 1 else X_k
    return row_normalize(np.power(calculate_SN(X_k_2d, epsilon), theta), epsilon)

# ============================================================================
# OPINION DYNAMICS MODELS
# ============================================================================

def update_opinions_pure_degroot(X_k: np.ndarray, A_k: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Pure DeGroot opinion update: X[k+1] = W X[k]"""
    n = A_k.shape[0]
    row_sums = A_k.sum(axis=1, keepdims=True)
    W = np.eye(n)
    
    connected_mask = row_sums.flatten() > 0
    if np.any(connected_mask):
        W[connected_mask] = A_k[connected_mask] / row_sums[connected_mask]
    
    return np.dot(W, X_k)

def update_opinions_friedkin_johnsen(X_k: np.ndarray, A_k: np.ndarray, lambda_values: np.ndarray, 
                                   X_0: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Friedkin-Johnsen opinion update: X[k+1] = ΛX[0] + (I - Λ)WX[k]"""
    n = A_k.shape[0]
    row_sums = A_k.sum(axis=1, keepdims=True)
    W = np.eye(n)
    
    connected_mask = row_sums.flatten() > 0
    if np.any(connected_mask):
        W[connected_mask] = A_k[connected_mask] / row_sums[connected_mask]
    
    Lambda = np.diag(lambda_values)
    return np.dot(Lambda, X_0) + np.dot(np.eye(n) - Lambda, np.dot(W, X_k))

def update_edges(A_k: np.ndarray, X_k: np.ndarray, theta: int, epsilon: float = 1e-6, 
                update_probability: float = 1.0) -> np.ndarray:
    """Lazy edge update with similarity-based edge formation"""
    n = A_k.shape[0]
    S_hat_k = calculate_S_hat(X_k, theta, epsilon)
    A_next = A_k.copy()
    
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < update_probability:
                gamma = np.random.rand()
                threshold = max(S_hat_k[i, j], S_hat_k[j, i], epsilon)
                if gamma < threshold:
                    A_next[i, j] = A_next[j, i] = 1
                else:
                    A_next[i, j] = A_next[j, i] = 0
    
    np.fill_diagonal(A_next, 0)
    return A_next

# ============================================================================
# NETWORK GENERATION
# ============================================================================

def create_connected_degroot_network(n_agents: int, connectivity: float = 0.3) -> np.ndarray:
    """Create a connected network for DeGroot model"""
    A = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if np.random.rand() < connectivity:
                A[i, j] = A[j, i] = 1
    return A

def create_ring_lattice(n_agents: int, k: int) -> np.ndarray:
    """Create a ring lattice network where each agent connects to k neighbors on each side"""
    A = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(1, k + 1):
            left_neighbor = (i - j) % n_agents
            right_neighbor = (i + j) % n_agents
            A[i, left_neighbor] = A[i, right_neighbor] = 1
    return A

def create_watts_strogatz(n_agents: int, k: int, beta: float, random_seed: Optional[int] = None) -> np.ndarray:
    """Create a Watts-Strogatz small-world network"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    A = create_ring_lattice(n_agents, k)
    
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if A[i, j] == 1 and np.random.rand() < beta:
                A[i, j] = A[j, i] = 0
                possible_targets = [x for x in range(n_agents) if x != i and A[i, x] == 0]
                if possible_targets:
                    new_target = np.random.choice(possible_targets)
                    A[i, new_target] = A[new_target, i] = 1
    
    return A

def create_barabasi_albert(n_agents: int, m: int, random_seed: Optional[int] = None) -> np.ndarray:
    """Create a Barabási-Albert scale-free network using preferential attachment"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    A = np.zeros((n_agents, n_agents))
    for i in range(min(m, n_agents)):
        for j in range(i + 1, min(m, n_agents)):
            A[i, j] = A[j, i] = 1
    
    for new_node in range(m, n_agents):
        degrees = np.sum(A, axis=1)
        total_degree = np.sum(degrees)
        
        if total_degree == 0:
            targets = np.random.choice(new_node, size=min(m, new_node), replace=False)
        else:
            probabilities = degrees[:new_node] / total_degree
            targets = np.random.choice(new_node, size=min(m, new_node), replace=False, p=probabilities)
        
        for target in targets:
            A[new_node, target] = A[target, new_node] = 1
    
    return A

def create_erdos_renyi(n_agents: int, p: float, random_seed: Optional[int] = None) -> np.ndarray:
    """Create an Erdős-Rényi random graph"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    A = np.random.random((n_agents, n_agents))
    A = (A < p).astype(float)
    A = np.triu(A, 1) + np.triu(A, 1).T
    np.fill_diagonal(A, 0)
    return A

def create_stochastic_block_model(n_agents: int, n_communities: int, p_intra: float, 
                                p_inter: float, random_seed: Optional[int] = None) -> np.ndarray:
    """Create a Stochastic Block Model network"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    community_size = n_agents // n_communities
    community_sizes = [community_size] * n_communities
    community_sizes[0] += n_agents - sum(community_sizes)
    
    probs = np.full((n_communities, n_communities), p_inter)
    np.fill_diagonal(probs, p_intra)
    
    G = nx.stochastic_block_model(community_sizes, probs, seed=random_seed)
    return nx.adjacency_matrix(G).toarray().astype(float)

def create_zachary_karate_club() -> np.ndarray:
    """Create Zachary's Karate Club network"""
    G = nx.karate_club_graph()
    G_binary = nx.Graph()
    G_binary.add_edges_from(G.edges())
    return nx.adjacency_matrix(G_binary).toarray().astype(float)

def create_complete_graph(n_agents: int) -> np.ndarray:
    """Create a complete graph where every agent is connected to every other agent"""
    A = np.ones((n_agents, n_agents))
    np.fill_diagonal(A, 0)
    return A

# ============================================================================
# OPINION INITIALIZATION
# ============================================================================

def initialize_opinions_normal(n_agents: int, mu: float = 0.0, sigma: float = 0.3, 
                              random_seed: Optional[int] = None) -> np.ndarray:
    """Initialize opinions using normal distribution, clipped to [-1, 1]"""
    if random_seed is not None:
        np.random.seed(random_seed)
    opinions = np.random.normal(mu, sigma, n_agents)
    return np.clip(opinions, -1, 1)

# ============================================================================
# EVALUATION METRICS
# ============================================================================


def check_convergence(X_current: np.ndarray, X_previous: np.ndarray, 
                     threshold: float = 1e-6) -> bool:
    """Check if opinions have converged based on L2 norm threshold"""
    return np.linalg.norm(X_current - X_previous, ord=2) < threshold

# ============================================================================
# NETWORK ANALYSIS
# ============================================================================

def get_network_info(adjacency_matrix: np.ndarray) -> Dict[str, Any]:
    """Get information about a network"""
    n_agents = adjacency_matrix.shape[0]
    total_edges = np.sum(adjacency_matrix) / 2
    max_possible_edges = n_agents * (n_agents - 1) / 2
    density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
    average_degree = np.mean(np.sum(adjacency_matrix, axis=1))
    
    return {
        "n_agents": n_agents,
        "total_edges": int(total_edges),
        "density": density,
        "average_degree": average_degree,
        "max_degree": int(np.max(np.sum(adjacency_matrix, axis=1))),
        "min_degree": int(np.min(np.sum(adjacency_matrix, axis=1))),
        "adjacency_matrix": adjacency_matrix.tolist()
    }

def create_network(topology: str, topology_params: Dict[str, Any], 
                  random_seed: Optional[int] = None) -> np.ndarray:
    """Create a network based on topology type and parameters"""
    generators = {
        "smallworld": lambda: create_watts_strogatz(
            topology_params["n_agents"], topology_params["k"], 
            topology_params["beta"], random_seed),
        "scalefree": lambda: create_barabasi_albert(
            topology_params["n_agents"], topology_params["m"], random_seed),
        "random": lambda: create_erdos_renyi(
            topology_params["n_agents"], topology_params["p"], random_seed),
        "echo": lambda: create_stochastic_block_model(
            topology_params["n_agents"], topology_params["n_communities"],
            topology_params["p_intra"], topology_params["p_inter"], random_seed),
        "karate": lambda: create_zachary_karate_club(),
        "complete": lambda: create_complete_graph(topology_params["n_agents"]),
        "stubborn": lambda: create_watts_strogatz(
            topology_params["n_agents"], topology_params["k"], 
            topology_params["beta"], random_seed)  # Same as smallworld for network structure
    }
    
    if topology not in generators:
        raise ValueError(f"Unknown topology: {topology}")
    
    return generators[topology]()
