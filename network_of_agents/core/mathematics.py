"""
Core mathematical functions for the network of agents simulation.

This module implements the mathematical framework described in the research paper,
including opinion dynamics, network evolution, and similarity calculations.
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Optional

def row_normalize(M: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Row-normalize a matrix: R(M) = diag(M1 + ε1))^(-1) M
    
    Args:
        M: Input matrix
        epsilon: Small positive parameter to prevent division by zero
        
    Returns:
        Row-normalized matrix
    """
    return M / (M.sum(axis=1, keepdims=True) + epsilon)

def calculate_DN(M: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calculate row-wise difference matrix DN(M) := R(D(M))
    
    Where d_ij(M) = ||m_i^T - m_j^T||_1 (L1-norm of difference between row vectors)
    
    Args:
        M: Input matrix
        epsilon: Small positive parameter for normalization
        
    Returns:
        Row-normalized difference matrix
    """
    M_2d = M.reshape(-1, 1) if M.ndim == 1 else M
    D_M = np.linalg.norm(M_2d[:, np.newaxis, :] - M_2d[np.newaxis, :, :], ord=1, axis=2)
    np.fill_diagonal(D_M, 0)
    return row_normalize(D_M, epsilon)

def calculate_SN(M: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calculate row-wise similarity matrix SN(M) := R(1 - (I + DN(M)))
    Diagonal entries are zero before normalization.
    
    Args:
        M: Input matrix
        epsilon: Small positive parameter for normalization
        
    Returns:
        Row-normalized similarity matrix
    """
    M_2d = M.reshape(-1, 1) if M.ndim == 1 else M
    DN_M = calculate_DN(M_2d, epsilon)
    n = DN_M.shape[0]
    return row_normalize(np.ones_like(DN_M) - (np.eye(n) + DN_M), epsilon)

def calculate_W(X_k: np.ndarray, A_k: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calculate weighting matrix per Eq. (2):
    W(X[k], A[k]) := SN(X[k]) ◦ A[k] + (I - diag([SN(X[k]) ◦ A[k]]1))
    
    Args:
        X_k: Opinion vector at time k (single topic)
        A_k: Adjacency matrix at time k
        epsilon: Small positive parameter
        
    Returns:
        Weighting matrix W
    """
    X_k_2d = X_k.reshape(-1, 1) if X_k.ndim == 1 else X_k
    SN_Xk = calculate_SN(X_k_2d, epsilon)
    W_temp = SN_Xk * A_k
    W_Xk_Ak = W_temp + (np.eye(X_k_2d.shape[0]) - np.diag(W_temp.sum(axis=1)))
    assert np.allclose(W_Xk_Ak.sum(axis=1), np.ones_like(W_Xk_Ak.sum(axis=1)), atol=1e-8), "W is not row-stochastic within tolerance"
    return W_Xk_Ak

def update_opinions(X_k: np.ndarray, A_k: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Update opinions: X[k+1] = W(X[k], A[k])X[k]
    
    Args:
        X_k: Opinion vector at time k (single topic)
        A_k: Adjacency matrix at time k
        epsilon: Small positive parameter
        
    Returns:
        Updated opinion vector X[k+1]
    """
    return np.dot(calculate_W(X_k, A_k, epsilon), X_k)

def calculate_S_hat(X_k: np.ndarray, theta: int, epsilon: float) -> np.ndarray:
    """
    Calculate probabilistic similarity matrix Ŝ[k] := R(SN^◦θ(X[k]))
    
    Args:
        X_k: Opinion vector at time k (single topic)
        theta: Positive integer parameter for edge formation
        epsilon: Small positive parameter
        
    Returns:
        Probabilistic similarity matrix Ŝ[k]
    """
    X_k_2d = X_k.reshape(-1, 1) if X_k.ndim == 1 else X_k
    return row_normalize(np.power(calculate_SN(X_k_2d, epsilon), theta), epsilon)

def update_opinions_pure_degroot(X_k: np.ndarray, A_k: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Pure DeGroot opinion update: X[k+1] = W X[k]
    
    Implements the classic DeGroot model where:
    - W is a fixed row-stochastic matrix derived from the adjacency matrix A
    - W[i,j] = A[i,j] / (Σ_k A[i,k] + ε) for numerical stability
    - No network evolution occurs (A remains constant)
    
    Mathematical foundation:
    The DeGroot model assumes agents update their opinions by taking a weighted
    average of their neighbors' opinions, where weights are proportional to
    connection strength and normalized to sum to 1 for each agent.
    
    Convergence properties:
    - If the network is strongly connected and aperiodic, opinions converge
    - The consensus value is a weighted average of initial opinions
    - Convergence rate depends on the second largest eigenvalue of W
    
    Args:
        X_k: Opinion vector at time k (single topic)
        A_k: Fixed adjacency matrix (symmetric, hollow)
        epsilon: Small positive parameter for numerical stability
        
    Returns:
        Updated opinion vector X[k+1]
    """
    n = A_k.shape[0]
    
    # Create row-stochastic weight matrix W from adjacency matrix A
    # For isolated nodes (degree 0), keep opinion unchanged (W[i,i] = 1)
    # For connected nodes, W[i,j] = A[i,j] / Σ_k A[i,k]
    row_sums = A_k.sum(axis=1, keepdims=True)
    
    # Initialize W as identity matrix (isolated nodes keep their opinion)
    W = np.eye(n)
    
    # For connected nodes, use adjacency-based weights
    connected_mask = row_sums.flatten() > 0
    if np.any(connected_mask):
        W[connected_mask] = A_k[connected_mask] / row_sums[connected_mask]
    
    # Ensure W is row-stochastic (each row sums to 1)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-10), "W is not row-stochastic"
    
    # DeGroot update: X[k+1] = W X[k]
    return np.dot(W, X_k)

def create_connected_degroot_network(n_agents: int, connectivity: float = 0.3) -> np.ndarray:
    """
    Create a connected network for DeGroot model.
    
    Args:
        n_agents: Number of agents
        connectivity: Target connectivity (0.0 to 1.0)
        
    Returns:
        Symmetric, hollow adjacency matrix
    """
    # Start with empty network
    A = np.zeros((n_agents, n_agents))
    
    # Add edges with probability based on connectivity
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if np.random.rand() < connectivity:
                A[i, j] = 1
                A[j, i] = 1
    
    return A

def update_edges(A_k: np.ndarray, X_k: np.ndarray, theta: int, epsilon: float, update_probability: float = 1.0) -> np.ndarray:
    """
    Lazy edge update per Eq. (4): for each unordered pair (i<j), with
    probability `update_probability` resample the edge using a single γ; otherwise
    keep the previous value. Resampling rule:
    a_ij[k+1] = a_ji[k+1] = 1 if γ < max(ŝ_ij[k], ε) and i ≠ j; else 0.

    Args:
        A_k: Adjacency matrix at time k
        X_k: Opinion vector at time k (single topic)
        theta: Positive integer parameter for edge formation
        epsilon: Small positive parameter

    Returns:
        Updated symmetric, hollow adjacency matrix A[k+1]
    """
    n = A_k.shape[0]
    S_hat_k = calculate_S_hat(X_k, theta, epsilon)

    # Start from current adjacency and selectively resample edges
    A_next = A_k.copy()

    # Iterate over unordered pairs (i < j)
    for i in range(n):
        for j in range(i + 1, n):
            # With probability update_probability, resample this pair per Eq. (4)
            if np.random.rand() < update_probability:
                gamma = np.random.rand()
                # Use symmetric per-pair probability for undirected edge formation
                threshold = max(S_hat_k[i, j], S_hat_k[j, i], epsilon)
                if gamma < threshold:
                    A_next[i, j] = 1
                    A_next[j, i] = 1
                else:
                    A_next[i, j] = 0
                    A_next[j, i] = 0

    # Enforce hollowness explicitly
    np.fill_diagonal(A_next, 0)

    # Lightweight sanity checks
    assert np.allclose(A_next, A_next.T), "A_next is not symmetric"
    assert np.all(np.diag(A_next) == 0), "A_next diagonal is not zero"

    return A_next

def update_opinions_friedkin_johnsen(X_k: np.ndarray, A_k: np.ndarray, lambda_values: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Friedkin-Johnsen opinion update: X[k+1] = ΛX[0] + (I - Λ)WX[k]
    
    The Friedkin-Johnsen model extends DeGroot by allowing agents to have
    different levels of susceptibility to influence (λ). When λ=1, the agent
    behaves like in DeGroot. When λ=0, the agent is completely stubborn and
    never changes from its initial opinion.
    
    Args:
        X_k: Current opinion vector at time k
        A_k: Adjacency matrix at time k
        lambda_values: Susceptibility values for each agent (0 ≤ λ ≤ 1)
        epsilon: Small positive parameter for numerical stability
        
    Returns:
        Updated opinion vector X[k+1]
    """
    n = A_k.shape[0]
    
    # Create row-stochastic weight matrix W from adjacency matrix A
    row_sums = A_k.sum(axis=1, keepdims=True)
    W = np.eye(n)
    
    connected_mask = row_sums.flatten() > 0
    if np.any(connected_mask):
        W[connected_mask] = A_k[connected_mask] / row_sums[connected_mask]
    
    # Create diagonal susceptibility matrix Λ
    Lambda = np.diag(lambda_values)
    
    # Friedkin-Johnsen update: X[k+1] = ΛX[0] + (I - Λ)WX[k]
    # Note: X[0] (initial opinions) should be passed separately in practice
    return np.dot(Lambda, X_k) + np.dot(np.eye(n) - Lambda, np.dot(W, X_k))

def create_ring_lattice(n_agents: int, k: int) -> np.ndarray:
    """
    Create a ring lattice network where each agent connects to k neighbors on each side.
    
    Args:
        n_agents: Number of agents
        k: Number of neighbors on each side
        
    Returns:
        Symmetric, hollow adjacency matrix
    """
    A = np.zeros((n_agents, n_agents))
    
    for i in range(n_agents):
        for j in range(1, k + 1):
            # Connect to k neighbors on each side
            left_neighbor = (i - j) % n_agents
            right_neighbor = (i + j) % n_agents
            
            A[i, left_neighbor] = 1
            A[i, right_neighbor] = 1
    
    return A

def create_watts_strogatz(n_agents: int, k: int, beta: float, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Create a Watts-Strogatz small-world network.
    
    Args:
        n_agents: Number of agents
        k: Number of neighbors on each side in initial ring lattice
        beta: Rewiring probability (0 ≤ beta ≤ 1)
        random_seed: Random seed for reproducible results
        
    Returns:
        Symmetric, hollow adjacency matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Start with ring lattice
    A = create_ring_lattice(n_agents, k)
    
    # Rewire edges with probability beta
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if A[i, j] == 1 and np.random.rand() < beta:
                # Remove existing edge
                A[i, j] = 0
                A[j, i] = 0
                
                # Find a new random connection
                possible_targets = [x for x in range(n_agents) if x != i and A[i, x] == 0]
                if possible_targets:
                    new_target = np.random.choice(possible_targets)
                    A[i, new_target] = 1
                    A[new_target, i] = 1
    
    return A

def create_barabasi_albert(n_agents: int, m: int, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Create a Barabási-Albert scale-free network using preferential attachment.
    
    Args:
        n_agents: Number of agents
        m: Number of edges each new node forms
        random_seed: Random seed for reproducible results
        
    Returns:
        Symmetric, hollow adjacency matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Start with m nodes fully connected
    A = np.zeros((n_agents, n_agents))
    for i in range(min(m, n_agents)):
        for j in range(i + 1, min(m, n_agents)):
            A[i, j] = 1
            A[j, i] = 1
    
    # Add remaining nodes one by one
    for new_node in range(m, n_agents):
        # Calculate degree for preferential attachment
        degrees = np.sum(A, axis=1)
        total_degree = np.sum(degrees)
        
        if total_degree == 0:
            # Fallback: connect to random existing nodes
            targets = np.random.choice(new_node, size=min(m, new_node), replace=False)
        else:
            # Preferential attachment based on degree
            probabilities = degrees[:new_node] / total_degree
            targets = np.random.choice(new_node, size=min(m, new_node), replace=False, p=probabilities)
        
        # Connect new node to selected targets
        for target in targets:
            A[new_node, target] = 1
            A[target, new_node] = 1
    
    return A

def initialize_opinions_uniform(n_agents: int, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Initialize opinions uniformly at random in [-1, 1].
    
    Args:
        n_agents: Number of agents
        random_seed: Random seed for reproducible results
        
    Returns:
        Array of initial opinions
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    return np.random.uniform(-1, 1, n_agents)

def initialize_opinions_normal(n_agents: int, mu: float = 0.0, sigma: float = 0.3, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Initialize opinions using normal distribution, clipped to [-1, 1].
    
    Args:
        n_agents: Number of agents
        mu: Mean of the distribution
        sigma: Standard deviation of the distribution
        random_seed: Random seed for reproducible results
        
    Returns:
        Array of initial opinions
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    opinions = np.random.normal(mu, sigma, n_agents)
    return np.clip(opinions, -1, 1)

def initialize_opinions_polarized(n_agents: int, cluster1_value: float, cluster2_value: float, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Initialize opinions in two distinct clusters.
    
    Args:
        n_agents: Number of agents
        cluster1_value: Opinion value for first cluster
        cluster2_value: Opinion value for second cluster
        random_seed: Random seed for reproducible results
        
    Returns:
        Array of initial opinions
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Randomly assign agents to clusters
    cluster_assignments = np.random.choice([0, 1], size=n_agents)
    opinions = np.where(cluster_assignments == 0, cluster1_value, cluster2_value)
    
    return opinions

def create_complete_graph(n_agents: int) -> np.ndarray:
    """
    Create a complete graph where every agent is connected to every other agent.
    
    Args:
        n_agents: Number of agents
        
    Returns:
        Symmetric, hollow adjacency matrix
    """
    A = np.ones((n_agents, n_agents))
    np.fill_diagonal(A, 0)  # Remove self-loops
    return A

def create_erdos_renyi(n_agents: int, p: float, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Create an Erdős-Rényi random graph.
    
    Args:
        n_agents: Number of agents
        p: Edge probability (0 ≤ p ≤ 1)
        random_seed: Random seed for reproducible results
        
    Returns:
        Symmetric, hollow adjacency matrix
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate random adjacency matrix
    A = np.random.random((n_agents, n_agents))
    A = (A < p).astype(float)
    
    # Make symmetric and remove self-loops
    A = np.triu(A, 1) + np.triu(A, 1).T
    np.fill_diagonal(A, 0)
    
    return A

def initialize_opinions_bimodal(n_agents: int, mu1: float, mu2: float, sigma: float, 
                               weight1: float = 0.5, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Initialize opinions using a bimodal distribution (two normal distributions).
    
    Args:
        n_agents: Number of agents
        mu1: Mean of first distribution
        mu2: Mean of second distribution
        sigma: Standard deviation of both distributions
        weight1: Weight of first distribution (0 ≤ weight1 ≤ 1)
        random_seed: Random seed for reproducible results
        
    Returns:
        Array of initial opinions
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Determine which distribution each agent belongs to
    cluster_assignments = np.random.random(n_agents) < weight1
    
    # Generate opinions from appropriate distribution
    opinions = np.zeros(n_agents)
    opinions[cluster_assignments] = np.random.normal(mu1, sigma, np.sum(cluster_assignments))
    opinions[~cluster_assignments] = np.random.normal(mu2, sigma, np.sum(~cluster_assignments))
    
    # Clip to [-1, 1]
    return np.clip(opinions, -1, 1)
