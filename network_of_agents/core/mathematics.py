"""
Core mathematical functions for the network of agents simulation.

This module implements the mathematical framework described in the research paper,
including opinion dynamics, network evolution, and similarity calculations.
"""

import numpy as np
from typing import Tuple


def row_normalize(M: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Row-normalize a matrix: R(M) = (1 / (M1 + ε)) * M
    
    Args:
        M: Input matrix
        epsilon: Small positive parameter to prevent division by zero
        
    Returns:
        Row-normalized matrix
    """
    row_sums = M.sum(axis=1, keepdims=True) + epsilon
    return (1 / row_sums) * M


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
    # Vectorized computation using broadcasting
    # Reshape for broadcasting: (n, 1, d) - (1, n, d) = (n, n, d)
    M_expanded = M[:, np.newaxis, :]  # Shape: (n, 1, d)
    M_broadcast = M[np.newaxis, :, :]  # Shape: (1, n, d)
    
    # Compute L1-norm differences using broadcasting
    D_M = np.linalg.norm(M_expanded - M_broadcast, ord=1, axis=2)
    
    # Set diagonal to 0 (self-differences)
    np.fill_diagonal(D_M, 0)
    
    return row_normalize(D_M, epsilon)


def calculate_SN(M: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calculate row-wise similarity matrix SN(M) := R(1 - (I + DN(M)))
    
    Args:
        M: Input matrix
        epsilon: Small positive parameter for normalization
        
    Returns:
        Row-normalized similarity matrix
    """
    num_rows = M.shape[0]
    DN_M = calculate_DN(M, epsilon)
    
    # 1 - (I + DN_M) means 1 - DN_M with -1 on the diagonal
    # Since DN is hollow, (I+DN) puts 1s on diagonal
    # Then (1 - (I+DN)) makes diagonal 0 and off-diagonal 1-DN
    temp_matrix = np.ones((num_rows, num_rows)) - DN_M
    np.fill_diagonal(temp_matrix, 0)  # Ensure hollow property
    
    return row_normalize(temp_matrix, epsilon)


def calculate_W(X_k: np.ndarray, A_k: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Calculate weighting matrix W(X[k], A[k]) := SN(X[k]) ◦ A[k] + (I - diag([SN(X[k]) ◦ A[k]]1))
    
    Args:
        X_k: Opinion matrix at time k (single topic)
        A_k: Adjacency matrix at time k
        epsilon: Small positive parameter
        
    Returns:
        Weighting matrix W
    """
    # Reshape X_k to 2D matrix for compatibility with existing functions
    X_k_2d = X_k.reshape(-1, 1)
    
    SN_Xk = calculate_SN(X_k_2d, epsilon)
    W_temp = SN_Xk * A_k  # Hadamard product
    
    # Calculate row sums of W_temp
    row_sums_W_temp = W_temp.sum(axis=1)
    
    # Create diagonal matrix for correction
    identity_matrix = np.eye(X_k_2d.shape[0])
    diag_correction = np.diag(row_sums_W_temp)
    
    W_Xk_Ak = W_temp + (identity_matrix - diag_correction)
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
    W_k = calculate_W(X_k, A_k, epsilon)
    X_next = np.dot(W_k, X_k)
    
    # Ensure opinions stay within [-1, 1] bounds
    # X_next = np.clip(X_next, -1.0, 1.0)
    
    return X_next


def calculate_S_hat(X_k: np.ndarray, theta: int, epsilon: float) -> np.ndarray:
    """
    Calculate probabilistic similarity matrix Ŝ[k] := R(SN(X[k]) ◦ θ)
    
    Args:
        X_k: Opinion vector at time k (single topic)
        theta: Positive integer parameter for edge formation
        epsilon: Small positive parameter
        
    Returns:
        Probabilistic similarity matrix Ŝ[k]
    """
    # Reshape X_k to 2D matrix for compatibility
    X_k_2d = X_k.reshape(-1, 1)
    
    SN_Xk = calculate_SN(X_k_2d, epsilon)
    S_hat_k = row_normalize(np.power(SN_Xk, theta), epsilon)
    return S_hat_k


def update_edges(A_k: np.ndarray, X_k: np.ndarray, theta: int, epsilon: float) -> np.ndarray:
    """
    Update edges a_ij[k+1] based on probabilistic similarity
    
    Args:
        A_k: Adjacency matrix at time k
        X_k: Opinion vector at time k (single topic)
        theta: Positive integer parameter for edge formation
        epsilon: Small positive parameter
        
    Returns:
        Updated adjacency matrix A[k+1]
    """
    n = A_k.shape[0]
    S_hat_k = calculate_S_hat(X_k, theta, epsilon)
    
    # Vectorized edge update using broadcasting
    # Generate random values for all pairs at once
    gamma_matrix = np.random.rand(n, n)
    
    # Create mask for upper triangle (i < j)
    upper_triangle_mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    
    # Apply threshold condition vectorized
    edge_mask = (gamma_matrix < np.maximum(S_hat_k, epsilon)) & upper_triangle_mask
    
    # Create symmetric adjacency matrix
    A_next = np.zeros((n, n))
    A_next[edge_mask] = 1
    A_next += A_next.T  # Make symmetric
    
    return A_next


 