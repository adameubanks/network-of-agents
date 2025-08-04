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
    num_rows = M.shape[0]
    D_M = np.zeros((num_rows, num_rows))
    
    for i in range(num_rows):
        for j in range(num_rows):
            if i != j:
                D_M[i, j] = np.linalg.norm(M[i, :] - M[j, :], ord=1)
    
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
    
    # Ensure opinions stay within [0, 1] bounds
    X_next = np.clip(X_next, 0.0, 1.0)
    
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
    
    A_next = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):  # Iterate only for i < j due to symmetry
            gamma = np.random.rand()
            if gamma < max(S_hat_k[i, j], epsilon):
                A_next[i, j] = 1
                A_next[j, i] = 1  # Ensure symmetry
    
    return A_next


def validate_matrices(X: np.ndarray, A: np.ndarray) -> bool:
    """
    Validate that matrices meet the required properties
    
    Args:
        X: Opinion vector (single topic)
        A: Adjacency matrix
        
    Returns:
        True if matrices are valid, False otherwise
    """
    # Check dimensions
    if X.shape[0] != A.shape[0] or A.shape[0] != A.shape[1]:
        return False
    
    # Check opinion bounds
    if np.any(X < 0) or np.any(X > 1):
        return False
    
    # Check adjacency matrix properties
    if not np.allclose(A, A.T):  # Check symmetry
        return False
    
    if np.any(np.diag(A) != 0):  # Check hollow property
        return False
    
    return True 