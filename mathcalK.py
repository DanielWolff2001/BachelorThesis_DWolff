import numpy as np
def specialk(A, C, K):
    """
    Constructs a new matrix K_new with:
    - Diagonal elements from A
    - Lower triangular elements from C (negated)
    - Upper triangular elements from K
    
    Parameters:
    A (numpy.ndarray): Matrix from which to take the diagonal elements.
    C (numpy.ndarray): Matrix from which to take the lower triangular elements.
    K (numpy.ndarray): Matrix from which to take the upper triangular elements.
    
    Returns:
    numpy.ndarray: The constructed matrix.
    """
    # Ensure A, C, and K are numpy arrays
    A = np.array(A)
    C = np.array(C)
    K = np.array(K)
    
    # Check if the matrices have the same shape
    if not (A.shape == C.shape == K.shape):
        raise ValueError("All matrices must have the same shape.")
    
    # Initialize the new matrix with zeros
    K_new = np.zeros_like(K)
    
    # Set the diagonal elements from A
    np.fill_diagonal(K_new, np.diag(A))
    
    # Set the lower triangular elements from C (negated)
    lower_triangular_indices = np.tril_indices_from(K, -1)
    K_new[lower_triangular_indices] = np.where(C[lower_triangular_indices] != 0, -C[lower_triangular_indices], 0)
    
    # Set the upper triangular elements from K
    upper_triangular_indices = np.triu_indices_from(K, 1)
    K_new[upper_triangular_indices] = K[upper_triangular_indices]
    
    return K_new
