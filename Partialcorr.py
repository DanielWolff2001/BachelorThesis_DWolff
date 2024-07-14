# from manipulation import K_est
import numpy as np

def construct_matrices(K):
    """
    Construct matrices A and C from a given precision matrix K.

    Parameters:
    - K (np.array): A symmetric matrix (n x n) representing the precision matrix.

    Returns:
    - A (np.array): A diagonal matrix with square roots of diagonal elements of K.
    - C (np.array): A matrix constructed as per the relationship of partial correlations.
    """
    n = K.shape[0]  # Number of variables/nodes
    A = np.zeros((n, n))  # Initialize A as a zero matrix of size n x n
    C = np.ones((n, n))  # Initialize C with ones on the diagonal

    # Fill diagonal of A with square roots of diagonal entries of K
    for i in range(n):
        A[i, i] = np.sqrt(K[i, i])

    # Compute C using the formula for partial correlations
    for i in range(n):
        for j in range(n):
            if i != j:
                if K[i,j] == 0:
                    C[i,j] = 0
                else:
                    C[i, j] = K[i, j] / (A[i, i] * A[j, j])
    
    return A, C

# print('Matrix K:')
# print(K_est)
# A, C = construct_matrices(K_est)
# print("Matrix A:")
# print(A)
# print("\nMatrix C:")
# print(C)
# # print(np.dot(np.dot(A,C),A))
