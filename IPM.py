import numpy as np
from numpy.core.fromnumeric import trace
import pandas as pd
import matplotlib.pyplot as plt

def hadamard_product(A, B):
    return np.multiply(A, B)

def update_delta_u(T_u, C, A, W, f, zero_edges):
    C_inv = np.linalg.inv(C)
    T_uC_inv = T_u @ C_inv
    Delta_u = np.trace(T_uC_inv) - np.trace(T_u @ A @ W @ A.T) / f
    delta_u = Delta_u / (np.trace(T_uC_inv @ T_uC_inv) + Delta_u**2 / 2)
    # Apply delta_u while respecting zero constraints
    updated_C = C + delta_u * T_u
    for (i, j) in zero_edges:
        updated_C[i, j] = 0
        updated_C[j, i] = 0
    return updated_C

def update_eta_u(Q, a, V_u, f):
    B = np.sum([Q[alpha, beta] * a[beta] for alpha in V_u for beta in range(len(a)) if beta not in V_u])
    D = np.sum([Q[alpha, beta] for alpha in V_u for beta in V_u])
    if D == 0:
        raise ValueError("D must be greater than 0.")
    return (-B + np.sqrt(B**2 + 4 * f * len(V_u) * D)) / (2 * D)

def estimation_algorithm(T, W, A_init, C_init, f, E, V, zero_edges):
    A = A_init
    C = C_init
    log_likelihoods = []
    # Iterative process
    for _ in range(150):  # Number of iterations can be adjusted
        # Maximize over C
        for T_u in E:
            C = update_delta_u(T_u, C, A, W, f, zero_edges)

        # Maximize over A
        Q = hadamard_product(C, W)
        for V_u in V:
            a_u = update_eta_u(Q, np.diag(A), V_u, f)
            for i in V_u:
                A[i, i] = a_u
        
        logL = calculate_log_likelihood(C, A, W, n)
        log_likelihoods.append(logL)
    
    return A, C, log_likelihoods

def create_edge_color_classes(size):
    T = []
    for i in range(size):
        for j in range(i + 1, size):
            T_u = np.zeros((size, size))
            T_u[i, j] = T_u[j, i] = 1
            T.append(T_u)
    return T

def calculate_log_likelihood(C, A, W, n):
    part1 = (n/2) * np.log(np.linalg.det(C))
    part2 = n * np.log(np.linalg.det(A))
    ACAW = np.dot(np.dot(np.dot(A, C), A), W)
    part3 = (1/2) * np.trace(ACAW)
    return part1 + part2 - part3

# Load the dataset
file_path = 'gaussian3d_250samples.csv'
sample = pd.read_csv(file_path).values
size = sample.shape[1]  # Number of variables
n = sample.shape[0]  # Number of observations

# Calculate empirical covariance matrix and Wishart matrix
S = np.cov(sample, rowvar=False)
W = n * S

# Initialize A and C matrices
A_init = np.eye(size)
C_init = np.eye(size)
f = n  # Degrees of freedom

# Define vertex color classes (example: each vertex is its own class)
V = [[i] for i in range(size)]

# Define edge color classes (example: each edge is its own class)
E = create_edge_color_classes(size)

# Specify edges to be set to zero (example)
zero_edges = [[0,1], [1,2]]  # Modify this list as needed


A_final, C_final, log_likelihoods = estimation_algorithm(E, W, A_init, C_init, f, E, V, zero_edges)
print("Final A:\n", A_final)
print("Final C:\n", C_final)
print('logL: ', calculate_log_likelihood(C_final, A_final, W, n))



print(A_final@C_final@A_final)
