import numpy as np
import pandas as pd
import os

# Define multiple K matrices, choose one

# Simulation study
K_matrix = np.array([
    [14,  3.07,  0, -7.26, -3.89,  0],
    [ 3.07,  7.5,  4.28,  0, -1.42, -1.13],
    [ 0,  4.28,  7.5, -2.85,  0,  0],
    [-7.26,  0, -2.85, 12, -1.8,  0],
    [-3.89, -1.42,  0, -1.8, 12,  5.41],
    [ 0, -1.13,  0,  0,  5.41,  7.5]
])

# Check for symmetry
is_symmetric = np.allclose(K_matrix, K_matrix.T)
if not is_symmetric:
    raise ValueError("The K matrix is not symmetric.")

Covmatrix = np.linalg.inv(K_matrix)

# Check for positive definiteness
eigenvalues = np.linalg.eigvals(Covmatrix)
is_positive_definite = np.all(eigenvalues > 0)
if not is_positive_definite:
    raise ValueError("The covariance matrix is not positive definite.")

# Define mean vector
mean_vector = np.zeros(K_matrix.shape[0])

# Define sample sizes and number of samples
sample_sizes = [50, 500, 10000]
sample_sizes = [1000]
num_samples_per_size = 100

# Create directory to save CSV files if it does not exist
output_dir = '/Users/danielwolff/Library/Mobile Documents/com~apple~CloudDocs/TU/Jaar 3/BEP/Python/SimStudy_Samples'
os.makedirs(output_dir, exist_ok=True)

# Generate and save samples
for size in sample_sizes:
    for i in range(num_samples_per_size):
        # Generate random samples
        random_vectors = np.random.multivariate_normal(mean_vector, Covmatrix, size)
        
        # Create DataFrame
        df = pd.DataFrame(random_vectors)
        
        # Save to CSV
        csv_file_path = os.path.join(output_dir, f'sample_size_{size}_sample_{i+1}.csv')
        df.to_csv(csv_file_path, index=False)

print(f"Generated samples saved to '{output_dir}' directory.")
