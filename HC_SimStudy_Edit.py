import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# Define the directory containing the K matrices
input_dir = '/Users/danielwolff/Library/Mobile Documents/com~apple~CloudDocs/TU/Jaar 3/BEP/Python/K_Matrices'
output_dir = '/Users/danielwolff/Library/Mobile Documents/com~apple~CloudDocs/TU/Jaar 3/BEP/Python/Clustering_Results'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all CSV files in the directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# Function to extract A and C values from a K matrix
def extract_A_C_values(K_matrix):
    A = np.diag(K_matrix)
    C = K_matrix[np.tril_indices_from(K_matrix, k=-1)]
    return A, C

# Function to group elements based on clustering results
def group_elements_with_indices(labels, indices):
    grouped_elements = {}
    for idx, label in enumerate(labels):
        row, col = indices[idx]
        if label not in grouped_elements:
            grouped_elements[label] = []
        grouped_elements[label].append((int(row), int(col)))
    return grouped_elements

# List of distance thresholds
distance_thresholds = [0.05, 0.15, 0.25]

# Process each CSV file
for file in csv_files:
    file_path = os.path.join(input_dir, file)
    print(f"Processing file: {file_path}")  # Debugging statement
    # Read the CSV file and skip the first row and column
    K_matrix = pd.read_csv(file_path, index_col=0).values
    A, C = extract_A_C_values(K_matrix)
    A_values = []
    C_values = []
    A_indices = []
    C_indices = []
    A_values.extend(A)
    C_values.extend(C)
    A_indices.extend([(i, i) for i in range(len(A))])
    C_indices.extend(list(zip(*np.tril_indices_from(K_matrix, k=-1))))

    # Filter out zero values and their indices
    A_values, A_indices = zip(*[(val, idx) for val, idx in zip(A_values, A_indices) if val != 0])
    C_values, C_indices = zip(*[(val, idx) for val, idx in zip(C_values, C_indices) if val != 0])

    # Convert lists to numpy arrays for clustering
    A_values = np.array(A_values).reshape(-1, 1)
    C_values = np.array(C_values).reshape(-1, 1)

    # Ensure there are at least 2 samples for clustering
    if A_values.shape[0] < 2 or C_values.shape[0] < 2:
        print("Insufficient data for clustering. At least 2 samples are required.")
        continue

    for threshold in distance_thresholds:
        # Clustering for matrix C (lower triangular values) with distance threshold
        clustering_C = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None, linkage='ward')
        clustering_C.fit(C_values)

        # Clustering for matrix A (diagonal values) with distance threshold
        clustering_A = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None, linkage='ward')
        clustering_A.fit(A_values)

        # Group elements based on clustering results
        labels_C = clustering_C.labels_
        labels_A = clustering_A.labels_

        # Create indices for the values (1-based index)
        A_indices = np.array(A_indices)
        C_indices = np.array(C_indices)

        grouped_elements_C = group_elements_with_indices(labels_C, C_indices)
        grouped_elements_A = group_elements_with_indices(labels_A, A_indices)

        # Create lists of clustered elements
        clustered_edges = []
        clustered_vertices = []

        for cluster_id, elements in grouped_elements_C.items():
            cluster = []
            for row, col in elements:
                edge = f"~X{row + 1}:X{col + 1}"
                cluster.append(edge)
                # Add symmetric edge
                symmetric_edge = f"~X{col + 1}:X{row + 1}"
                if symmetric_edge not in cluster:
                    cluster.append(symmetric_edge)
            clustered_edges.append(cluster)

        for cluster_id, elements in grouped_elements_A.items():
            cluster = []
            for row, col in elements:
                vertex = f"~X{row + 1}"
                if vertex not in cluster:
                    cluster.append(vertex)
            clustered_vertices.append(cluster)

        # Save the results to CSV files
        threshold_str = str(threshold).replace('.', '_')
        edges_df = pd.DataFrame({'Clustered Edges': [', '.join(cluster) for cluster in clustered_edges]})
        vertices_df = pd.DataFrame({'Clustered Vertices': [', '.join(cluster) for cluster in clustered_vertices]})

        base_filename = os.path.splitext(file)[0]
        edges_df.to_csv(os.path.join(output_dir, f'{base_filename}_threshold_{threshold_str}_clustered_edges.csv'), index=False)
        vertices_df.to_csv(os.path.join(output_dir, f'{base_filename}_threshold_{threshold_str}_clustered_vertices.csv'), index=False)

print("Clustering completed and results saved successfully.")
