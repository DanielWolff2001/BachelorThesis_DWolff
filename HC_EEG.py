from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from Partialcorr import construct_matrices
from mathcalK import specialk

def plot_dendrogram(model, title, labels, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(20, 10))
    dendro = dendrogram(linkage_matrix, labels=labels, **kwargs)
    plt.xticks(rotation=90, fontsize=8)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Distance")
    plt.show()

def group_elements_with_labels(labels, indices, element_labels):
    grouped_elements = {}
    for idx, label in enumerate(labels):
        row, col = indices[idx]
        if label not in grouped_elements:
            grouped_elements[label] = []
        grouped_elements[label].append((element_labels[row-1], element_labels[col-1]))
    return grouped_elements

element_labels = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8']


P = np.array([
    [ 1.1369937 , -0.46192346,  0.00000000,  0.00000000,  0.0000000 ,  0.00000000,  0.000000000,  0.00000000,  0.166846235,  0.0000000 ,  0.00000000, -0.24516772, -0.20235031,  0.0000000],
    [ 0.3055580 ,  1.32959194, -0.20449466, -0.26163200, -0.3123715 , -0.32987663,  0.005470922,  0.00000000, -0.346863956, -0.1560049 ,  0.19630726,  0.02018285,  0.19472900, -0.1784597],
    [ 0.0000000 ,  0.12720105,  1.20912974, -0.54829392,  0.0000000 ,  0.17829954, -0.277799227,  0.00000000,  0.000000000,  0.0000000 ,  0.00000000, -0.18239847, -0.12910762,  0.0000000],
    [ 0.0000000 ,  0.13864612,  0.31950359,  1.41926919,  0.0000000 , -0.29125369,  0.095311503,  0.00000000,  0.000000000, -0.3026304 , -0.67925571,  0.33326791, -0.03438744, -0.3158365],
    [ 0.0000000 ,  0.20283149,  0.00000000,  0.00000000,  1.1582909 , -0.34303842, -0.153320042,  0.00000000,  0.190377604,  0.0000000 , -0.21063747,  0.00000000,  0.00000000,  0.0000000],
    [ 0.0000000 ,  0.17801320, -0.10580261,  0.14723996,  0.2124928 ,  1.39373743,  0.067902148, -0.32149801, -0.009133629,  0.0000000 , -0.34541296,  0.03070287, -0.44997456, -0.3088033],
    [ 0.0000000 , -0.00351626,  0.19633467, -0.05738778,  0.1131150 , -0.04163335,  1.170202824,  0.00000000, -0.249273827,  0.0000000 , -0.03654442, -0.09262450, -0.21020118, -0.2837883],
    [ 0.0000000 ,  0.00000000,  0.00000000,  0.00000000,  0.0000000 ,  0.19920693,  0.000000000,  1.15795820, -0.189651238,  0.0000000 ,  0.00000000, -0.30799645, -0.12785339,  0.0000000],
    [-0.1207336 ,  0.21463993,  0.00000000,  0.00000000, -0.1352284 ,  0.00539178,  0.175260991,  0.13475116,  1.215430907,  0.0000000 , -0.31554153, -0.35785540,  0.01388497,  0.0000000],
    [ 0.0000000 ,  0.10253270,  0.00000000,  0.18633327,  0.0000000 ,  0.00000000,  0.000000000,  0.00000000,  0.000000000,  1.1443461 ,  0.00000000, -0.26180538,  0.00000000, -0.1858336],
    [ 0.0000000 , -0.11545604,  0.00000000,  0.37425465,  0.1422056 ,  0.19380115,  0.024420731,  0.00000000,  0.203013509,  0.0000000 ,  1.27879614, -0.15774778, -0.01841561,  0.2722956],
    [ 0.1734789 , -0.01221252,  0.12136392, -0.18891657,  0.0000000 , -0.01772309,  0.063680439,  0.21399033,  0.236874674,  0.1840613 ,  0.09924378,  1.24296435, -0.09985941, -0.1829535],
    [ 0.1514365 , -0.12462269,  0.09085814,  0.02061674,  0.0000000 ,  0.27472100,  0.152847686,  0.09395161, -0.009720747,  0.0000000 ,  0.01225377,  0.06836207,  1.17520908,  0.0000000],
    [ 0.0000000 ,  0.11010724,  0.00000000,  0.18255412,  0.0000000 ,  0.18175864,  0.198942385,  0.00000000,  0.000000000,  0.1332174 , -0.17467603,  0.12074694,  0.00000000,  1.2190064]
])


Q,R = construct_matrices(P)
M = specialk(Q,R,P)

# Initialize matrices A and C
n = M.shape[0]
A = np.zeros((n, n))
C = np.zeros((n, n))

# Fill matrix A and matrix C
for i in range(n):
    # A[i, i] = M[i, i]  # sqrt of variances on the diagonal for matrix A
    A= np.diag(np.array([
    82.979884, 28.210062, 20.728451, 89.560848, 59.421006, 65.603921,
    46.097725, 27.293750, 27.058308, 35.157451, 33.469820, 21.267717,
    39.773886, 22.794063
]))
    for j in range(i):
        C[i, j] = C[j, i] = -M[i, j]  # negative partial correlations for matrix C

# Clustering for matrix C (lower triangular values)
matrix_data_C = C
lower_triangular_indices_C = np.tril_indices(matrix_data_C.shape[0], k=-1)
lower_triangular_values_C = matrix_data_C[lower_triangular_indices_C]

# Filter out zero values
nonzero_indices_C = np.nonzero(lower_triangular_values_C)[0]
lower_triangular_values_C_nonzero = lower_triangular_values_C[nonzero_indices_C]

# Generate unique labels for the dendrogram
lower_triangular_values_C_labels = [f"{value:.4f}" for value in lower_triangular_values_C_nonzero]

distance_threshold_C = 0.09  # Adjust the distance threshold as needed
clustering_C = AgglomerativeClustering(distance_threshold=distance_threshold_C, n_clusters=None, linkage='ward')
labels_C = clustering_C.fit_predict(lower_triangular_values_C_nonzero.reshape(-1, 1))

# Get the original row and column indices for the non-zero values
nonzero_row_col_indices_C = [(int(lower_triangular_indices_C[0][i] + 1), int(lower_triangular_indices_C[1][i] + 1)) for i in nonzero_indices_C]

# Clustering for matrix A (diagonal values)
matrix_data_A = A
diagonal_values_A = np.diag(matrix_data_A)

# Generate unique labels for the dendrogram
diagonal_values_A_labels = [f"{value:.4f}" for value in diagonal_values_A]

distance_threshold_A = 5  # Adjust the distance threshold as needed
clustering_A = AgglomerativeClustering(distance_threshold=distance_threshold_A, n_clusters=None, linkage='ward')
labels_A = clustering_A.fit_predict(diagonal_values_A.reshape(-1, 1))

# Group elements based on clustering results
grouped_elements_C = group_elements_with_labels(labels_C, nonzero_row_col_indices_C, element_labels)
grouped_elements_A = group_elements_with_labels(labels_A, [(i + 1, i + 1) for i in range(n)], element_labels)

print("Grouped Elements for C:")
for cluster_id, elements in grouped_elements_C.items():
    print(f"Cluster {cluster_id}: {elements}")

print("Grouped Elements for A:")
for cluster_id, elements in grouped_elements_A.items():
    print(f"Cluster {cluster_id}: {elements}")

# Plotting the dendrograms separately
# plot_dendrogram(clustering_C, title='Dendrogram for Lower Triangular Values (C)', labels=lower_triangular_values_C_labels)
# plot_dendrogram(clustering_A, title='Dendrogram for Diagonal Elements (A)', labels=diagonal_values_A_labels)
print("Done")