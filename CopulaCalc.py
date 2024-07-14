import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata
from numpy.linalg import inv

# Load the dataset
file_path = '/Users/danielwolff/Library/Mobile Documents/com~apple~CloudDocs/TU/Jaar 3/BEP/Python/eeg_sample_1000_c.csv'
data = pd.read_csv(file_path)

# Compute variance for each column except the 'event' column
variances = data.drop(columns=['event']).var()

# Define tolerance for grouping variances
tolerance = 1e-3

# Group columns by similar variances
variance_groups = {}
for column, variance in variances.items():
    found_group = False
    for key in variance_groups:
        if np.abs(variance - key) < tolerance:
            variance_groups[key].append(column)
            found_group = True
            break
    if not found_group:
        variance_groups[variance] = [column]

# Transform each column to a uniform distribution using empirical probabilities
def to_uniform(data):
    return rankdata(data) / (len(data) + 1)

uniform_data = data.drop(columns=['event']).apply(to_uniform)

# Transform the uniform distributions to standard normal distributions
normal_data = uniform_data.apply(norm.ppf)

# Save the normal data into a DataFrame
EEG_1000_edited = normal_data
EEG_1000_uniform = uniform_data
# Save the DataFrame to a CSV file
EEG_1000_edited.to_csv('/Users/danielwolff/Library/Mobile Documents/com~apple~CloudDocs/TU/Jaar 3/BEP/Python/EEG_1000_edited.csv', index=False)
EEG_1000_uniform.to_csv('/Users/danielwolff/Library/Mobile Documents/com~apple~CloudDocs/TU/Jaar 3/BEP/Python/EEG_1000_uniform.csv', index=False)

# Calculate the covariance matrix
cov_matrix = normal_data.cov()

# Inverse of the covariance matrix (precision matrix)
precision_matrix = inv(cov_matrix)

# Partial correlations
partial_corr_matrix = -precision_matrix / np.sqrt(np.outer(np.diagonal(precision_matrix), np.diagonal(precision_matrix)))
np.fill_diagonal(partial_corr_matrix, 1)

# Convert to DataFrame for better readability
partial_corr_df = pd.DataFrame(partial_corr_matrix, index=cov_matrix.index, columns=cov_matrix.columns)

# Display the partial correlation matrix
print(partial_corr_df)
