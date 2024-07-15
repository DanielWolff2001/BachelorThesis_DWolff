import pandas as pd

# Load the CSV file from the specified path
file_path = 'merged_model_results.csv'
data = pd.read_csv(file_path)

# Define the true edges (ignoring the tilde)
true_edges = {"X1:X5", "X3:X4", "X2:X5", "X2:X6", "X4:X5", "X2:X3", "X5:X6", "X1:X2", "X1:X4"}

# Define the true symmetries
true_symmetries = [
    {"X1:X5", "X3:X4"},
    {"X2:X5", "X2:X6", "X4:X5"},
    {"X2:X3", "X5:X6"},
    {"X1:X4"},
    {"X1:X2"}
]

def extract_edges(ecc_string):
    # Normalize the ECC string by replacing + with ; and splitting on both
    edges = ecc_string.replace('+', ';').split('; ')
    processed_edges = set()
    for edge in edges:
        sub_edges = edge.split(';')
        for sub_edge in sub_edges:
            sub_edge = sub_edge.replace('~', '').strip()
            nodes = sub_edge.split(':')
            nodes.sort()  # Sort nodes within the edge
            processed_edges.add(f"{':'.join(nodes)}")
    return processed_edges

def extract_symmetries(ecc_string):
    # Split the ECC string by ';' to separate symmetry groups
    symmetry_groups = ecc_string.split('; ')
    processed_symmetries = []
    for group in symmetry_groups:
        edges = group.replace('~', '').split('+')
        normalized_edges = set()
        for edge in edges:
            edge = edge.strip()
            nodes = edge.split(':')
            nodes.sort()  # Sort nodes within the edge
            normalized_edges.add(f"{':'.join(nodes)}")
        processed_symmetries.append(normalized_edges)
    return processed_symmetries

def compare_symmetries(dataset_symmetries, true_symmetries):
    correct_symmetries = []
    for symmetry in dataset_symmetries:
        if symmetry in true_symmetries:
            correct_symmetries.append(symmetry)
    all_symmetries_correct = len(correct_symmetries) == len(dataset_symmetries)
    return correct_symmetries, all_symmetries_correct

def compare_edges(dataset_edges, true_edges):
    # Compare the extracted edges to the true edges and count the number of matches
    correct_matches = dataset_edges.intersection(true_edges)
    return len(correct_matches), len(dataset_edges), len(correct_matches) == len(true_edges)

# Apply the extraction functions to the ECC column
data['extracted_edges'] = data['ECC'].apply(extract_edges)
data['extracted_symmetries'] = data['ECC'].apply(extract_symmetries)

# Compare the extracted edges and symmetries to the true edges and true symmetries
data[['correct_edge_count', 'total_edge_count', 'edges_correct']] = data['extracted_edges'].apply(
    lambda x: pd.Series(compare_edges(x, true_edges))
)
data[['matched_symmetries', 'all_symmetries_correct']] = data['extracted_symmetries'].apply(
    lambda x: pd.Series(compare_symmetries(x, true_symmetries))
)

# Calculate the number of correct symmetries per sample
data['correct_symmetry_count'] = data['matched_symmetries'].apply(len)

# Extract sample number and size from the Sample column
data['sample_number'] = data['Sample'].apply(lambda x: x.split('_')[-1])
data['sample_size'] = data['Sample'].apply(lambda x: x.split('_')[2])
data['threshold_level'] = data['Threshold']

# Calculate extra edges
data['extra_edges'] = data['total_edge_count'] - 9

# Create the final dataframe
final_data = data[['sample_number', 'sample_size', 'threshold_level', 'ECC', 'edges_correct', 'matched_symmetries', 'correct_symmetry_count', 'all_symmetries_correct', 'extra_edges']].copy()
final_data.rename(columns={'ECC': 'ecc'}, inplace=True)

# Save the final dataframe to a new CSV file
final_data.to_csv('consolidated_samples_with_symmetries.csv', index=False)

# Display the first few rows of the final dataframe
print(final_data.head())

# Summary of correct edge and symmetry counts
total_rows = len(data)
matching_edges_rows = len(data[data['edges_correct']])
matching_symmetries_rows = data[data['edges_correct']]['correct_symmetry_count'].sum()
samples_all_symmetries_correct = len(data[data['all_symmetries_correct'] & data['edges_correct']])

print(f"Total rows: {total_rows}")
print(f"Total rows matching all edges: {matching_edges_rows} out of {total_rows}")
print(f"Total symmetries matched (only in rows with correct edges): {matching_symmetries_rows} instances")
print(f"Total samples with all symmetries correct: {samples_all_symmetries_correct} out of {matching_edges_rows}")

# Optionally, save the mismatched rows to a new CSV file for further inspection
mismatched_edges_rows = data[~data['edges_correct']]
mismatched_edges_rows.to_csv('mismatched_edges_rows.csv', index=False)

# Print one example of a sample that has a correct symmetry and one that does not
if not final_data[final_data['all_symmetries_correct']].empty:
    sample_correct_edges_and_symmetry = final_data[final_data['all_symmetries_correct']].iloc[0]
    print("\nSample with correct edges and symmetries:")
    print(sample_correct_edges_and_symmetry[['sample_number', 'sample_size', 'threshold_level', 'ecc', 'matched_symmetries', 'correct_symmetry_count']])
else:
    print("\nNo sample with all symmetries correct.")

if not final_data[~final_data['all_symmetries_correct']].empty:
    sample_correct_edges_incorrect_symmetry = final_data[~final_data['all_symmetries_correct']].iloc[0]
    print("\nSample with correct edges but incorrect symmetries:")
    print(sample_correct_edges_incorrect_symmetry[['sample_number', 'sample_size', 'threshold_level', 'ecc', 'matched_symmetries', 'correct_symmetry_count']])

# Summary of samples with all correct edges per sample size
summary_correct_edges_per_size = data[data['edges_correct']].groupby('sample_size').size()
print("\nSummary of samples with all correct edges per sample size:")
print(summary_correct_edges_per_size)

# Summary of correct symmetries per threshold level and sample size (only in rows with correct edges)
summary_correct_symmetries_per_size_threshold = data[data['edges_correct']].groupby(['sample_size', 'threshold_level'])['correct_symmetry_count'].sum()
print("\nSummary of correct symmetries per threshold level and sample size (only in rows with correct edges):")
print(summary_correct_symmetries_per_size_threshold)

# Count the number of samples with exactly 9 correct edges
samples_with_exactly_nine_edges = data[(data['edges_correct']) & (data['correct_edge_count'] == 9) & (data['total_edge_count'] == 9)]
num_samples_with_exactly_nine_edges = len(samples_with_exactly_nine_edges)

print(f"Number of samples with exactly 9 correct edges: {num_samples_with_exactly_nine_edges}")

# Breakdown of samples with exactly 9 correct edges by sample size
samples_with_exactly_nine_edges_by_size = samples_with_exactly_nine_edges.groupby('sample_size').size()
print("\nBreakdown of samples with exactly 9 correct edges by sample size:")
print(samples_with_exactly_nine_edges_by_size)

# Summary of how many correct symmetries found per sample size and threshold level
correct_symmetries_summary = data.groupby(['sample_size', 'threshold_level'])['correct_symmetry_count'].sum()
print("\nSummary of how many correct symmetries found per sample size and threshold level:")
print(correct_symmetries_summary)

# Calculate average number of extra edges per sample size and threshold level
average_extra_edges = data.groupby(['sample_size', 'threshold_level'])['extra_edges'].mean()
print("\nAverage number of extra edges per sample size and threshold level:")
print(average_extra_edges)

# Calculate total number of edges found and total correct edges found
total_edges_found = data.groupby(['sample_size', 'threshold_level'])['total_edge_count'].sum()
total_correct_edges_found = data.groupby(['sample_size', 'threshold_level'])['correct_edge_count'].sum()

print("\nTotal number of edges found per sample size and threshold level:")
print(total_edges_found)

print("\nTotal number of correct edges found per sample size and threshold level:")
print(total_correct_edges_found)
