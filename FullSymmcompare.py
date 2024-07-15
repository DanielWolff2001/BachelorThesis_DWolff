import pandas as pd
from scipy.stats import chi2

# Define the file paths
merged_model_results_file = 'merged_model_results.csv'
full_model_logL_results_file = 'full_model_logL_results.csv'
output_file = 'comparisonfullandsymm.csv'

# Read the CSV files
merged_model_results = pd.read_csv(merged_model_results_file)
full_model_logL_results = pd.read_csv(full_model_logL_results_file)


print("Merged Model Results Columns: ", merged_model_results.columns)
print("Full Model LogL Results Columns: ", full_model_logL_results.columns)

# Define the columns we want to merge
columns_to_merge = ['Sample', 'Threshold', 'final_model_logL', 'new_dim', 'FullModelLogL', 'FullModelDimension']

# Merge the dataframes on the 'Sample' column
merged_data = pd.merge(
    merged_model_results[['Sample', 'Threshold', 'final_model_logL', 'new_dim']],
    full_model_logL_results[['Sample', 'FullModelLogL', 'FullModelDimension']],
    on='Sample'
)
# Calculate the likelihood ratio test statistic and p-value
merged_data['LikelihoodRatio'] = -2 * (merged_data['final_model_logL'] - merged_data['FullModelLogL'])
merged_data['df'] = merged_data['FullModelDimension'] - merged_data['new_dim']
merged_data['p_value'] = merged_data.apply(lambda row: chi2.sf(row['LikelihoodRatio'], df=row['df']), axis=1)

# Save the results to a new CSV file
merged_data.to_csv(output_file, index=False)

# Extract sample size from the 'Sample' column
merged_data['sample_size'] = merged_data['Sample'].apply(lambda x: x.split('_')[2])

# Group by sample size and threshold level to count the p-values larger than 0.01
summary = merged_data.groupby(['sample_size', 'Threshold']).apply(lambda x: (x['p_value'] > 0.01).sum()).reset_index()
summary.columns = ['Sample Size', 'Threshold', 'p_value_count']

# Print the summary in a readable format
print("Summary of p-values > 0.01 for each sample size and threshold level:")
for sample_size in summary['Sample Size'].unique():
    print(f"Sample Size: {sample_size}")
    for threshold in summary[summary['Sample Size'] == sample_size]['Threshold'].unique():
        count = summary[(summary['Sample Size'] == sample_size) & (summary['Threshold'] == threshold)]['p_value_count'].values[0]
        print(f"  Threshold {threshold}: {count} p-values > 0.01")

# Load the CSV file
merged_model_results_file = 'merged_model_results.csv'
merged_model_results = pd.read_csv(merged_model_results_file)

# Extract sample size from the 'Sample' column
merged_model_results['sample_size'] = merged_model_results['Sample'].apply(lambda x: x.split('_')[2])

# Group by sample size and threshold level to count the number of p-values above 100
p_value_summary_above_100 = merged_model_results.groupby(['sample_size', 'Threshold']).apply(lambda x: (x['Chi'] > 0.01).sum()).reset_index()
p_value_summary_above_100.columns = ['sample_size', 'Threshold', 'p_value_above_100_count']

# Display the summary
print(p_value_summary_above_100)
