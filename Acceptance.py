import pandas as pd

# Load the CSV files
merged_model_results = pd.read_csv('merged_model_results.csv')
acceptance_summary = pd.read_csv('acceptance_summary.csv')

# Define the true edges and symmetries from the original model
true_edges = {"~X1:X5", "~X3:X4", "~X2:X5", "~X2:X6", "~X4:X5", "~X2:X3", "~X5:X6", "~X1:X2", "~X1:X4"}
true_symmetries = [
    {"~X1:X5", "~X3:X4"},
    {"~X2:X5", "~X2:X6", "~X4:X5"},
    {"~X2:X3", "~X5:X6"},
    {"~X1:X4"},
    {"~X1:X2"}
]

# Define the true VCC symmetries
true_vcc_symmetries = {"~X1", "~X2+X3+X6", "~X4+X5"}

# Function to parse detected symmetries and edges
def parse_symmetries(symmetry_str):
    return set(symmetry_str.replace(' ', '').split(';'))

# Function to check if detected edges match true edges
def check_edges(detected_edges):
    return detected_edges == true_edges

# Function to check if detected symmetries match true symmetries
def check_symmetries(detected_symmetries):
    detected_sets = [set(sym.split('+')) for sym in detected_symmetries]
    for true_sym in true_symmetries:
        if true_sym not in detected_sets:
            return False
    return True

# Initialize lists to store comparison results
correct_vcc_symmetries_count = []
correct_ecc_edges_count = []
correct_ecc_symmetries_count = []

# Iterate over the merged model results and compare with the true model
for index, row in merged_model_results.iterrows():
    detected_vcc = parse_symmetries(row['VCC'])
    detected_ecc = parse_symmetries(row['ECC'])
    
    # Check for correct VCC symmetries
    correct_vcc_symmetries = detected_vcc == true_vcc_symmetries
    correct_vcc_symmetries_count.append(int(correct_vcc_symmetries))
    
    # Check for correct ECC edges and symmetries separately
    correct_ecc_edges = check_edges(detected_ecc)
    correct_ecc_symmetries = check_symmetries(detected_ecc)
    correct_ecc_edges_count.append(int(correct_ecc_edges))
    correct_ecc_symmetries_count.append(int(correct_ecc_symmetries))

# Add the comparison results to the merged model results
merged_model_results['correct_vcc_symmetries'] = correct_vcc_symmetries_count
merged_model_results['correct_ecc_edges'] = correct_ecc_edges_count
merged_model_results['correct_ecc_symmetries'] = correct_ecc_symmetries_count

# Save the updated merged model results to a new CSV file
comparison_results_file = 'comparison_results.csv'
merged_model_results.to_csv(comparison_results_file, index=False)

# Calculate the overall acceptance rates based on the acceptance summary
acceptance_rates = acceptance_summary.copy()

# Ensure 'Sample' and 'Threshold' columns exist in both DataFrames and match correctly
merged_model_results['Sample_Size'] = merged_model_results['Sample'].str.extract(r'sample_size_(\d+)_')[0].astype(str)
merged_model_results['Threshold'] = merged_model_results['Threshold'].astype(str).str.strip()
acceptance_rates['Sample_Size'] = acceptance_rates['Sample_Size'].astype(str).str.strip()
acceptance_rates['Threshold'] = acceptance_rates['Threshold'].astype(str).str.strip()

# Merge the average correct symmetries and edges into the acceptance rates
grouped_results = merged_model_results.groupby(['Sample_Size', 'Threshold']).agg(
    correct_vcc_symmetries_avg=('correct_vcc_symmetries', 'mean'),
    correct_ecc_edges_avg=('correct_ecc_edges', 'mean'),
    correct_ecc_symmetries_avg=('correct_ecc_symmetries', 'mean')
).reset_index()

# Ensure 'Sample_Size' and 'Threshold' columns are of the same type and values match
grouped_results['Sample_Size'] = grouped_results['Sample_Size'].astype(str)
grouped_results['Threshold'] = grouped_results['Threshold'].astype(str)

# Merge with acceptance_rates
acceptance_rates = acceptance_rates.merge(
    grouped_results,
    on=['Sample_Size', 'Threshold'],
    how='left'
)

# Convert the averages to percentages
acceptance_rates['correct_vcc_symmetries_avg'] *= 100
acceptance_rates['correct_ecc_edges_avg'] *= 100
acceptance_rates['correct_ecc_symmetries_avg'] *= 100

# Save the final acceptance rates to a new CSV file
final_acceptance_rates_file = 'final_acceptance_rates.csv'
acceptance_rates.to_csv(final_acceptance_rates_file, index=False)

print("Comparison results saved successfully.")
print("Final acceptance rates saved successfully.")
