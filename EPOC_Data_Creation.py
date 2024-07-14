import pandas as pd
import numpy as np
import time

# Start the timer
start_time = time.time()

# Define the path to your text file
file_path = 'EP1.01.txt'

# Read the text file
df = pd.read_csv(file_path, sep='\t', header=None, names=['id', 'event', 'device', 'channel', 'code', 'size', 'data'])

# Take the first 280 rows, which corresponds to 20 events with 14 channels each
n_events = 1000
n_rows = 14*n_events
df_sample = df.iloc[:n_rows]

# Convert the 'data' column from string to a list of floats
df_sample['data'] = df_sample['data'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

# Initialize a list to hold the expanded data
expanded_data_list = []

# Iterate through each row and expand the data
for index, row in df_sample.iterrows():
    data_length = len(row['data'])
    temp_df = pd.DataFrame({
        'event': row['event'],
        'channel': row['channel'],
        'value': row['data']
    })
    expanded_data_list.append(temp_df)

# Concatenate all DataFrames in the list at once
expanded_data = pd.concat(expanded_data_list, ignore_index=True)

# Aggregate the data to have one row per event and one column per channel, taking the mean of the values
pivot_df = expanded_data.groupby(['event', 'channel'])['value'].apply(np.mean).unstack()

# Fill missing values with the mean of each channel
pivot_df = pivot_df.apply(lambda x: x.fillna(x.mean()), axis=0)

# Reset the index to make 'event' a regular column
pivot_df.reset_index(inplace=True)

# Center the data to mean zero
pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].apply(lambda x: x - x.mean(), axis=0)

# End the timer
end_time = time.time()

# Calculate the duration
duration = end_time - start_time
print(f"Processing time: {duration} seconds")

# Save the processed DataFrame for further use
pivot_df.to_csv('eeg_sample_100_centered.csv', index=False)

# Display the first few rows of the DataFrame
print(pivot_df.head())

# Compute the mean vector and covariance matrix across events
mean_vector = pivot_df.drop(columns=['event']).mean()
cov_matrix = pivot_df.drop(columns=['event']).cov()

# Save the covariance matrix to a CSV file
cov_matrix.to_csv('covmatrix_EEG_centered.csv')

print("Mean Vector:\n", mean_vector)
print("Covariance Matrix:\n", cov_matrix)

# Plot the distribution of values for each channel
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 10))
sns.boxplot(data=pivot_df.drop(columns=['event']))
plt.title('Distribution of EEG Values Across Channels (Centered)')
plt.xlabel('Channel')
plt.ylabel('EEG Value')
plt.xticks(rotation=90)
plt.show()
