# --------------------
# Import Necessary Libraries
# --------------------
import h5py
import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import resample

# --------------------
# Load the .mat File and Set Up Parameters
# --------------------
file_path = 'C:/Dr-Elisa-Project/Data_NN_Masoud/Input_data.mat'

# Load the .mat file
with h5py.File(file_path, 'r') as mat_file:
    if 'Accelerations' in mat_file:
        data = mat_file['Accelerations'][:]
    else:
        raise KeyError("Dataset 'Accelerations' not found in the provided .mat file.")

# Ensure data has the expected 3 dimensions
if len(data.shape) != 3:
    raise ValueError(f"Unexpected data shape: {data.shape}. Expected 3 dimensions.")

# Extract dimensions
num_sensors = data.shape[0]      # Number of sensors (114)
num_time_steps = data.shape[1]   # Time steps per dataset
num_datasets = data.shape[2]     # Number of datasets (73)

# Define the subset of sensors to use for MI calculation
set_numbers = [1, 3, 11, 13, 19, 26, 28, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 53, 54, 55, 56,
               57, 60, 61, 66, 68, 69, 71, 73, 74, 76, 77, 82, 87, 88, 92, 93, 95, 97, 98, 100, 103, 105, 107, 108, 110, 111]
# Adjust set_numbers to 0-based indexing for Python
set_numbers = [s - 1 for s in set_numbers]

# Output directory for results
output_dir = 'sensor_analysis_downsampled_100_mi'
os.makedirs(output_dir, exist_ok=True)

# --------------------
# Downsample Data to 100 Points
# --------------------
downsampled_data = np.zeros((len(set_numbers), 100, num_datasets))  # Shape: (num_sensors, 100 points, num_datasets)

for dataset_idx in range(num_datasets):
    for sensor_idx, sensor_num in enumerate(set_numbers):
        # Downsample the time-series data to 100 points
        downsampled_data[sensor_idx, :, dataset_idx] = resample(data[sensor_num, :, dataset_idx], 100)

print(f"Downsampled data shape: {downsampled_data.shape}")  # Should print (num_sensors, 100, num_datasets)

# --------------------
# Calculate MI Between Each Sensor and All Other Sensors
# --------------------
# Initialize a dictionary to store MI values for each dataset
mi_results = []

for dataset_idx in range(num_datasets):
    dataset_results = {}
    current_dataset = downsampled_data[:, :, dataset_idx]  # Shape: (num_sensors, 100)
    for i, sensor_i in enumerate(set_numbers):
        dataset_results[sensor_i] = {}
        for j, sensor_j in enumerate(set_numbers):
            if i != j:  # Skip self-MI
                x = current_dataset[i, :].reshape(-1, 1)  # Downsampled data for sensor i
                y = current_dataset[j, :]  # Downsampled data for sensor j
                try:
                    mi = mutual_info_regression(x, y, n_neighbors=3)  # Use n_neighbors=3 for robustness
                    dataset_results[sensor_i][sensor_j] = mi[0]
                except ValueError as e:
                    print(f"[Dataset {dataset_idx + 1}] Error calculating MI between "
                          f"Sensor {sensor_i + 1} and Sensor {sensor_j + 1}: {e}")
                    dataset_results[sensor_i][sensor_j] = np.nan
    mi_results.append(dataset_results)

# --------------------
# Save the MI Results
# --------------------
# Convert the results into a DataFrame for each dataset
for dataset_idx, dataset_result in enumerate(mi_results):
    mi_df = pd.DataFrame.from_dict(dataset_result, orient='index').fillna(0)
    mi_output_file = os.path.join(output_dir, f"mutual_information_dataset_{dataset_idx + 1}.csv")
    mi_df.to_csv(mi_output_file)
    print(f"MI values for Dataset {dataset_idx + 1} saved to {mi_output_file}")

# --------------------
# Visualization: Heatmap of MI for One Dataset (Example: Dataset 1)
# --------------------
example_dataset = 0  # Example: First dataset
example_mi_df = pd.DataFrame.from_dict(mi_results[example_dataset], orient='index').fillna(0)

plt.figure(figsize=(20, 15))  # Increase figure size for better readability
sns.heatmap(
    example_mi_df, cmap="YlOrBr",
    xticklabels=[f"Sensor_{sensor + 1}" for sensor in set_numbers],
    yticklabels=[f"Sensor_{sensor + 1}" for sensor in set_numbers],
    annot=False
)
plt.title(f"Mutual Information Heatmap (Dataset {example_dataset + 1})", fontsize=18)
plt.xlabel("Sensor", fontsize=14)
plt.ylabel("Sensor", fontsize=14)
plt.xticks(fontsize=10, rotation=90)
plt.yticks(fontsize=10)

# Save the heatmap
heatmap_output_file = os.path.join(output_dir, f"mi_heatmap_dataset_{example_dataset + 1}.png")
plt.savefig(heatmap_output_file, dpi=300, bbox_inches='tight')
print(f"Heatmap for Dataset {example_dataset + 1} saved to {heatmap_output_file}")

plt.show()
