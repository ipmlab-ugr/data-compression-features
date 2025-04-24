# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:20:48 2024

@author: mhagh
"""

import scipy.io
import numpy as np
import h5py
import pandas as pd
import os
from scipy.stats import kurtosis, skew

# Load the .mat file
file_path = 'C:/Dr-Elisa-Project/Data_NN_Masoud/Input_data.mat'

with h5py.File(file_path, 'r') as mat_file:
    # Check the variable names to find the one you need
    for key in mat_file.keys():
        print(key)

    # Load the specific dataset; adjust 'Accelerations' to the actual variable name
    data = mat_file['Accelerations'][:]

# Parameters
desired_time_steps = 1# Define the number of time steps you want to downsample to
original_time_steps = data.shape[1]
window_size = original_time_steps // desired_time_steps  # Define the window size

# Define the output directory for CSV files
output_dir = 'downsampled_data-114-mainonly-1-only-std_csv'
os.makedirs(output_dir, exist_ok=True)

# Iterate through each time series set, process, and save separately
for i in range(data.shape[0]):  # Iterate over each time series set
    all_means = []
    all_stds = []
    all_kurtoses = []
    all_skewness = []
    all_medians = []
    all_ranges = []
    all_mads = []  # Mean Absolute Deviation
    
    for j in range(data.shape[2]):  # Iterate over each sensor
        for k in range(desired_time_steps):
            # Extract the window using sliding window approach
            window_start = k * window_size
            window_end = window_start + window_size
            window = data[i, window_start:window_end, j]

            # Calculate various statistical features for the window
            window_mean = np.mean(window)
            window_std = np.std(window)
            window_kurtosis = kurtosis(window)
            window_skewness = skew(window)
            window_median = np.median(window)
            window_range = np.max(window) - np.min(window)
            window_mad = np.mean(np.abs(window - window_mean))

            # Append the values to their respective lists
            all_means.append(window_mean)
            all_stds.append(window_std)
            all_kurtoses.append(window_kurtosis)
            all_skewness.append(window_skewness)
            all_medians.append(window_median)
            all_ranges.append(window_range)
            all_mads.append(window_mad)

    # Combine the lists into a DataFrame for this specific time series set
    df = pd.DataFrame({
        'Mean': all_means,
        'StdDev': all_stds,
        'Kurtosis': all_kurtoses,
        'Skewness': all_skewness,
        'Median': all_medians,
        'Range': all_ranges,
        'MAD': all_mads
    })

    # Save each DataFrame to a separate CSV file
    output_file = os.path.join(output_dir, f'time_series_set_{i + 1}_features.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Features for time series set {i + 1} saved to {output_file}")
