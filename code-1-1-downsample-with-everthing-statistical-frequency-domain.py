# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:17:59 2024
@author: mhagh
"""

import scipy.io
import numpy as np
import h5py
import pandas as pd
import os
import math
from scipy.stats import kurtosis, skew
from scipy.signal import welch, find_peaks
from numpy.fft import fft, fftfreq


# Define feature extraction functions
def abs_energy(window):
    return np.sum(window ** 2)

def auc(window, fs=125):
    return np.trapz(window, dx=1/fs)

def autocorr(window):
    return np.correlate(window, window, mode='full')[len(window)-1]

def average_power(window, fs=125):
    return np.mean(window ** 2)

def calc_centroid(window, fs=125):
    freqs, psd = welch(window, fs=fs)
    return np.sum(freqs * psd) / np.sum(psd)

def calc_max(window):
    return np.max(window)

def calc_mean(window):
    return np.mean(window)

def calc_median(window):
    return np.median(window)

def calc_min(window):
    return np.min(window)

def calc_std(window):
    return np.std(window)

def calc_var(window):
    return np.var(window)

def distance(window):
    return np.sum(np.abs(np.diff(window)))

def ecdf(window, d=10):
    sorted_window = np.sort(window)
    return sorted_window[int(d * len(window) / 100)]

def ecdf_percentile(window, percentile=[0.2, 0.8]):
    sorted_window = np.sort(window)
    return [sorted_window[int(p * len(window))] for p in percentile]

def ecdf_percentile_count(window, percentile=[0.2, 0.8]):
    sorted_window = np.sort(window)
    return len([x for x in window if percentile[0] <= x <= percentile[1]])

def ecdf_slope(window, p_init=0.2, p_end=0.8):
    sorted_window = np.sort(window)
    return (sorted_window[int(p_end * len(window))] - sorted_window[int(p_init * len(window))]) / (p_end - p_init)

def entropy(window):
    p_window = np.abs(window) / np.sum(np.abs(window))
    return -np.sum(p_window * np.log2(p_window + 1e-12))

def max_frequency(window, fs=125):
    freqs = fftfreq(len(window), 1/fs)
    return freqs[np.argmax(np.abs(fft(window)))]

def max_power_spectrum(window, fs=125):
    freqs, psd = welch(window, fs=fs)
    return np.max(psd)

def mean_abs_deviation(window):
    return np.mean(np.abs(window - np.mean(window)))

def mean_abs_diff(window):
    return np.mean(np.abs(np.diff(window)))

def mean_diff(window):
    return np.mean(np.diff(window))

def median_abs_deviation(window):
    return np.median(np.abs(window - np.median(window)))

def median_abs_diff(window):
    return np.median(np.abs(np.diff(window)))

def median_diff(window):
    return np.median(np.diff(window))

def median_frequency(window, fs=125):
    freqs, psd = welch(window, fs=fs)
    cumulative_psd = np.cumsum(psd)
    return freqs[np.searchsorted(cumulative_psd, cumulative_psd[-1] / 2)]

def mse(window, m=2, maxscale=10, tolerance=0.15):
    return np.mean([(window[i:i + m] - np.mean(window[i:i + m])) ** 2 for i in range(len(window) - m)])

def neighbourhood_peaks(window, n=10):
    peaks, _ = find_peaks(window)
    return len(peaks[:n])

def petrosian_fractal_dimension(window):
    N_delta = np.sum(np.diff(window > np.median(window)).astype(int))
    return np.log(len(window)) / (np.log(len(window)) + np.log(len(window) / (len(window) + 0.4 * N_delta)))

def pk_pk_distance(window):
    return np.max(window) - np.min(window)

def power_bandwidth(window, fs=125):
    freqs, psd = welch(window, fs=fs)
    return freqs[np.searchsorted(np.cumsum(psd), np.sum(psd) * 0.95)]

def rms(window):
    return np.sqrt(np.mean(window ** 2))

def skewness(window):
    return skew(window)

def spectral_centroid(window, fs=125):
    freqs, psd = welch(window, fs=fs)
    return np.sum(freqs * psd) / np.sum(psd)

def spectral_decrease(window, fs=125):
    freqs, psd = welch(window, fs=fs)
    return np.mean(np.diff(psd) / psd[:-1])

def spectral_entropy(window, fs=125):
    freqs, psd = welch(window, fs=fs)
    psd_norm = psd / np.sum(psd)
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

def spectral_spread(window, fs=125):
    centroid = spectral_centroid(window, fs)
    freqs, psd = welch(window, fs=fs)
    return np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd))

def zero_cross(window):
    return np.sum(np.diff(window > 0).astype(int))


# Load the .mat file and set up parameters
file_path = 'C:/Dr-Elisa-Project/Data_NN_Masoud/Input_data.mat'

with h5py.File(file_path, 'r') as mat_file:
    if 'Accelerations' in mat_file:
        data = mat_file['Accelerations'][:]
    else:
        raise KeyError("Dataset 'Accelerations' not found in the provided .mat file.")

if len(data.shape) != 3:
    raise ValueError(f"Unexpected data shape: {data.shape}. Expected 3 dimensions.")

# Parameters
desired_time_steps = 1
original_time_steps = data.shape[1]
window_size = math.ceil(original_time_steps / desired_time_steps)

# Output directory for CSV files
output_dir = 'downsampled_data-114-mainonly-1-only-everything-3website_csv'
os.makedirs(output_dir, exist_ok=True)

# Feature extraction loop
for i in range(data.shape[0]):
    all_features = {
        'AbsEnergy': [], 'AUC': [], 'Autocorr': [], 'AveragePower': [], 'Centroid': [],
        'Max': [], 'Mean': [], 'Median': [], 'Min': [], 'StdDev': [], 'Variance': [],
        'Distance': [], 'ECDF': [], 'ECDFPercentile': [], 'ECDFPercentileCount': [],
        'ECDFSlope': [], 'Entropy': [], 'MaxFrequency': [], 'MaxPowerSpectrum': [], 
        'MeanAbsDeviation': [], 'MeanAbsDiff': [], 'MeanDiff': [], 'MedianAbsDeviation': [],
        'MedianAbsDiff': [], 'MedianDiff': [], 'MedianFrequency': [], 'MSE': [],
        'NeighbourhoodPeaks': [], 'PetrosianFractalDimension': [], 'PeakToPeakDistance': [],
        'PowerBandwidth': [], 'RMS': [], 'Skewness': [], 'SpectralCentroid': [],
        'SpectralDecrease': [], 'SpectralEntropy': [], 'SpectralSpread': [], 'ZeroCrossingRate': []
    }

    for j in range(data.shape[2]):
        for k in range(desired_time_steps):
            window_start = k * window_size
            window_end = min(window_start + window_size, original_time_steps)
            window = data[i, window_start:window_end, j]

            if len(window) > 0:
                all_features['AbsEnergy'].append(abs_energy(window))
                all_features['AUC'].append(auc(window))
                all_features['Autocorr'].append(autocorr(window))
                all_features['AveragePower'].append(average_power(window))
                all_features['Centroid'].append(calc_centroid(window))
                all_features['Max'].append(calc_max(window))
                all_features['Mean'].append(calc_mean(window))
                all_features['Median'].append(calc_median(window))
                all_features['Min'].append(calc_min(window))
                all_features['StdDev'].append(calc_std(window))
                all_features['Variance'].append(calc_var(window))
                all_features['Distance'].append(distance(window))
                all_features['ECDF'].append(ecdf(window))
                all_features['ECDFPercentile'].append(ecdf_percentile(window))
                all_features['ECDFPercentileCount'].append(ecdf_percentile_count(window))
                all_features['ECDFSlope'].append(ecdf_slope(window))
                all_features['Entropy'].append(entropy(window))
                all_features['MaxFrequency'].append(max_frequency(window))
                all_features['MaxPowerSpectrum'].append(max_power_spectrum(window))
                all_features['MeanAbsDeviation'].append(mean_abs_deviation(window))
                all_features['MeanAbsDiff'].append(mean_abs_diff(window))
                all_features['MeanDiff'].append(mean_diff(window))
                all_features['MedianAbsDeviation'].append(median_abs_deviation(window))
                all_features['MedianAbsDiff'].append(median_abs_diff(window))
                all_features['MedianDiff'].append(median_diff(window))
                all_features['MedianFrequency'].append(median_frequency(window))
                all_features['MSE'].append(mse(window))
                all_features['NeighbourhoodPeaks'].append(neighbourhood_peaks(window))
                all_features['PetrosianFractalDimension'].append(petrosian_fractal_dimension(window))
                all_features['PeakToPeakDistance'].append(pk_pk_distance(window))
                all_features['PowerBandwidth'].append(power_bandwidth(window))
                all_features['RMS'].append(rms(window))
                all_features['Skewness'].append(skewness(window))
                all_features['SpectralCentroid'].append(spectral_centroid(window))
                all_features['SpectralDecrease'].append(spectral_decrease(window))
                all_features['SpectralEntropy'].append(spectral_entropy(window))
                all_features['SpectralSpread'].append(spectral_spread(window))
                all_features['ZeroCrossingRate'].append(zero_cross(window))

    # Save each set to CSV in Google Drive
    df = pd.DataFrame(all_features)
    output_file = os.path.join(output_dir, f'time_series_set_{i + 1}_features.csv')
    df.to_csv(output_file, index=False)
    print(f"Features for time series set {i + 1} saved to {output_file}")
