# Data-compression-features
A Python library for time series signal processing, featuring downsampling routines and a rich set of statistical, spectral, and nonlinear feature extraction functions. Designed for .mat datasets with multi-channel acceleration data, this toolkit enables efficient preprocessing for machine learning and signal analysis workflows.
# Overview
This Python script performs feature extraction on time series data, specifically acceleration data stored in a .mat file. The script uses various signal processing techniques and statistical methods to extract a wide range of features that can be used for machine learning tasks, particularly in predictive modeling and analysis of mechanical or structural systems.

# Requirements
The following Python packages are required to run this script:

- scipy: For statistical functions, signal processing, and Fourier transforms.

- numpy: For numerical operations.

- h5py: For reading .mat files.

- pandas: For handling data in a tabular format (e.g., saving features to CSV).

- os: For handling file paths and directories.

- math: For basic mathematical operations
- Features Extracted
The script extracts the following features from each window of the time series data:

# 1. Time-Domain Features:
AbsEnergy: The absolute energy of the window, calculated as the sum of squared values.

AUC (Area Under Curve): The area under the curve of the window, calculated using numerical integration (trapezoidal rule).

Autocorr: The autocorrelation of the window, a measure of similarity with a lag.

AveragePower: The average power of the window (mean of squared values).

Max: Maximum value of the window.

Mean: Mean value of the window.

Median: Median value of the window.

Min: Minimum value of the window.

StdDev: Standard deviation of the window.

Variance: Variance of the window.

Distance: The total distance (sum of absolute differences between consecutive points).

ZeroCrossingRate: The number of times the signal crosses zero.

# 2. Statistical Features:
ECDF (Empirical Cumulative Distribution Function): The sorted values of the window.

ECDFPercentile: Percentile values from the sorted ECDF (default 20% and 80%).

ECDFPercentileCount: Number of values within the specified ECDF percentiles.

ECDFSlope: The slope of the ECDF between the 20% and 80% percentiles.

Entropy: Entropy of the signal, a measure of its unpredictability.

Mutual Information: Mutual information is a measure of the amount of information that one channel contains about another.

# 3. Frequency-Domain Features:
MaxFrequency: The frequency with the maximum power in the signal's frequency spectrum (using FFT).

MaxPowerSpectrum: The maximum value of the power spectral density.

MedianFrequency: The frequency at which the cumulative power spectrum reaches 50%.

PowerBandwidth: The bandwidth that contains 95% of the signal's power.

SpectralCentroid: The "center of mass" of the power spectrum, indicating the balance point.

SpectralDecrease: The rate of decrease of the power spectrum.

SpectralEntropy: Entropy of the power spectrum, indicating complexity.

SpectralSpread: The spread (variance) of the frequency spectrum around the spectral centroid.

# 4. Other Features:
NeighbourhoodPeaks: The number of peaks in the signal within the first 'n' values.

PetrosianFractalDimension: A measure of the fractal dimension of the signal using a specific algorithm.

PeakToPeakDistance: The distance between the maximum and minimum values in the window.

RMS (Root Mean Square): The root mean square of the window.

Skewness: The skewness (asymmetry) of the window’s distribution.

# 5. Higher-Order Statistical Features:
MeanAbsDeviation: The mean absolute deviation from the mean.

MeanAbsDiff: The mean of absolute differences between consecutive points.

MeanDiff: The mean difference between consecutive points.

MedianAbsDeviation: The median absolute deviation from the median.

MedianAbsDiff: The median of absolute differences between consecutive points.

MedianDiff: The median difference between consecutive points.

6. Miscellaneous:
MSE (Mean Squared Error): The mean squared error between the values in the window.

# Usage
Loading Data:
The .mat file is expected to contain a dataset named 'Accelerations'.

The script loads the data using the h5py library. Ensure that the .mat file structure contains a key 'Accelerations' for the script to proceed.

# Parameters:
desired_time_steps: The target time step frequency after downsampling.

original_time_steps: The original number of time steps in the data.

window_size: The size of each window used for feature extraction, determined by the original time steps and desired time steps.

The features are extracted for each window of the data.

# Output:
The extracted features for each data sample are saved in a CSV file within the specified output directory.
