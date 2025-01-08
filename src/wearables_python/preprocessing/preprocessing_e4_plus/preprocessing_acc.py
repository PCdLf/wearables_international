# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:11:40 2025

@author: selaca
"""

import warnings
import os
import sys
import pandas as pd
import numpy as np

from zipfile import ZipFile

from itertools import chain


import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


from datetime import datetime,timezone,timedelta
import time

from scipy.signal import welch
from scipy.stats import entropy

from scipy.signal import cheby2, sosfilt, sosfreqz
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter,filtfilt 

from scipy.interpolate import interp1d
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
from scipy.signal import fftconvolve
import heartpy as hp

from scipy.stats import zscore
from scipy.signal import correlate
from scipy import stats

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, SpectralClustering, MeanShift, AffinityPropagation
from sklearn.metrics.pairwise import rbf_kernel

from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN, AffinityPropagation
from sklearn.manifold import Isomap, TSNE, SpectralEmbedding
from scipy.sparse import csr_matrix
import umap
from sklearn.metrics import pairwise_distances_argmin_min
from antropy import perm_entropy
#from nolds import sampen
#from nolds import hurst_rs
from scipy.stats import entropy
#from nolds import lyap_r
from antropy import lziv_complexity
# Example: Combined ZCR and MAD to refine flat region detection
from scipy.stats import zscore

import pywt
from scipy.signal import butter, filtfilt

import warnings
from collections import Counter
# Suppress all warnings
warnings.filterwarnings("ignore")


def lowpass_filter(data, highcut, fs, order=2):
    nyquist = 0.5 * fs
    high = highcut / nyquist
    b, a = butter(order, high, btype='low')
    y = filtfilt(b, a, data)
    return y

# Define the bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Apply moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Apply Z-score normalization
def normalize_data(data):
    return zscore(data)



def compute_hjorth_parameters(data):
    first_derivative = np.diff(data)
    second_derivative = np.diff(first_derivative)
    var_zero = np.var(data)
    var_d1 = np.var(first_derivative)
    var_d2 = np.var(second_derivative)
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility
    return mobility, complexity


def compute_statistical_features(data_list):
    features = {}
    features['mean'] = np.mean(data_list)
    features['std'] = np.std(data_list)
    features['aad'] = np.mean(np.absolute(data_list - np.mean(data_list)))  # Average Absolute Deviation
    features['min'] = np.min(data_list)
    features['max'] = np.max(data_list)
    features['maxmin_diff'] = np.max(data_list) - np.min(data_list)
    features['median'] = np.median(data_list)
    features['mad'] = np.median(np.absolute(data_list - np.median(data_list)))  # Median Absolute Deviation
    features['IQR'] = np.percentile(data_list, 75) - np.percentile(data_list, 25)  # Interquartile Range
    features['neg_count'] = np.sum(data_list < 0)
    features['pos_count'] = np.sum(data_list > 0)
    features['above_mean'] = np.sum(data_list > np.mean(data_list))
    
    # Peak detection with prominence and distance parameters for robustness
    peaks, _ = find_peaks(data_list, prominence=1, distance=10)
    features['peak_count'] = len(peaks)
    
    features['skewness'] = stats.skew(data_list)
    features['kurtosis'] = stats.kurtosis(data_list)
    features['energy'] = np.sum(data_list**2) / len(data_list)  # Signal energy
    features['zero_crossing_rate'] = ((data_list[:-1] * data_list[1:]) < 0).sum()
    features['rms'] = np.sqrt(np.mean(data_list**2))
    """
    features['permutation_entropy'] = perm_entropy(data_list, order=3, normalize=True)
    features['hjorth_mobility'], features['hjorth_complexity'] = compute_hjorth_parameters(data_list)
    features['approximate_entropy'] = sampen(data_list)
    features['fractal_dimension'] = hurst_rs(data_list)
    features['signal_range'] = np.ptp(data_list)  # Equivalent to max - min
    features['snr'] = np.mean(data_list) / np.std(data_list)
    features['variance_abs_diff'] = np.var(np.abs(np.diff(data_list)))
    
    value_counts = np.histogram(data_list, bins=10)[0] + 1  # Avoid zero counts
    features['shannon_entropy'] = entropy(value_counts / sum(value_counts))
    
    
    # Calculate Lempel-Ziv Complexity, convert data to a list if needed
    try:
       features['lempel_ziv_complexity'] = lziv_complexity(data_list.tolist(), normalize=True)
    except Exception as e:
       print(f"Error calculating Lempel-Ziv complexity: {e}")
       features['lempel_ziv_complexity'] = np.nan  # Or handle as you see fit
       
     
    features['hurst_exponent'] = hurst_rs(data_list)
    
   
    coeffs = pywt.wavedec(data_list, 'db1', level=3)
    features['wavelet_cA3'] = np.mean(coeffs[0])  # Approximation coefficient at level 3
    features['wavelet_cD1'] = np.mean(coeffs[1])  # Detail coefficient at level 1

    features['wavelet_energy'] = np.sum([np.sum(np.square(c)) for c in coeffs])

    # Ensure data_list is an array
    data_array = np.asarray(data_list)
    features['sma'] = np.sum(np.abs(data_array))
    features['slope'] = (data_list[-1] - data_list[0]) / len(data_list)
    """
    #features['angle_gravity'] = np.arccos(data_list[:, 2] / np.linalg.norm(data_list, axis=1)).mean()
    jerk_signal = np.diff(data_list, axis=0)
    features['jerk_mean'] = np.mean(jerk_signal)
    features['jerk_std'] = np.std(jerk_signal)


    return features

def compute_frequency_features(data_list, sampling_rate):
    features = {}
    
    # Power Spectral Density (PSD) is simply the square of the normalized FFT (acc_fft)
    psd = data_list ** 2  # No need for further normalization since acc_fft is already normalized
    #features['psd'] = psd  # Keep this for future use if needed, though it's not a typical feature
    
    # Dominant frequency: the index of the highest power in the PSD
    features['dominant_freq'] = np.argmax(psd)  # This gives the index of the dominant frequency
    
    # Normalizing the PSD for spectral entropy calculation, avoid division by zero
    psd_norm = psd / np.sum(psd) + 1e-10  # Adding a small constant to prevent log(0)
    
    # Spectral entropy: measure of the spread of power across frequencies
    features['spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm))
    
    # Spectral energy: total energy in the frequency domain
    features['spectral_energy'] = np.sum(psd)
    
    # Frequencies corresponding to FFT values
    freqs = np.fft.fftfreq(len(data_list), d=1/sampling_rate)[:len(data_list)]
    
    # Spectral centroid: the weighted mean of the frequency components
    features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
    
    # Spectral bandwidth: the spread of the power spectrum around the centroid
    features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * psd) / np.sum(psd))
    
    # Spectral flatness: how "flat" the power spectrum is (tone-like vs. noise-like)
    features['spectral_flatness'] = np.exp(np.mean(np.log(psd + 1e-10))) / np.mean(psd)  # Avoid log(0) issues with small constant
    
    features['spectral_flux'] = np.sqrt(np.sum(np.diff(psd)**2)) / len(psd)
    
    cumulative_psd = np.cumsum(psd)
    features['spectral_rolloff'] = freqs[np.where(cumulative_psd >= 0.85 * cumulative_psd[-1])[0][0]]
    features['spectral_skewness'] = stats.skew(psd)
    features['spectral_kurtosis'] = stats.kurtosis(psd)
    features['spectral_variability'] = np.var(psd)
   
    return features



def calculate_sampling_rate(df, timestamp_column):
    """
    Calculate the sampling rate of a signal given a DataFrame with timestamps.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the signal data.
    timestamp_column (str): The name of the column containing timestamp values.

    Returns:
    float: The sampling rate in Hertz (samples per second).
    """
    # Ensure the timestamps are sorted
    df = df.sort_values(by=timestamp_column)

    # Calculate the time differences between consecutive timestamps
    time_diffs = df[timestamp_column].diff().dropna()

    # Calculate the average sampling interval (in seconds)
    avg_sampling_interval = time_diffs.mean().total_seconds()

    # Calculate the sampling rate in Hz
    sampling_rate = 1.0 / avg_sampling_interval

    return sampling_rate


def resample_dataframe(data,target_samp_freq='31.25L'):
    # 1. Ensure the timestamp column is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # 2. Set the timestamp column as the DataFrame index
    data.set_index('timestamp', inplace=True)
    
    # 3. Resample the data to 32 Hz (every 31.25 milliseconds)
    resampled_data = data.resample(target_samp_freq).mean()  # 'L' stands for milliseconds
    
    # 4. Interpolate missing values if needed
    resampled_data = resampled_data.interpolate(method='linear')
    
    # Reset the index if needed (optional)
    resampled_data.reset_index(inplace=True)
    
    return resampled_data
    
# Function to apply t-SNE and plot clustering results
def plot_tsne_results(data, clusters, title, perplexity=30, learning_rate=200, n_iter=1000):
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    tsne_features_2d = tsne_2d.fit_transform(data)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_features_2d[:, 0], tsne_features_2d[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar()
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(f"{title}\n(perplexity={perplexity}, learning_rate={learning_rate}, n_iter={n_iter})")
    plt.show()

    



def plot_umap_results(features, labels, title, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    """
    Function to plot UMAP results for clustering visualization.
    
    Parameters:
    - features: The feature matrix (data to be reduced).
    - labels: The cluster labels to color the points.
    - title: Title for the plot.
    - n_neighbors: Number of neighbors to consider for UMAP (default 15).
    - min_dist: Minimum distance between embedded points (default 0.1).
    - n_components: Number of UMAP components for dimensionality reduction (default 2 for 2D plot).
    - random_state: Random seed for reproducibility.
    """
    # Apply UMAP for dimensionality reduction
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    umap_results = umap_model.fit_transform(features)
    
    # Plot UMAP results
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()




 # Low-pass filter design
def lowpass_filter(data, cutoff, fs, order=4):
     nyquist = 0.5 * fs
     normal_cutoff = cutoff / nyquist
     b, a = butter(order, normal_cutoff, btype='low', analog=False)
     return filtfilt(b, a, data)
 

def flat_check(segment_data, col_name, flat_variance_threshold=1e-4, zcr_threshold=0.05, mad_threshold=0.01, window_size=32, flat_percentage_threshold=20):
    magnitude = segment_data[col_name]
    rolling_variance = magnitude.rolling(window=window_size).var()
    rolling_mad = magnitude.rolling(window=window_size).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    zero_crossings = magnitude.diff().apply(np.sign).diff().fillna(0) != 0
    rolling_zcr = zero_crossings.rolling(window=window_size).mean()

    # Flat region criterion: low variance, low ZCR, and low MAD
    is_flat_region = (rolling_variance < flat_variance_threshold) & (rolling_mad < mad_threshold) & (rolling_zcr < zcr_threshold)

    # Calculate flat region percentage and check against threshold
    flat_region_percentage = is_flat_region.mean() * 100
    is_flat = flat_region_percentage > flat_percentage_threshold

    if is_flat:
        print(f"Warning: Flat region percentage is {flat_region_percentage:.2f}% - exceeds threshold of {flat_percentage_threshold}%")
    return flat_region_percentage, is_flat


def process(acc_df): 
    ######################## LOAD DATA ############################## 

    #sampling_rate = calculate_sampling_rate(acc_df, 'timestamp')
    #print(f"Sampling rate: {sampling_rate:.2f} Hz")
    
    acc_df['timestamp'] = pd.to_datetime(acc_df['timestamp'])
    acc_df['timestamp'] = acc_df['timestamp'].dt.tz_localize(None)
    acc_df = acc_df.sort_values('timestamp')
    
    ############################ FIND CONTIONUS PERIODS WHERE THE DEVICE IS WORN ###############################
    # Assuming 'timestamp' is in datetime format in 'merged_df'
    acc_df['timestamp'] = pd.to_datetime(acc_df['timestamp'])
    
    # Calculate time differences between consecutive rows
    acc_df['time_diff'] = acc_df['timestamp'].diff().dt.total_seconds()
    
    # Define a threshold for identifying gaps (e.g., 60 seconds of no data indicates discontinuity)
    gap_threshold = 60  # 60 seconds
    
    # Create a new column 'segment_id' to mark continuous segments
    # We increment the segment ID whenever there's a gap larger than the threshold
    acc_df['segment_id'] = (acc_df['time_diff'] > gap_threshold).cumsum()
    
    # Optional: Drop the 'time_diff' column if no longer needed
    acc_df = acc_df.drop(columns=['time_diff'])
    
    ############################ PROCESS EACH SEGMENT SEPARATELY ##############################################
    segment_features_list = {}
    plot_count=0
   
    # Group by 'segment_id' and process each segment separately
    for segment_id, segment_data in acc_df.groupby('segment_id'):
        ################## RESAMPLE TO MAKE SURE IT IS 32 Hz ################
        
        segment_data=resample_dataframe(segment_data,target_samp_freq='31.25L')    
      
        ####################### COMPUTE MAGNITUDE ################################
        segment_data['Magnitude'] = np.sqrt(segment_data['x']**2 + segment_data['y']**2 + segment_data['z']**2)
        
        ######################### MAGNITUDE NORMALIZED ################################
        # Normalize the data by subtracting the mean and dividing by the standard deviation
        magnitude_data = (segment_data['Magnitude'] - segment_data['Magnitude'].mean()) / segment_data['Magnitude'].std()
        segment_data['Magnitude_normalized'] = magnitude_data

        ####################### PLOT THE ORIGINAL SIGNAL ########################
        """
        # Plot the accelerometer signal
        plt.figure(figsize=(10, 6))
        plt.plot(segment_data['timestamp'], segment_data['Magnitude_normalized'], label='Normalized ACC Magnitude')
        plt.xlabel('Time')
        plt.ylabel('Accelerometer')
        plt.title(f'Accelerometer Signal After Normalizing Magnitude {segment_id}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout() 
        plt.show()
        """
        
        
        ##################### MAGNITUDE BANDPASS FILTERED #####################
        # Apply the lowpass filter to the smoothed signal
        sampling_rate = 64  # Hz
        lowcut = 0.5        # Lower threshold frequency (Hz)
        highcut = 10        # Upper threshold frequency (Hz)
        # Drop NaN values and filter the signal
        magnitude_data = segment_data['Magnitude_normalized'].dropna().to_numpy()
        filtered_data = bandpass_filter(magnitude_data, lowcut, highcut, fs=sampling_rate)
    
        # Create a new column in merged_df with NaN values
        segment_data['Magnitude_filtered'] =filtered_data
        """
        # Plot the accelerometer signal
        plt.figure(figsize=(10, 6))
        plt.plot(segment_data['timestamp'], segment_data['Magnitude_filtered'], label='Filtered ACC Magnitude')
        plt.xlabel('Time')
        plt.ylabel('Accelerometer')
        plt.title(f'Accelerometer Signal After Band-Pass Filtering Magnitude {segment_id}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout() 
        plt.show()
        """
        #################### MAGNITUDE LOWPASS FILTERING ##########################
        
        # Apply low-pass filter after bandpass
        cutoff = 8  # Set cutoff frequency for low-pass filter
        magnitude_lowpassed = lowpass_filter(filtered_data, cutoff, sampling_rate, order=6)
        
        segment_data['Magnitude_lowpass']=magnitude_lowpassed
        
        """
        # Plot the accelerometer signal
        plt.figure(figsize=(10, 6))
        plt.plot(segment_data['timestamp'], segment_data['Magnitude_lowpass'], label='Filtered ACC Magnitude')
        plt.xlabel('Time')
        plt.ylabel('Accelerometer')
        plt.title(f'Accelerometer Signal After Low-Pass Filtering Magnitude {segment_id}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout() 
        plt.show()
        """
        
        ##################### MEDIAN FILTERING #################################
        from scipy.signal import medfilt

        window_size = 3  # Choose an odd number, e.g., 3 or 5
        magnitude_med_smoothed = medfilt(magnitude_lowpassed, kernel_size=window_size)
        segment_data['Magnitude_Median_Filtered'] = magnitude_med_smoothed
           
        """
        # Plot the accelerometer signal
        plt.figure(figsize=(10, 6))
        plt.plot(segment_data['timestamp'], segment_data['Magnitude_Median_Filtered'], label='Filtered ACC Magnitude')
        plt.xlabel('Time')
        plt.ylabel('Accelerometer')
        plt.title(f'Accelerometer Signal After Median Filtering Magnitude {segment_id}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout() 
        plt.show()
        """

        ######################## MAGNITUDE SMOOTH ############################
       
        segment_data['Magnitude_smoothed'] = segment_data['Magnitude_Median_Filtered'].rolling(window=7).mean()  # Use a window size suitable for your data
        segment_data=segment_data.dropna()
        segment_data.reset_index(drop=True, inplace=True)
        
        """
        # Plot the accelerometer signal
        plt.figure(figsize=(10, 6))
        plt.plot(segment_data['timestamp'], segment_data['Magnitude_smoothed'], label='Smoothed ACC Magnitude')
        plt.xlabel('Time')
        plt.ylabel('Accelerometer')
        plt.title(f'Accelerometer Signal After Smoothing Magnitude {segment_id}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout() 
        plt.show()
        """

        ################### MAGNITUDE SAVITSYKY GOLAY FILTER ########################
        
        from scipy.signal import savgol_filter
    
        # Define the Savitzky-Golay filter parameters
        window_length = 15  # Must be an odd number
        polyorder = 2  # Polynomial order to fit
        
        # Apply the Savitzky-Golay filter to smooth the signal
        segment_data['Magnitude_savgol_golay_smoothed'] = savgol_filter(segment_data['Magnitude_smoothed'], window_length, polyorder)
        segment_data=segment_data.dropna().reset_index(drop=True)
        
        """
        # Plot the accelerometer signal
        plt.figure(figsize=(10, 6))
        plt.plot(segment_data['timestamp'], segment_data['Magnitude_savgol_golay_smoothed'], label='Smoothed ACC Magnitude')
        plt.xlabel('Time')
        plt.ylabel('Accelerometer')
        plt.title(f'Accelerometer Signal After Savgol-Golay Smoothing Magnitude {segment_id}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout() 
        plt.show()
        """
        
        
        ###################### GAUSSIAN FILTERING ###############################
        
        from scipy.ndimage import gaussian_filter1d

        # Apply Gaussian smoothing with a specified sigma (smoothness factor)
        segment_data['Magnitude_gaussian_smooth'] = gaussian_filter1d(segment_data['Magnitude_savgol_golay_smoothed'], sigma=5)
        """
        # Plot the accelerometer signal
        plt.figure(figsize=(10, 6))
        plt.plot(segment_data['timestamp'], segment_data['Magnitude_gaussian_smooth'], label='Smoothed ACC Magnitude')
        plt.xlabel('Time')
        plt.ylabel('Accelerometer')
        plt.title(f'Accelerometer Signal After Gaussian Smoothing Magnitude {segment_id}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout() 
        plt.show()
        """
        
        #################### MAGNITUDE PEAK DETECTION ############################
        # Detect peaks
        delta = 1  # Adjusted from 5 to capture more peaks
        height = 1  # Lowered from 5 to 1
        distance = 1
        # Detect peaks on original and smoothed signals
        original_peaks, _ = find_peaks(segment_data['Magnitude_normalized'], prominence=delta, distance=distance, height=height)
        smoothed_peaks, _ = find_peaks(segment_data['Magnitude_gaussian_smooth'], prominence=delta, distance=distance, height=height)

       
        
        ###################### PLOT ORIGINAL, SMOOTHED SIGNAL, PEAKS, AND OUTLIERS ########################################
        """
        # Plot side-by-side original vs smoothed signal with detected peaks
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        
        # Plot 1: Original magnitude signal with detected peaks
        axes[0].plot(segment_data['timestamp'], segment_data['Magnitude_normalized'], label='Original Magnitude', color='blue', linewidth=1)
        axes[0].scatter(segment_data['timestamp'].iloc[original_peaks], 
                        segment_data['Magnitude_normalized'].iloc[original_peaks], 
                        label='Peaks on Original', color='orange', marker='o', s=50)
        axes[0].set_title('Original Magnitude with Detected Peaks')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Magnitude')
        axes[0].legend()
        axes[0].grid(True)
        # axes[0].set_xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))  # Removed to plot the entire signal
        
        # Plot 2: Smoothed signal (Savitzky-Golay) with detected peaks
        axes[1].plot(segment_data['timestamp'], segment_data['Magnitude_gaussian_smooth'], label='Smoothed Magnitude (Savitzky-Golay)', color='green', linewidth=1.5)
        axes[1].scatter(segment_data['timestamp'].iloc[smoothed_peaks], 
                        segment_data['Magnitude_gaussian_smooth'].iloc[smoothed_peaks], 
                        color='purple', s=50, label='Peaks on Smoothed', marker='x')
        axes[1].set_title('Smoothed Signal with Detected Peaks')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Magnitude')
        axes[1].legend()
        axes[1].grid(True)
        # axes[1].set_xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))  # Removed to plot the entire signal
        
        # Show the plots side by side
        plt.tight_layout()
        plt.show()

        """   
        ########################### PREPARE FOR STATISTICAL ANALYSIS ######################
        segment_data=segment_data.set_index('timestamp')
        analysis_columns = ['x', 'y', 'z', 'Magnitude',
               'Magnitude_normalized', 'Magnitude_filtered', 'Magnitude_lowpass', 'Magnitude_Median_Filtered', 
               'Magnitude_smoothed', 'Magnitude_savgol_golay_smoothed', 'Magnitude_gaussian_smooth']
        
        acc_analysis_df = segment_data[analysis_columns]
    
        ########################### ANALYZE ##############################################
        window_length_in_seconds = 30  # seconds
        window_length_in_samples = sampling_rate * window_length_in_seconds
        overlap = 0.9  # 90% overlap
        shift = int(window_length_in_samples * (1 - overlap))  # Number of samples to shift for each window
        num_windows = int(np.floor((len(acc_analysis_df) - window_length_in_samples) / shift) + 1)
        skipped=0
        features_df=[]
        
        for i in range(num_windows):
            # Extract the window
            start_idx = i * shift  # This is where the overlap is taken into account
            end_idx = start_idx + window_length_in_samples
            signal_window = acc_analysis_df[start_idx:end_idx]
            signal_timestamp = acc_analysis_df.index[start_idx:end_idx]
            
            
            if len(signal_window) < window_length_in_samples:
                print('Segment: ', segment_id, 'Window: ', i ,' skipped')
                continue
            
            # Check if there are any NaN values
            if signal_window.isnull().values.any():
                # Interpolate NaN values
                signal_window = signal_window.interpolate(method='linear', limit_direction='forward', axis=0)
                print("NaN values interpolated.")
               
            # Convert the signals from time domain to frequency domain using FFT
            acc_fft = np.abs(np.fft.fft(signal_window['Magnitude_gaussian_smooth']))[1:51]
            acc_fft = acc_fft / len(signal_window['Magnitude_gaussian_smooth'])  # Normalization
            
            # For each window, you would call:
            time_features = compute_statistical_features(signal_window['Magnitude_gaussian_smooth'])
            freq_features = compute_frequency_features(acc_fft, sampling_rate)
            
            # Merge features into one DataFrame
            combined_features = {**time_features, **freq_features}
            combined_features['timestamp'] = signal_timestamp[0]

            
            features_df+=[combined_features]
        
      
        if features_df != []:        
            acc_metrics_df = pd.DataFrame(features_df)
            acc_metrics_df=acc_metrics_df.reset_index()
            acc_metrics_df = acc_metrics_df.sort_values('timestamp')
        
            # Convert 'timestamp' to datetime if not already
            segment_data=segment_data.reset_index()
            segment_data['timestamp'] = pd.to_datetime(segment_data['timestamp'])
            acc_metrics_df['timestamp'] = pd.to_datetime(acc_metrics_df['timestamp'])
            
            # Perform the merge operation on the 'timestamp' column
            acc_metrics_df = pd.merge(acc_metrics_df, segment_data[['timestamp', 'x', 'y', 'z', 'Magnitude_savgol_golay_smoothed']],
                         on='timestamp', how='left')  # Use 'left' join to keep all rows from acc_metrics_df
    
            segment_features_list[segment_id]= acc_metrics_df
        
    # Combine all DataFrames into one DataFrame
    extracted_features_df = pd.concat(segment_features_list.values(), ignore_index=True)
    return extracted_features_df
    
   
