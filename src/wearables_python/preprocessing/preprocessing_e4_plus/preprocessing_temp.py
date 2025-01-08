# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:44:02 2025

@author: selaca
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:58:55 2024

@author: selaca
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:26:27 2024

@author: selaca
"""


import pandas as pd
import sys
import numpy as np
import os
from zipfile import ZipFile
from datetime import timedelta
from collections import Counter

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import stats

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

def cosinor_model(t, mesor, amplitude, acrophase):
    """
    Define the Cosinor model function for circadian rhythm.
    
    Args:
    t (float): Time in hours since midnight.
    mesor (float): The baseline (mean) value.
    amplitude (float): The amplitude of oscillation.
    acrophase (float): The phase shift (peak timing).
    
    Returns:
    float: The predicted value for the time t.
    """
    return mesor + amplitude * np.cos((2 * np.pi * t / 24) + acrophase)

def fit_cosinor_model(data, time_col, temp_col):
    """
    Fit a Cosinor model to temperature data.
    
    Args:
    data (pd.DataFrame): DataFrame containing temperature data.
    time_col (str): Name of the column containing time in hours.
    temp_col (str): Name of the column containing temperature values.
    
    Returns:
    dict: A dictionary with fitted parameters for mesor, amplitude, acrophase, and R-squared.
    """
    # Define initial parameter guesses: [mesor, amplitude, acrophase]
    initial_guess = [data[temp_col].mean(), (data[temp_col].max() - data[temp_col].min()) / 2, 0]

    # Fit the Cosinor model to the data
    params, _ = curve_fit(cosinor_model, data[time_col], data[temp_col], p0=initial_guess)

    mesor, amplitude, acrophase = params
    fitted_values = cosinor_model(data[time_col], mesor, amplitude, acrophase)
    """
    # Plot the original data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(data[time_col], data[temp_col], 'bo', label='Original Data')
    plt.plot(data[time_col], fitted_values, 'r-', label='Fitted Cosinor Model')
    plt.xlabel('Time of Day (hours)')
    plt.ylabel('Temperature (°C)')
    plt.title('Cosinor Analysis of Circadian Rhythm')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, 24])  # Set x-axis to 24 hours for clarity
    plt.show()
    """
    # Calculate the R-squared value for goodness of fit
    residuals = data[temp_col] - fitted_values
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data[temp_col] - np.mean(data[temp_col]))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        'mesor': mesor,
        'amplitude': amplitude,
        'acrophase': acrophase,
        'r_squared': r_squared
    }



def low_pass_filter(data, cutoff_freq, sampling_rate):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data



def get_sampling_rate(df, timestamp_col='timestamp'):
    """
    Calculate the sampling rate of a time series based on the time differences between consecutive timestamps.
    
    Args:
    df (pd.DataFrame): DataFrame containing the time series data.
    timestamp_col (str): The name of the column containing the timestamp (default is 'timestamp').
    
    Returns:
    float: The average sampling rate in Hz (samples per second).
    """
    # Ensure the timestamp column is in datetime format
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Calculate time differences in seconds between consecutive rows
    time_diffs = df[timestamp_col].diff().dt.total_seconds()
    
    # Drop NaN values that result from the diff() calculation (first row will be NaN)
    time_diffs = time_diffs.dropna()
    
    # Calculate the average time difference (interval)
    avg_time_diff = time_diffs.mean()  # average time difference in seconds
    
    # Calculate the sampling rate (samples per second)
    if avg_time_diff > 0:
        sampling_rate = 1 / avg_time_diff  # Sampling rate in Hz
    else:
        sampling_rate = None  # No valid sampling rate if avg_time_diff is zero or negative
    
    return sampling_rate


def analyze_and_store_temp_changes_with_timestamp(df, sf, std_factor=3):
    """
    Analyzes temperature changes and stores significant changes based on standard deviation criteria,
    using the 'timestamp' column for time information.

    Parameters:
    - df: DataFrame containing the temperature data and a 'timestamp' column.
    - sf: Sampling frequency of the data.
    - std_factor: Factor of standard deviation to consider a change significant.

    Returns:
    - DataFrame with an added 'significant_change' column indicating significant changes in temperature,
      marked as True or False.
    """
    # Convert 'timestamp' column to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate the standard deviation and mean of the temperature values
    temp_std = df['TEMP_value'].std()
    temp_mean = df['TEMP_value'].mean()

    # Define thresholds for significant changes
    upper_threshold = temp_mean + (std_factor * temp_std)
    lower_threshold = temp_mean - (std_factor * temp_std)

    # Initialize a column for significant temperature changes with default False
    df['significant_change'] = False

    # Identify significant changes in temperature
    significant_changes = (df['TEMP_value'] > upper_threshold) | (df['TEMP_value'] < lower_threshold)
    
    # Update the 'significant_change' column for rows with significant changes
    df.loc[significant_changes, 'significant_change'] = True

    return df



def mark_peaks(df,column_name):
    """
    Identifies peaks in the 'column_name' column of a DataFrame and marks them in a new column 'is_peak'.
    
    Parameters:
    - df: DataFrame containing the temperature data in a column named 'column_name'.
    
    Returns:
    - DataFrame with an added boolean column 'is_peak' indicating peak rows.
    """
    # Ensure 'TEMP_value' is numeric
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    # Find peaks
    peaks_indices, _ = find_peaks(df[column_name])
    
    # Initialize the 'is_peak' column to False
    df['is_peak'] = False
    
    # Mark the peaks in 'is_peak' column
    df.loc[peaks_indices, 'is_peak'] = True
    
    return df

def resample_dataframe(data, target_samp_freq='250L'):  # Adjust to match original sampling rate (e.g., 4 Hz)
    # 1. Ensure the timestamp column is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # 2. Set the timestamp column as the DataFrame index
    data.set_index('timestamp', inplace=True)
    
    # 3. Resample the data (use .asfreq() instead of .mean() to avoid averaging)
    resampled_data = data.resample(target_samp_freq).asfreq()
    
    # 4. Interpolate missing values with linear interpolation (simpler than spline)
    resampled_data = resampled_data.interpolate(method='linear')
    
    # Reset the index if needed (optional)
    resampled_data.reset_index(inplace=True)
    
    return resampled_data




def plot_temperature_signal(data, segment_id,  start_time, end_time, timestamp_col='timestamp', temp_col='TEMP_value'):
    """
    Plot the temperature signal over a specified period.
    
    Args:
    data (pd.DataFrame): The DataFrame containing the temperature data.
    start_time (str or pd.Timestamp): The start time of the period for plotting.
    end_time (str or pd.Timestamp): The end time of the period for plotting.
    timestamp_col (str): The name of the timestamp column in the DataFrame.
    temp_col (str): The name of the temperature column in the DataFrame.
    """
    # Ensure the timestamp column is in datetime format
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    
    # Filter the data for the specified time range
    mask = (data[timestamp_col] >= start_time) & (data[timestamp_col] <= end_time)
    filtered_data = data[mask]
    
    # Plot the temperature signal
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data[timestamp_col], filtered_data[temp_col], label='Temperature Signal')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Signal from {start_time} to {end_time} for Segment {segment_id}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent label overlap
    plt.show()


def filter_outliers_iqr(data, temp_col, multiplier=1.5):
    """
    Filter out temperature values based on IQR outlier detection.
    
    Args:
    data (pd.DataFrame): The DataFrame containing the temperature data.
    temp_col (str): The name of the temperature column in the DataFrame.
    multiplier (float): The IQR multiplier to define the range of outliers (default is 1.5).
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = data[temp_col].quantile(0.25)
    Q3 = data[temp_col].quantile(0.75)
    
    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Define the lower and upper bounds for detecting outliers
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Filter out data points outside the lower and upper bounds
    filtered_data = data[(data[temp_col] >= lower_bound) & (data[temp_col] <= upper_bound)].copy()
    
    return filtered_data


def filter_temperature_by_threshold(data, temp_col, min_temp=28, max_temp=38):
    """
    Filter out temperature values outside a realistic range.
    
    Args:
    data (pd.DataFrame): The DataFrame containing the temperature data.
    temp_col (str): The name of the temperature column in the DataFrame.
    min_temp (float): The minimum valid temperature.
    max_temp (float): The maximum valid temperature.
    
    Returns:
    pd.DataFrame: DataFrame with values outside the range removed.
    """
    # Filter out unrealistic temperature values
    filtered_data = data[(data[temp_col] >= min_temp) & (data[temp_col] <= max_temp)]
    
    return filtered_data


def smooth_temperature_moving_average(data, temp_col, window_size=5):
    """
    Apply a moving average filter to smooth the temperature data.
    
    Args:
    data (pd.DataFrame): The DataFrame containing the temperature data.
    temp_col (str): The name of the temperature column in the DataFrame.
    window_size (int): The window size for the moving average filter.
    
    Returns:
    pd.DataFrame: DataFrame with smoothed temperature values.
    """
    # Apply moving average filter
    data['MovingAverage_smoothed'] = data[temp_col].rolling(window=window_size, min_periods=1).mean()
    
    return data

def separate_into_24hr_cycles(data, timestamp_col='timestamp'):
    """
    Separates the combined dataframe into 24-hour cycles.
    
    Args:
    data (pd.DataFrame): DataFrame containing the time-series data with a timestamp.
    timestamp_col (str): The name of the timestamp column.
    
    Returns:
    list: A list of DataFrames, each representing a 24-hour cycle.
    """
    # 1. Ensure the timestamp column is in datetime format
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    
    # 2. Extract the date part from the timestamp (this will act as a "24-hour cycle" identifier)
    data['date'] = data[timestamp_col].dt.date
    
    # 3. Group by the date to create separate DataFrames for each 24-hour period
    grouped_data = [group for _, group in data.groupby('date')]
    
    # 4. Optionally, drop the 'date' column from the separated data (if not needed anymore)
    for df in grouped_data:
        df.drop(columns=['date'], inplace=True)
    
    return grouped_data

def check_24hr_cycle(day_data, time_col='time_of_day'):
    """
    Check if the day_data contains a full 24-hour cycle or a partial cycle.
    
    Args:
    day_data (pd.DataFrame): DataFrame containing time series data with a time_of_day column.
    time_col (str): Name of the column containing time of day in hours.
    
    Returns:
    bool: True if it's a full 24-hour cycle, False if it's a partial cycle.
    """
    # Check the range of the time_of_day column
    min_time = day_data[time_col].min()
    max_time = day_data[time_col].max()

    # Check if the range covers the full 24-hour period
    if min_time <= 2 and max_time >= 22:
        return True  # Full 24-hour cycle
    else:
        return False  # Partial 
    
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

    features['above_mean_percentage'] = np.sum(data_list > np.mean(data_list)) / len(data_list) * 100

    
   
    features['skewness'] = stats.skew(data_list)
    features['kurtosis'] = stats.kurtosis(data_list)
   

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
    
  
    # Frequencies corresponding to FFT values
    freqs = np.fft.fftfreq(len(data_list), d=1/sampling_rate)[:len(data_list)]
    
    # Spectral centroid: the weighted mean of the frequency components
    features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
    
    # Spectral bandwidth: the spread of the power spectrum around the centroid
    features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * psd) / np.sum(psd))
   
    return features   

def process(temp_df):
    temp_df.reset_index(drop=True,inplace=True)
    ############################ FIND CONTIONUS PERIODS WHERE THE DEVICE IS WORN FOR EMPATICA ###############################
    # Assuming 'timestamp' is in datetime format in 'merged_df'
    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
    
    # Calculate time differences between consecutive rows
    temp_df['time_diff'] = temp_df['timestamp'].diff().dt.total_seconds()
    temp_df = temp_df.dropna()
    # Define a threshold for identifying gaps (e.g., 60 seconds of no data indicates discontinuity)
    gap_threshold = 300  # 300 seconds
    
    # Create a new column 'segment_id' to mark continuous segments
    # We increment the segment ID whenever there's a gap larger than the threshold
    temp_df['segment_id'] = (temp_df['time_diff'] > gap_threshold).cumsum()
    
    # Optional: Drop the 'time_diff' column if no longer needed
    temp_df = temp_df.drop(columns=['time_diff'])
    
    ############################ PROCESS EACH SEGMENT SEPARATELY ##############################################   
    segment_features_list = {}
    
    #print('Number of segments in day:', i, 'is :', Counter(day_data['segment_id']))
    # Group by 'segment_id' and process each segment separately
    for segment_id, data in temp_df.groupby('segment_id'):
        
        segment_data = temp_df[temp_df['segment_id']==segment_id]
        #################### FILTER THE SIGNAL VIA LOW PASS FILTER ###########################
        # Apply the low-pass filter with a cutoff frequency of 0.01 Hz
        cutoff_frequency = 0.01  # Adjust to 0.05 Hz if you want to capture faster changes
        sampling_rate = 4  # Hz
        filtered_temperature = low_pass_filter(segment_data['temp'], cutoff_frequency, sampling_rate)
        segment_data['LowPass_filtered']=filtered_temperature
      
        """
        # Plot the temperature signal
        plt.figure(figsize=(10, 6))
        plt.plot(segment_data['timestamp'], segment_data['LowPass_filtered'], label='Low Pass Filtered Temperature Signal')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.title(f'Temperature Signal After Low Pass Filtering')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()  # Adjust layout to prevent label overlap
        plt.show()
        """
   
        ####################### FILTER SIGNAL VIA IQR ###################
        
        # Apply IQR-based filtering to remove outliers
        segment_data = filter_outliers_iqr(segment_data, temp_col='LowPass_filtered')
       
        """
        # Plot the temperature signal
        plt.figure(figsize=(10, 6))
        plt.plot(segment_data['timestamp'], segment_data['LowPass_filtered'], label='IQR Filtered Temperature Signal')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.title(f'Temperature Signal After Z-Score Filtering')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()  # Adjust layout to prevent label overlap
        plt.show()
        """
        
        if len(segment_data) < 2400:
            #print('B. Day ID:',i,'segment_id:',segment_id, 'skipped out of ', len(Counter(segment_data['segment_id'])) ,'segments because length!')
            continue
        ############### THERESHOLD BASED FILTERING ##################################
        # Example usage:
        segment_data = filter_temperature_by_threshold(segment_data, temp_col='LowPass_filtered', min_temp=28, max_temp=38)
       
        """
        # Plot the temperature signal
        plt.figure(figsize=(10, 6))
        plt.plot(segment_data['timestamp'], segment_data['LowPass_filtered'], label='Thereshold Filtered Temperature Signal')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.title(f'Temperature Signal AfterThereshold Filtering')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()  # Adjust layout to prevent label overlap
        plt.show()
        """
        
        if len(segment_data) < 2400:
            #print('C. Day ID:',i,'segment_id:',segment_id, 'skipped out of ', len(Counter(segment_data['segment_id'])) ,'segments because length!')
            continue
        ################## RESAMPLE TO MAKE SURE IT IS 4 Hz ################
        segment_data['timestamp'] = segment_data['timestamp'].dt.round('250L')
        segment_data=resample_dataframe(segment_data)
        #plot_temperature_signal(segment_data, segment_id, start_time=segment_data['timestamp'][0], end_time=segment_data['timestamp'][len(segment_data['timestamp'])-1])
     
        ################ APPLY MOVING AVERAGE TO SMOOTH OUT THE SIGNAL #############
        segment_data = smooth_temperature_moving_average(segment_data, temp_col='LowPass_filtered', window_size=5)
        """
        # Plot the temperature signal
        plt.figure(figsize=(10, 6))
        plt.plot(segment_data['timestamp'], segment_data['MovingAverage_smoothed'], label='Smothed Temperature Signal')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.title(f'Temperature Signal After Moving Average Smoothing {segment_id}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout() 
        plt.show()
        """
        if segment_data.empty:
            #print('D. Day ID:',i,'segment_id:',segment_id, 'skipped out of ', len(Counter(segment_data['segment_id'])) ,'segments because length!')
            continue
        
        ######################## EXTRACT STATISTICAL AND FREQUENCY FEATURES FROM TEMP_RESULTS_DAILY #########################
       
        window_length_in_seconds = 120  # seconds
        window_length_in_samples = sampling_rate * window_length_in_seconds
        overlap = 0.95  # 90% overlap
        shift = int(window_length_in_samples * (1 - overlap))  # Number of samples to shift for each window
        num_windows = int(np.floor((len(segment_data) - window_length_in_samples) / shift) + 1)
        skipped=0
       
        features=[]
        for i in range(num_windows):
            # Extract the window
            start_idx = i * shift  # This is where the overlap is taken into account
            end_idx = start_idx + window_length_in_samples
            signal_window = segment_data[start_idx:end_idx]
            signal_timestamp = segment_data['timestamp'][start_idx:end_idx]
          
            if len(signal_window) < window_length_in_samples:
                #print('Segment: ', segment_id, 'Window: ', i ,' skipped')
                continue
            
            # Check if there are any NaN values
            if signal_window.isnull().values.any():
                # Interpolate NaN values
                signal_window = signal_window.interpolate(method='linear', limit_direction='forward', axis=0)
                #print("NaN values interpolated.")
            
            # Convert the signals from time domain to frequency domain using FFT
            temp_fft = np.abs(np.fft.fft(signal_window['MovingAverage_smoothed']))[1:51]
            temp_fft = temp_fft / len(signal_window['MovingAverage_smoothed'])  # Normalization
            
            # For each window, you would call:
            time_features = compute_statistical_features(signal_window['MovingAverage_smoothed'])
            freq_features = compute_frequency_features(temp_fft, sampling_rate)
            
            # Merge features into one DataFrame
            combined_features = {**time_features, **freq_features}
            combined_features['timestamp'] = signal_timestamp.iloc[0]
            
            features+=[combined_features]
        features_df = pd.DataFrame(features)
        segment_features_list[segment_id]=features_df
    
    return segment_features_list
    
    """
     ############### CHECK IF FEATURES ARE WITHIN EXPECTED QUALITY RANGES ##############
     
     # Define the adjusted expected ranges for each feature
     expected_ranges = {
         'mean': (15, 35),                   # Mean temperature (°C)
         'std': (0, 5),                      # Standard deviation (°C)
         'aad': (0, 5),                      # Average absolute deviation (°C)
         'min': (0, 35),                     # Minimum temperature (°C)
         'max': (15, 40),                    # Maximum temperature (°C)
         'maxmin_diff': (0, 5),              # Lowered range for daily max-min difference, considering minimal variation
         'median': (15, 35),                 # Median temperature (°C)
         'mad': (0, 5),                      # Median absolute deviation (°C)
         'IQR': (0, 10),                     # Interquartile range (°C)
         'above_mean_percentage': (10, 90),  # Broader range for percentage of values above mean
         'skewness': (-1.5, 1.5),            # Broadened range for skewness
         'kurtosis': (-1, 3),                # Broadened range for kurtosis
         'dominant_freq': (0, 0.5),          # Dominant frequency (Hz)
         'spectral_entropy': (0, 1.2),       # Slightly increased upper bound for spectral entropy
         'spectral_centroid': (0, 2),        # Adjusted lower bound to allow lower centroid values
         'spectral_bandwidth': (0, 3)        # Spectral bandwidth (Hz)
     }
     
     # Function to check if each feature is within the expected range
     def check_temperature_ranges(features, expected_ranges):
         results = {}
         for feature, value in features.items():
             min_expected, max_expected = expected_ranges.get(feature, (None, None))
             if min_expected is not None and max_expected is not None:
                 if min_expected <= value <= max_expected:
                     results[feature] = "Pass"
                 else:
                     results[feature] = f"Fail - {value} (expected {min_expected} to {max_expected})"
             else:
                 results[feature] = "No expected range defined"
         return results
     
     # Run the check for each daily DataFrame in features_daily
     for i, daily_features_df in enumerate(features_daily):
         # Convert the daily DataFrame to a dictionary for feature checking
         daily_features = daily_features_df.iloc[0].to_dict()  # Assuming each DataFrame has one row of daily features
         
         # Perform the range check
         check_results = check_temperature_ranges(daily_features, expected_ranges)
         
         # Print results for each day's features
         print(f"Results for Day {i + 1}:")
         for feature, result in check_results.items():
             print(f"  {feature}: {result}")
         print("\n" + "-"*40 + "\n")
     
    
     
    """
             
             
         
         
