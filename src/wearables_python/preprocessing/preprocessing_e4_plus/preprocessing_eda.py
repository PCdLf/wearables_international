# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:23:46 2024

@author: selaca
"""

import os
import sys
import pandas as pd
import numpy as np

from zipfile import ZipFile
from datetime import datetime,timezone,timedelta
import time
from scipy.signal import welch
from scipy.stats import entropy


from scipy.signal import resample
import heartpy as hp

from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import medfilt
import neurokit2 as nk
import matplotlib.pyplot as plt
import heartpy as hp
from collections import Counter

from scipy.signal import convolve
import scipy.signal as signal
import matplotlib.pyplot as plt
import pywt



from scipy.stats import skew, kurtosis  # For calculating skewness and kurtosis
#from scipy.integrate import simps     # For calculating area under the curve (AUC)
#from scipy.integrate import trapz

from scipy.fft import fft 


import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def resample_dataframe(data, target_samp_freq='250L'):
    # Ensure the timestamp column is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Set the timestamp column as the DataFrame index
    data.set_index('timestamp', inplace=True)
    
    # Resample and interpolate missing values
    resampled_data = data.resample(target_samp_freq).interpolate(method='linear')
    
    # Reset the index (optional)
    resampled_data.reset_index(inplace=True)
    
    return resampled_data




def process(eda_df):        
    eda_df['timestamp'] = pd.to_datetime(eda_df['timestamp'])
    eda_df['timestamp'] = eda_df['timestamp'].dt.tz_localize(None)
    eda_df = eda_df.sort_values('timestamp')
    eda_df=eda_df.dropna()
    """
    plt.figure(figsize=(10, 4))
    plt.plot(eda_df['eda'])
    plt.title('EDA Data Segment')
    plt.xlabel('Sample number')
    plt.ylabel('EDA signal')
    plt.show()
    """
    #Find interrupted segments:
    # Calculate differences between consecutive timestamps
    eda_df['time_diff'] = eda_df['timestamp'].diff()
    eda_df['new_segment'] = eda_df['time_diff'] > pd.Timedelta(minutes=1)
    eda_df['segment_id'] = eda_df['new_segment'].cumsum()
    

    # Initialize variables
    eda_metrics_full = []
    sampling_rate = 4  # 4 Hz
    window_length_in_seconds = 300  # 10 minutes
    window_length_in_samples = sampling_rate * window_length_in_seconds  # no. samples for 10 minutes at 4 Hz
    overlap = 0.9  # 80% overlap
    shift = int(window_length_in_samples * (1 - overlap))  # Number of samples to shift for each window
    
    segment_features=[]
    for segment_id, segment_data in eda_df.groupby('segment_id'):
        """
        # Plot the signal
        plt.figure(figsize=(12, 6))
        plt.plot(segment_data['eda'], label="Original EDA Signal", color="orange")
        plt.title("Original EDA Signal")
        plt.xlabel("Time")
        plt.ylabel("EDA (µS)")
        plt.legend()
        plt.show()
        """
        ################## Ensure Resampling is at 4 Hz ################
        # Resample to 4 Hz if needed (using  ms approximation)
        segment_data['timestamp'] = segment_data['timestamp'].dt.round('250L')
        eda_resampled = resample_dataframe(segment_data[['timestamp', 'eda']], target_samp_freq='250L')  # Close to 4 Hz
       
        """
        # Plot the original and resampled signals in subplots
        plt.figure(figsize=(12, 10))
        
        # Plot original EDA signal
        plt.subplot(2, 1, 1)
        plt.plot(segment_data['timestamp'], segment_data['eda'], label="Original EDA Signal", alpha=0.7)
        plt.title("Original EDA Signal")
        plt.xlabel("Time")
        plt.ylabel("EDA")
        
        # Plot resampled EDA signal
        plt.subplot(2, 1, 2)
        plt.plot(eda_resampled['timestamp'], eda_resampled['eda'], label="Resampled EDA Signal (Approx 64 Hz)", color="orange", alpha=0.7)
        plt.title("Resampled EDA Signal (Approx 64 Hz)")
        plt.xlabel("Time")
        plt.ylabel("EDA")
        
        # Display the plot
        plt.tight_layout()
        plt.show()
        """
        #print('Data resampled to 4 Hz')
        
        ################# CHECK SEGMENT LENGTH ###########################
        segment_length_seconds = (eda_resampled['timestamp'].iloc[-1] - eda_resampled['timestamp'].iloc[0]).total_seconds()
        if segment_length_seconds < window_length_in_seconds:
            #print(f"Skipping segment {segment_id} as it is shorter than {window_length_in_seconds} seconds.")
            continue
        else:
            #print(f"Processing segment {segment_id}. Segment length: {segment_length_seconds} seconds.")
            """
            # Plot the entire segment to see the overall signal
            plt.figure(figsize=(12, 6))
            plt.plot(eda_resampled['timestamp'], eda_resampled['eda'], label=f"Segment {segment_id}")
            plt.title(f"EDA Signal - Segment {segment_id}")
            plt.xlabel("Time")
            plt.ylabel("EDA Value")
            plt.legend()
            plt.show()
            """
            ###############################################
            
            windows_skipped=0
            # Extract and plot 10-minute windows within this segment
            num_windows = int(np.floor((len(eda_resampled) - window_length_in_samples) / shift) + 1)
            window_features=[]
            for i in range(num_windows):
                
                # Calculate the start and end indices for the window
                start_idx = i * shift
                end_idx = start_idx + window_length_in_samples
                
                #print('start_idx: ',start_idx, 'end_idx: ', end_idx)
                # Ensure that we do not exceed the segment length
                if end_idx > len(eda_resampled) :
                    break
                
                # Extract the window data
                window_data = eda_resampled.iloc[start_idx:end_idx]
                ##################### CHECK SENSOR QUALITY OF THE EDA ##########
                count_above_zero = (window_data['eda'] > 0).sum().sum()
                
                if (count_above_zero/len(window_data)*100) <=60:
                    #print('EDA Sensor quality extremely low!')
                    continue
                
                """
                # Plot the 10-minute window
                plt.figure(figsize=(12, 4))
                plt.plot(window_data['timestamp'], window_data['eda'], label=f"Segment {segment_id} - Window {i+1}")
                plt.title(f"EDA Signal - Segment {segment_id} - 10-Minute Window {i+1}")
                plt.xlabel("Time")
                plt.ylabel("EDA Value")
                plt.legend()
                plt.show()
                """
                
                ################### LOW PASS FILTER ########################
                # Low-pass filter settings
                cutoff_frequency = 1  # typically set around 5 Hz for EDA
                nyquist_frequency = 0.5 * sampling_rate
                normalized_cutoff = cutoff_frequency / nyquist_frequency
                
                # Design and apply the filter
                b, a = signal.butter(4, normalized_cutoff, btype="low", analog=False)
                eda_filtered = signal.filtfilt(b, a, window_data['eda'])
                """
                plt.figure(figsize=(10, 4))
                plt.plot(window_data['timestamp'],eda_filtered)
                plt.title("Low-Pass Filtered EDA Signal")
                plt.show()
                """
                #print('Low pass filter applied')
                
                ################ MEDIAN FILTERING ##########################

                # Apply a median filter (e.g., kernel size of 5)
                eda_median_filtered = medfilt(eda_filtered, kernel_size=5)
                """
                # Plot the filtered signal
                plt.figure(figsize=(12, 4))
                plt.plot(window_data['timestamp'],eda_median_filtered)
                plt.title("Median + Low-Pass Filtered EDA Signal")
                plt.show()
                """
                #print('Median filter applied')
                
                ################## NK CLEAN ################################
                # Clean the filtered EDA signal
                eda_cleaned = nk.eda_clean(eda_median_filtered, sampling_rate=sampling_rate)
                """
                plt.figure(figsize=(12, 4))
                plt.plot(window_data['timestamp'],eda_cleaned)
                plt.title("Cleaned EDA Signal")
                plt.show()
                """
                #print('Neurokit preprocessing applied')
                
                ############### STATISTICAL FILTERING ########################
                
                # Calculate the mean and standard deviation to define an outlier threshold
                threshold = eda_cleaned.mean() + 4 * eda_cleaned.std()
                
                # Convert to a pandas Series to handle NaNs and interpolation
                eda_series = pd.Series(eda_cleaned)
                
                # Identify points above this threshold
                outliers = eda_series > threshold
                
                # Replace outliers with NaN and interpolate
                eda_series[outliers] = np.nan
                eda_stats_filtered = eda_series.interpolate()
                # Fill any remaining NaNs at the edges using forward and backward fill
                eda_stats_filtered = eda_stats_filtered.fillna(method='bfill').fillna(method='ffill')
                
                
                # Convert back to a NumPy array if needed
                eda_stats_filtered = eda_stats_filtered.to_numpy()
                
                # Check if any NaNs are in the smoothed output
                if np.isnan(eda_stats_filtered).any():
                    #print('EDA signal too noisy')
                    continue
               
                
                """
                # Plot the filtered signal
                plt.figure(figsize=(12, 4))
                plt.plot(window_data['timestamp'],eda_stats_filtered)
                plt.title("Outlier Removal + Low-Pass Filtered EDA Signal")
                plt.xlabel("Time")
                plt.ylabel("EDA Value")
                plt.show()
                """
                
                ############# SAVGOL FILTER ##################################
                

                # Apply Savitzky-Golay filter
                eda_smoothed = savgol_filter(eda_stats_filtered, window_length=15, polyorder=2)  # Adjust window_length and polyorder as needed
                
                # Explicitly convert to a NumPy array, if not already one
                eda_smoothed = np.asarray(eda_smoothed)
                
                # Alternative check for NaNs or infinities in the smoothed output
                if np.isnan(eda_smoothed).any():
                    #print('NaN detected reverting back!')
                    eda_smoothed = eda_stats_filtered 
                
                """
                # Plot the smoothed signal
                plt.figure(figsize=(12, 4))
                plt.plot(window_data['timestamp'], eda_smoothed, label="Savitzky-Golay Smoothed EDA Signal")
                plt.title("Savitzky-Golay Filtered EDA Signal")
                plt.xlabel("Time")
                plt.ylabel("EDA Value")
                plt.legend()
                plt.show()
                """
                #################### AMPLITUDE RANGE CHECK ####################
            
                # Define the EDA range thresholds
                min_eda, max_eda = 0.01, 10.0  # MicroSiemens (µS)
                
                # Convert eda_stats_filtered to a pandas Series for easier manipulation
                eda_series = pd.Series(eda_smoothed)
                
                # Flag out-of-range values in the EDA series
                out_of_range = (eda_series < min_eda) | (eda_series > max_eda)
                
                # Save the values that will be marked as NaN
                out_of_range_values = eda_series[out_of_range]
                """
                print("Values marked as NaN (outliers):")
                print(out_of_range_values)
                """
                
                # Mark out-of-range values as NaN
                eda_series[out_of_range] = np.nan
                
                # Interpolate NaN values to fill in gaps caused by outliers
                eda_series = eda_series.interpolate()
                
                # Fill any remaining NaNs at the edges using forward and backward fill
                eda_series = eda_series.fillna(method='bfill').fillna(method='ffill')
                
                
                # Alternative check for NaNs or infinities in the smoothed output
                if np.isnan(eda_series).any():
                    #print('EDA window quality very low!')
                    continue
                
                # Calculate total counts for different ranges
                below_0_01 = (eda_series < 0.01).sum()
                between_0_01_and_10 = ((eda_series >= 0.01) & (eda_series <= 10)).sum()
                above_10 = (eda_series > 10).sum()
                
                """
                # Print out the information
                print("\nTotal number of values marked as NaN (outliers):", len(out_of_range_values))
                print("Total number of values below 0.01:", below_0_01)
                print("Total number of values between 0.01 and 10:", between_0_01_and_10)
                print("Total number of values above 10:", above_10)
                """
                # Convert back to a NumPy array if needed
                eda_amp_range_checked = eda_series.to_numpy()
                
                """
                # Plot the cleaned signal without outliers
                plt.figure(figsize=(12, 6))
                plt.plot(window_data['timestamp'],eda_amp_range_checked, label="EDA Signal with Out of Range Values Removed", color="orange")
                plt.title("EDA Signal After Removing Outliers")
                plt.xlabel("Time")
                plt.ylabel("EDA (µS)")
                plt.legend()
                plt.show()
                """
                ################## AMPLITUDE VARIATION CHECK ################
                # Assuming eda_amp_range_checked is a NumPy array, first convert it to a pandas Series
                eda_series = pd.Series(eda_amp_range_checked)
                
                # Calculate the amplitude variance over a 10-second window (assuming 64 Hz sampling rate)
                window_size = 64 * 10  # 10-second window at 64 Hz
                eda_variance = eda_series.rolling(window=window_size).var()
                
                # Flag windows with very low variance
                low_variance_threshold = 0.001  # Adjust based on data characteristics
                low_variance_segments = eda_variance < low_variance_threshold
                #print("Number of low variance segments:", low_variance_segments.sum())
                
                """
                # Plot the variance
                plt.figure(figsize=(12, 4))
                plt.plot(eda_variance, label="EDA Variance", color="green")
                plt.title("EDA Amplitude Variance")
                plt.xlabel("Time")
                plt.ylabel("Variance")
                plt.legend()
                plt.show()
                """
                
                # Define an amplitude threshold for near-zero values (e.g., below 0.01 µS)
                amplitude_threshold = 0.01
                
                # Create a mask for low amplitude in low variance segments
                low_amplitude_segments = (eda_series < amplitude_threshold) & low_variance_segments
                
                # Step 4: Handle Detected Artifacts
                # Option A: Mark detected artifacts as NaN
                eda_series[low_amplitude_segments] = np.nan
                
                # Option B: Interpolate over small gaps (if you prefer to keep continuity)
                eda_interpolated = eda_series.interpolate()
                
                # Fill any remaining NaNs at the edges using forward and backward fill
                eda_interpolated = eda_interpolated.fillna(method='bfill').fillna(method='ffill')
                
                # Alternative check for NaNs or infinities in the smoothed output
                if np.isnan(eda_interpolated).any():
                    #print('EDA window variation very low!')
                    continue
                
                
                """
                # Step 5: Plot to visualize the results
                plt.figure(figsize=(12, 6))
                plt.plot(window_data['timestamp'],eda_series, label="EDA with Low Variance Segments Marked as NaN", color="blue")
                plt.plot(window_data['timestamp'],eda_interpolated, label="Interpolated EDA Signal", color="orange")
                plt.title("EDA Signal with Low Variance and Low Amplitude Segments")
                plt.xlabel("Time")
                plt.ylabel("EDA (µS)")
                plt.legend()
                plt.show()
                """
                
                ##################### MOVING AVERAGE FILTERING ###################
                
                window_size = 5  # Adjust window size as needed
                eda_smooth = pd.Series(eda_interpolated).rolling(window=window_size, center=True).mean()
                
                """
                # Plot the smoothed signal
                plt.figure(figsize=(12, 4))
                plt.plot(window_data['timestamp'], eda_smooth, label="Smoothed EDA Signal")
                plt.title("Median + Low-Pass + Moving Average Filtered EDA Signal")
                plt.xlabel("Time")
                plt.ylabel("EDA (µS)")
                plt.legend()
                plt.show()
                """
                
                # Fill any remaining NaNs at the edges using forward and backward fill
                eda_smooth = eda_smooth.fillna(method='bfill').fillna(method='ffill')
                
                ######################### DECOMPOSE SIGNAL #########################
                
                # Decompose the EDA signal into tonic (SCL) and phasic (SCR) components
                eda_decomposed = nk.eda_phasic(eda_smooth, sampling_rate=sampling_rate)
                scl = eda_decomposed["EDA_Tonic"]  # Tonic component
                scr = eda_decomposed["EDA_Phasic"]  # Phasic component
                
                """
                # Plot the decomposed EDA signal (SCL and SCR)
                plt.figure(figsize=(12, 6))
                plt.plot(window_data['timestamp'], scl, label="SCL (Tonic Component)")
                plt.plot(window_data['timestamp'], scr, label="SCR (Phasic Component)")
                plt.title("Decomposed EDA Signal (SCL and SCR)")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.show()
                """
                
                """
                ######################### WAVELET DENOISING ##########################
                
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(scr, 'db4', level=3)
                
                # Apply soft thresholding to remove high-frequency noise in detail coefficients
                threshold = 0.06  # Set a threshold appropriate for noise level
                coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
                
                # Reconstruct the signal from the thresholded coefficients
                scr = pywt.waverec(coeffs, 'db4')
                
                # Plot the phasic component with detected peaks
                plt.figure(figsize=(12, 4))
                plt.plot(window_data['timestamp'], scr, label="Wavelet Denoised EDA Signal", color="blue")  # Use timestamps on x-axis
                plt.title("Wavelet Denoised EDA Signal")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.show()
                """
               
                
                ################### SIGNAL QUALITY CHECKS ########################################
                
                #################### CHECK BASELINE STABILITY OF THE SIGNAL ####################
                # Calculate variance of the tonic component (SCL)
                scl_variance = scl.var()
                scl_std = scl.std()
                
                # Threshold check
                baseline_stability_pass = scl_variance < 0.01  # Adjust threshold based on data
                #print("Baseline Stability Check:", "Pass" if baseline_stability_pass else "Fail")
                
                #################### DETECT PEAKS ON PHASIC COMPONENT (SCR) ####################
                
                # Detect peaks on the phasic component (SCR)
                scr_peaks, scr_peaks_info = nk.eda_peaks(scr, sampling_rate=sampling_rate, method='neurokit')
                
                # Fill NaNs in specific columns
                scr_peaks['SCR_Amplitude'].fillna(scr_peaks['SCR_Amplitude'].mean(), inplace=True)
                scr_peaks['SCR_RiseTime'].fillna(scr_peaks['SCR_RiseTime'].mean(), inplace=True)
                scr_peaks['SCR_RecoveryTime'].fillna(scr_peaks['SCR_RecoveryTime'].mean(), inplace=True)
                
                # Extract the indices where peaks were detected in the SCR component
                scr_peak_indices = scr_peaks[scr_peaks["SCR_Peaks"] == 1].index
                
                # Get the corresponding timestamps for the peaks
                peak_timestamps = window_data['timestamp'].iloc[scr_peak_indices]
                
                """
                # Plot the phasic component with detected peaks
                plt.figure(figsize=(12, 4))
                plt.plot(window_data['timestamp'], scr, label="SCR (Phasic Component)", color="blue")  # Use timestamps on x-axis
                plt.scatter(peak_timestamps, scr.iloc[scr_peak_indices], color="red", label="Detected Peaks", marker="o")
                plt.title("EDA Phasic Component with Detected Peaks Method:neurokit")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.show()
                """
                ######################## CHECK PEAK CONSISTENCY #########################
                
                # Calculate percentile-based thresholds
                amplitude_lower = scr_peaks['SCR_Amplitude'].quantile(0.1)
                amplitude_upper = scr_peaks['SCR_Amplitude'].quantile(0.9)
                
                # Check if amplitudes fall within this range
                amplitude_pass = scr_peaks['SCR_Amplitude'].between(amplitude_lower, amplitude_upper).mean() > 0.8
                #print(f"Adjusted Peak Consistency Check :", "Pass" if amplitude_pass else "Fail")
                
                
                ################### CHECK RISE AND RECOVERY TIMES ##########################
                
                # Extract rise and recovery times and drop NaNs
                rise_times = scr_peaks["SCR_RiseTime"].dropna()
                recovery_times = scr_peaks["SCR_RecoveryTime"].dropna()
                
                # Calculate mean and standard deviation
                rise_mean = rise_times.mean()
                rise_std = rise_times.std()
                recovery_mean = recovery_times.mean()
                recovery_std = recovery_times.std()
                
                # Define new thresholds based on mean ± 1 standard deviation
                rise_lower = rise_mean - rise_std
                rise_upper = rise_mean + rise_std
                recovery_lower = recovery_mean - recovery_std
                recovery_upper = recovery_mean + recovery_std
                
                # Check if 80% of rise and recovery times fall within these ranges
                rise_time_pass = rise_times.between(rise_lower, rise_upper).mean() > 0.8
                recovery_time_pass = recovery_times.between(recovery_lower, recovery_upper).mean() > 0.8
                
                #print(f"Adjusted Rise Time Check ({rise_lower:.2f}-{rise_upper:.2f}s):", "Pass" if rise_time_pass else "Fail")
                #print(f"Adjusted Recovery Time Check ({recovery_lower:.2f}-{recovery_upper:.2f}s):", "Pass" if recovery_time_pass else "Fail")
                                
                ################ CHECK STATISTICAL QUALITY ################################
                
                # Calculate skewness and kurtosis of SCR component
                scr_skewness = skew(scr)
                scr_kurtosis = kurtosis(scr)
                
                # Threshold check
                # Assuming `scr_skewness` and `scr_kurtosis` are the skewness and kurtosis of the current SCR segment
                skewness_pass = -3 <= scr_skewness <= 3
                kurtosis_pass = scr_kurtosis < 21
                
                #print("SCR Skewness Check (±2):", "Pass" if skewness_pass else "Fail")
                #print("SCR Kurtosis Check (<20):", "Pass" if kurtosis_pass else "Fail")


                ################## FLATLINE DETECTION ####################################
                
                # Calculate variance over time with a rolling window
                rolling_variance = eda_smooth.rolling(window=64*10).var()  # 10-second window
                flatline_detected = (rolling_variance < 0.0001).any()  # Adjust threshold based on signal characteristics
                
                # Threshold check
                flatline_pass = not flatline_detected
                #print("Flatline or Low-Variance Check:", "Pass" if flatline_pass else "Fail")
                
                ####################### UNUSUALLY HIGH/LOW PEAK DETECTION CHECK ###########################
                
                # Calculate the number of peaks per minute
                peaks_per_minute = scr_peaks['SCR_Peaks'].rolling(window=sampling_rate*60).sum()
                #print("Average peaks per minute:", peaks_per_minute.mean())
                
                # Flag segments with unusually high or low peak counts
                inconsistent_peaks = (peaks_per_minute < 2) | (peaks_per_minute > 15)  # Thresholds, adjust as needed
                
                # Check percentage-based threshold
                
                # Set a threshold for consecutive low-quality data (e.g., 2 minutes)
                consecutive_threshold = 2 * sampling_rate * 60  # 2 minutes in samples
                low_quality_ratio = inconsistent_peaks.sum() / len(scr) * 100
                
                # Check for consecutive low-quality segments
                consecutive_counts = (inconsistent_peaks != inconsistent_peaks.shift()).cumsum()
                consecutive_lengths = inconsistent_peaks.groupby(consecutive_counts).sum()
                long_stretches = consecutive_lengths[consecutive_lengths > consecutive_threshold].count()
                
                # Decide to discard based on both criteria
                if low_quality_ratio > 30 or long_stretches > 0:
                    #print("The signal is considered low quality and should be discarded.")
                    peak_no_pass=False
                else:
                    #print("The signal is acceptable for analysis.")
                    peak_no_pass=True

                
 ###############################################################################################################################################
                
                ######################### SIGNAL QUALITY CHECK ###############################
                
                # Summarize results
                quality_checks = {
                    "Baseline Stability": baseline_stability_pass,
                    "Amplitude Consistency": amplitude_pass,
                    "Rise Time": rise_time_pass,
                    "Recovery Time": recovery_time_pass,
                    "Peak Number": peak_no_pass,
                    "Skewness": skewness_pass,
                    "Kurtosis": kurtosis_pass,
                    "No Flatline": flatline_pass
                }
                
                high_quality = all(quality_checks.values())
                """
                print("\nOverall Signal Quality:", "High" if high_quality else "Low")
                # Output individual check results
                for check, result in quality_checks.items():
                    print(f"{check}: {'Pass' if result else 'Fail'}")
                """
                """
                # Plot the phasic component with detected peaks
                plt.figure(figsize=(12, 4))
                plt.plot(window_data['timestamp'], scr, label="SCR (Phasic Component)", color="blue")  # Use timestamps on x-axis
                plt.scatter(peak_timestamps, scr.iloc[scr_peak_indices], color="red", label="Detected Peaks", marker="o")
                plt.title("EDA Phasic Component with Detected Peaks Method:neurokit (ACCEPTED)")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.show()
                """
                                
                ####################### INTERVAL RELATED FEATURE EXTRACTION ###########################
                """
                if not high_quality:
                    windows_skipped+=1
                    
                    # Summarize results
                    quality_checks = {
                        "Baseline Stability": baseline_stability_pass,
                        "Amplitude Consistency": amplitude_pass,
                        "Rise Time": rise_time_pass,
                        "Recovery Time": recovery_time_pass,
                        "Peak Number": peak_no_pass,
                        "Skewness": skewness_pass,
                        "Kurtosis": kurtosis_pass,
                        "No Flatline": flatline_pass
                    }
                    
                    high_quality = all(quality_checks.values())
                    
                    print(quality_checks)
                    # Plot the phasic component with detected peaks
                    plt.figure(figsize=(12, 4))
                    plt.plot(window_data['timestamp'], scr, label="SCR (Phasic Component)", color="blue")  # Use timestamps on x-axis
                    plt.scatter(peak_timestamps, scr[scr_peak_indices], color="red", label="Detected Peaks", marker="o")
                    plt.title("EDA Phasic Component with Detected Peaks Method:neurokit (Rejected)")
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.legend()
                    plt.show()
                                        
                    #continue
                else:
                """
                ## TIME-DOMAIN FEATURES

                # 1. Number of Peaks
                num_peaks = scr_peaks['SCR_Peaks'].sum()
                
                # 2. Mean SCR Amplitude
                mean_amplitude = scr_peaks['SCR_Amplitude'].mean()
                
                # 3. Mean SCR Rise Time
                mean_rise_time = scr_peaks['SCR_RiseTime'].mean()
                
                # 4. Mean SCR Recovery Time
                mean_recovery_time = scr_peaks['SCR_RecoveryTime'].mean()
                
                # 5. Skin Conductance Level (SCL)
                mean_scl = scl.mean()
                
                # 6. Amplitude Variance (Phasic Component)
                amplitude_variance = scr.var()
                
                # 7. Area Under the Curve (AUC) of SCRs
                # Initialize an empty list for storing AUC values
                auc_values = []
                
                # Loop through each row in scr_peaks to calculate AUC for valid SCR events
                for i in range(len(scr_peaks)):
                    onset = scr_peaks['SCR_Onsets'][i]
                    recovery = scr_peaks['SCR_Recovery'][i]
                    
                    # Check that both onset and recovery indices are not NaN and are within bounds of scr
                    if not np.isnan(onset) and not np.isnan(recovery):
                        if 0 <= onset < len(scr) and 0 <= recovery < len(scr) and recovery > onset:
                            auc = np.trapz(scr[int(onset):int(recovery)])  # Integrate over the valid range
                            auc_values.append(auc)
                
                # Calculate mean AUC if values are available
                mean_auc = np.mean(auc_values) if auc_values else np.nan
                
                
                
                # 8. Peak-to-Peak Interval (PPI) and Variability
                ppi_intervals = np.diff(scr_peaks[scr_peaks['SCR_Peaks'] == 1].index) / sampling_rate  # Convert to seconds
                mean_ppi = np.mean(ppi_intervals) if len(ppi_intervals) > 0 else np.nan
                ppi_variability = np.std(ppi_intervals) if len(ppi_intervals) > 0 else np.nan


                # 10. Power Spectral Density (PSD)
                frequencies, psd = welch(scr, fs=sampling_rate, nperseg=1024)
                low_freq_power = np.trapz(psd[(frequencies >= 0.01) & (frequencies <= 0.1)], frequencies[(frequencies >= 0.01) & (frequencies <= 0.1)])
                high_freq_power = np.trapz(psd[(frequencies > 0.1) & (frequencies <= 0.5)], frequencies[(frequencies > 0.1) & (frequencies <= 0.5)])
                
                # Calculate the power ratio as an additional feature
                power_ratio = low_freq_power / high_freq_power if high_freq_power != 0 else np.nan
                
                ## ADDITIONAL STATISTICAL FEATURES
                
                # 11. Skewness and Kurtosis
                scr_skewness = skew(scr)
                scr_kurtosis = kurtosis(scr)
                
                ## ASSEMBLE FEATURES INTO A DICTIONARY
                window_data['timestamp'].reset_index(drop=True,inplace=True)
                features = {
                    "timestamp": window_data['timestamp'][0],
                    "num_peaks": num_peaks,
                    "mean_amplitude": mean_amplitude,
                    "mean_rise_time": mean_rise_time,
                    "mean_recovery_time": mean_recovery_time,
                    "mean_scl": mean_scl,
                    "amplitude_variance": amplitude_variance,
                    "mean_auc": mean_auc,
                    "mean_ppi": mean_ppi,
                    "ppi_variability": ppi_variability,
                    "low_freq_power": low_freq_power,
                    "high_freq_power": high_freq_power,
                    "power_ratio": power_ratio,
                    "scr_skewness": scr_skewness,
                    "scr_kurtosis": scr_kurtosis,
                    "quality": quality_checks
                }
                
                """
                # Display the extracted features
                for key, value in features.items():
                    print(f"{key}: {value}")
                """
                
                window_features+=[features]
            #print('Number of windows skipped are', windows_skipped, 'out of', num_windows)
            segment_features+=[window_features]
            window_features=[]
        
        # Check if extracted features makes sense ?
        #Match with activpal and check each metric to see whether it makes sense
    
    # Convert to DataFrame
    # Step 1: Flatten the list of lists into a single list of dictionaries
    flat_data = [item for sublist in segment_features for item in sublist]
    
    # Step 2: Convert to DataFrame
    eda_metrics_df = pd.DataFrame(flat_data)
    

    # Check if all values in the dictionary are True for each row
    all_true_count = eda_metrics_df['quality'].apply(lambda d: all(d.values())).sum()
    
    # Output the count
    #print(f"Number of rows where all dictionary values are True: {all_true_count}")
    return eda_metrics_df
                
         
                
                
                
                
                
                
                
                