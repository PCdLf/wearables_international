# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:34:55 2025

@author: selaca
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:23:46 2024

@author: selaca
"""

import os
import pandas as pd
import numpy as np

from zipfile import ZipFile

from scipy.signal import welch

from scipy.signal import butter, filtfilt

import neurokit2 as nk
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import skew, kurtosis
from scipy.signal import savgol_filter
from scipy.signal import medfilt


import warnings
warnings.filterwarnings('ignore')

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

def normalize_signal(signal):
    # Normalize signal to [0, 1]
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y



def process_peaks(peaks):
    running_avg = 0
    counts = 0
    results = []

    for peak in peaks:
        if counts == 0:
            # First peak
            running_avg = peak
            results.append((peak, running_avg, False))  # (Peak value, running average, is artifact)
            counts+=1
        else:
            # Calculate 20% above the current running average
            upper_threshold = 1.5 * running_avg  # 20% above the running average
            #lower_threshold = 0.95 * running_avg  # 20% below the running average

            if peak > upper_threshold:
                # Mark this peak as an artifact
                results.append((peak, running_avg, True))
            else:
                # Update running average
                running_avg = (running_avg * counts + peak) / (counts + 1)
                results.append((peak, running_avg, False))
                counts += 1
                
                
    return results


    

def reclassify_artifacts(artifact_peaks, normal_peaks, peaks, peak_amplitudes, signal_window):
    # Convert lists to arrays for easier indexing
    peaks = np.array(peaks)
    peak_amplitudes = np.array(peak_amplitudes)
    
    # Initialize new lists for updated classifications
    updated_normal_peaks = list(normal_peaks)  # Start with all initially classified normal peaks
    updated_artifact_peaks = []
    
    # Process each artifact peak
    for artifact_idx in artifact_peaks:
        # Find the index of the artifact peak in the original 'peaks' array
        artifact_peak_amplitude = peak_amplitudes[np.where(peaks == artifact_idx)[0][0]]
        
        # Collect amplitudes of the last four normal peaks before this artifact
        preceding_normal_indices = [i for i in normal_peaks if i < artifact_idx]
        
        if len(preceding_normal_indices) >= 4:
            # Only consider the last four
            last_four_normals = preceding_normal_indices[-4:]
            last_four_amplitudes = peak_amplitudes[[np.where(peaks == idx)[0][0] for idx in last_four_normals]]
            
            # Calculate the average amplitude of these four normal peaks
            average_amplitude = np.mean(last_four_amplitudes)
            
            # Check if the artifact peak is within 20% of this average
            #lower_bound = average_amplitude * 0.8
            upper_bound = average_amplitude * 1.5
            
            if artifact_peak_amplitude <= upper_bound:
                # Reclassify as normal
                updated_normal_peaks.append(artifact_idx)
            else:
                # Keep as artifact
                updated_artifact_peaks.append(artifact_idx)
        else:
            # Not enough normal peaks before this artifact, keep as artifact
            updated_artifact_peaks.append(artifact_idx)

    return updated_normal_peaks, updated_artifact_peaks

def process(bvp_df):   
    bvp_df['timestamp'] = pd.to_datetime(bvp_df['timestamp'])
    bvp_df['timestamp'] = bvp_df['timestamp'].dt.tz_localize(None)
    bvp_df = bvp_df.sort_values('timestamp')
     
    #Find interrupted segments:
    # Calculate differences between consecutive timestamps
    bvp_df['time_diff'] = bvp_df['timestamp'].diff()
    bvp_df['new_segment'] = bvp_df['time_diff'] > pd.Timedelta(minutes=1)
    bvp_df['segment_id'] = bvp_df['new_segment'].cumsum()
    
    #sampling_rate = calculate_sampling_rate(bvp_df, 'timestamp')
    #print(f"Sampling rate: {sampling_rate:.2f} Hz")
    
    skipped_segment=0
    hrv_metrics_full=[]
    for segment_id, segment_data in bvp_df.groupby('segment_id'):
        
        ################# RESAMPLE SEGMENT TO MAKE SURE 64 Hz ##############
        segment_data.set_index('timestamp', inplace=True)
        segment_timestamps = segment_data.index.copy()
        resampled_continuous = segment_data.resample('15.625L')
        resampled_df = segment_data
        
        """
        start_time2 = "2020-12-08 08:39:50"
        end_time2 = "2020-12-08 08:41:00"
        
        plt.figure(figsize=(10, 4))
        plt.plot(segment_timestamps,resampled_df['BVP_value'])
        plt.title('PPG Original')
        plt.xlabel('Sample number')
        plt.ylabel('PPG signal')
        plt.xlim(pd.to_datetime(start_time2), pd.to_datetime(end_time2))
        plt.show()
        """
        ############################# NORMALIZE SIGNALS ##########################################
       
        resampled_df['normalized_BVP_value']=normalize_signal(resampled_df['bvp'])
        
        """
        start_time2 = "2020-12-08 06:39:50"
        end_time2 = "2020-12-08 06:41:00"
        
        plt.figure(figsize=(10, 4))
        plt.plot(segment_timestamps,resampled_df['normalized_BVP_value'])
        plt.title('PPG Original')
        plt.xlabel('Sample number')
        plt.ylabel('PPG signal')
        plt.xlim(pd.to_datetime(start_time2), pd.to_datetime(end_time2))
        plt.show()
        """
        ############################### BUTTERWORTH BANDPASS FILTER #################################
        sampling_rate=64
        
        # Filter parameters
        order = 4
     
        lowcut = 0.4  # Hz
        highcut = 3  # Hz
                     
        # Apply the Butterworth bandpass filter
        resampled_df['filtered_BVP_signal']= butter_bandpass_filter(resampled_df['normalized_BVP_value'], lowcut, highcut, sampling_rate, order)
        """
        start_time2 = "2020-12-08 06:39:50"
        end_time2 = "2020-12-08 06:41:00"
        
        plt.figure(figsize=(10, 4))
        plt.plot(segment_timestamps,resampled_df['filtered_BVP_signal'])
        plt.title('PPG Original')
        plt.xlabel('Sample number')
        plt.ylabel('PPG signal')
        plt.xlim(pd.to_datetime(start_time2), pd.to_datetime(end_time2))
        plt.show()
        """
        
        ###################### SAVITSKY-GOLAY FILTER ###########################

        savitsky_bvp = savgol_filter(resampled_df['filtered_BVP_signal'], window_length=15, polyorder=2)  # Adjust parameters as needed
        resampled_df['savitsky_bvp'] = savitsky_bvp

        """
        start_time2 = "2020-12-08 06:39:50"
        end_time2 = "2020-12-08 06:41:00"
        
        plt.figure(figsize=(10, 4))
        plt.plot(segment_timestamps,savitsky_bvp)
        plt.title('PPG Original')
        plt.xlabel('Sample number')
        plt.ylabel('PPG signal')
        plt.xlim(pd.to_datetime(start_time2), pd.to_datetime(end_time2))
        plt.show()
        """

        ####################### MEDIAN FILTERING ######################
        

        medfilt_bvp = medfilt(savitsky_bvp, kernel_size=5)
        resampled_df['medfilt_bvp'] = medfilt_bvp
        
        """
        start_time2 = "2020-12-08 08:39:50"
        end_time2 = "2020-12-08 08:41:00"
        
        plt.figure(figsize=(10, 4))
        plt.plot(segment_timestamps,medfilt_bvp)
        plt.title('PPG Original')
        plt.xlabel('Sample number')
        plt.ylabel('PPG signal')
        plt.xlim(pd.to_datetime(start_time2), pd.to_datetime(end_time2))
        plt.show()
        """
        
        
        ################## NEUROKIT PREPROCESSING #####################################
        
        
        sampling_rate = 64  # Adjust based on your device

        # Clean the PPG signal
        cleaned_bvp = nk.ppg_clean(medfilt_bvp, sampling_rate=sampling_rate)
        resampled_df['cleaned_bvp'] = cleaned_bvp
        
        """
        start_time2 = "2020-12-07 21:39:50"
        end_time2 = "2020-12-07 21:41:00"
        
        plt.figure(figsize=(10, 4))
        plt.plot(segment_timestamps,cleaned_bvp)
        plt.title('PPG Original')
        plt.xlabel('Sample number')
        plt.ylabel('PPG signal')
        plt.xlim(pd.to_datetime(start_time2), pd.to_datetime(end_time2))
        plt.show()
        """
        
        ################################################################################
        window_length_in_seconds = 180  # seconds
        window_length_in_samples = sampling_rate * window_length_in_seconds
        overlap = 0.80  # 50% overlap
        shift = int(window_length_in_samples * (1 - overlap))  # Number of samples to shift for each window
        num_windows = int(np.floor((len(resampled_df['cleaned_bvp']) - window_length_in_samples) / shift) + 1)
        artifact_threshold = 0.60  # 30% threshold
        
        if len(resampled_df['cleaned_bvp'])< window_length_in_samples:
            print('Segment id:', segment_id, ' too short to be processed!' )
            print(len(resampled_df['cleaned_bvp']))
            continue

        window_hrv_metrics=[]
        # Prepare to collect BPM results
        for i in range(num_windows):
            # Extract the window
            start_idx = i * shift  # This is where the overlap is taken into account
            end_idx = start_idx + window_length_in_samples
            signal_window = resampled_df['cleaned_bvp'][start_idx:end_idx].values
            signal_timestamp = resampled_df.index[start_idx:end_idx]
            
            
            """
            start_time2 = "2020-12-07 21:39:50"
            end_time2 = "2020-12-07 21:41:00"
            
            plt.figure(figsize=(10, 4))
            plt.plot(signal_timestamp,signal_window)
            plt.title('PPG Original')
            plt.xlabel('Sample number')
            plt.ylabel('PPG signal')
            plt.xlim(pd.to_datetime(start_time2), pd.to_datetime(end_time2))
            
            plt.show()
            """
           

            ################### NEUROKIT SIGNAL SMOOTHING #####################
            
            # Convert the signal to a Pandas Series
            signal_window_series = pd.Series(signal_window.copy())
            
            # Detect spikes using a rolling standard deviation or NeuroKit’s processing
            rolling_std = signal_window_series.rolling(window=64).std()
            spikes = np.abs(signal_window_series - signal_window_series.mean()) > (1.5 * rolling_std.mean())  # Adjust multiplier based on threshold
            
            # Remove detected spikes (set to NaN)
            signal_window_series[spikes] = np.nan
            
            # Interpolate over NaNs from detected spikes
            signal_window_series = signal_window_series.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            # Optional: Additional smoothing (apply if needed for residual noise/artifacts)
            signal_window_series = nk.signal_smooth(signal_window_series, method="gaussian")
            
           

            ############################# VISUALIZE #############################
            """
            # Plot original vs. cleaned signal
            plt.figure(figsize=(12, 6))
            plt.plot(signal_window, label="Original BVP Signal", color="gray", alpha=0.6)
            plt.plot(signal_window_series, label="Cleaned BVP Signal", color="blue")
              
            plt.title("Original vs. Smoothed BVP Signal Gaussian")
            plt.xlabel("Sample Index")
            plt.ylabel("BVP Amplitude")
            plt.legend()
            plt.grid(True)
            #plt.xlim(2000,4000)
            plt.show()

            """        
            
            ######################### NEUROKIT CLEAN ################################
            

            cleaned_signal_elgendi = nk.ppg_clean(signal_window_series, sampling_rate=sampling_rate, method="elgendi")
                        
            # Plot to compare methods
            
            """
            #start_time2 = "2020-12-07 21:39:50"
            #end_time2 = "2020-12-07 21:41:00"
            
            plt.figure(figsize=(12, 6))
            plt.plot(signal_timestamp,signal_window_series, label="Original BVP Signal", color="gray", alpha=0.6)
            plt.plot(signal_timestamp, cleaned_signal_elgendi, label="Elgendi Method", color="green")
            plt.title("Comparison of Different NeuroKit BVP Cleaning Methods")
            plt.xlabel("Sample Index")
            plt.ylabel("BVP Amplitude")
            plt.legend()
            plt.grid(True)
            #plt.xlim(pd.to_datetime(start_time2), pd.to_datetime(end_time2))
            plt.show()
            """

            ########################
            
            # Assuming adaptively_smoothed_signal is the processed BVP signal
            sampling_rate = 64  # Adjust according to your data
            signal_quality = {}
            
            # Calculate Peak-to-Peak Intervals
            peaks_info = nk.ppg_findpeaks(cleaned_signal_elgendi, sampling_rate=sampling_rate)
            peaks = peaks_info["PPG_Peaks"]
            peak_intervals = np.diff(peaks) / sampling_rate  # Intervals in seconds
            # Convert intervals to a DataFrame for easier manipulation
            peak_intervals_df = pd.DataFrame({"IBI": peak_intervals})
            
            # Store peak interval-related metrics separately
            peak_interval_mean = np.mean(peak_intervals)
            peak_interval_std = np.std(peak_intervals)
            peak_interval_variability = (peak_interval_std / peak_interval_mean) * 100  # Variability as a percentage
            
            # Store peak-related metrics in signal_quality
            signal_quality["Mean_peak_interval"] = {"value": peak_interval_mean, "description": "Mean time between peaks (s)"}
            signal_quality["Std_dev_peak_interval"] = {"value": peak_interval_std, "description": "Standard deviation of peak intervals (s)"}
            signal_quality["Peak_interval_variability"] = {"value": peak_interval_variability, "description": "Variability in peak intervals (%)"}
            
            """
            # Plot the BVP signal with detected peaks
            plt.figure(figsize=(12, 6))
            plt.plot(adaptively_smoothed_signal, label="Adaptively Smoothed BVP Signal", color="blue")
            plt.plot(peaks, adaptively_smoothed_signal[peaks], "ro", label="Detected Peaks")  # Mark peaks with red dots
            plt.xlabel("Sample Index")
            plt.ylabel("BVP Amplitude")
            plt.title("Adaptively Smoothed BVP Signal with Detected Peaks")
            plt.legend()
            plt.xlim(9000,12000)
            plt.show()
            """



            
            # Signal Amplitude Check
            signal_std = np.std(cleaned_signal_elgendi)
            min_threshold = -3 * signal_std
            max_threshold = 3 * signal_std
            min_val, max_val = np.min(cleaned_signal_elgendi), np.max(cleaned_signal_elgendi)
            signal_quality['Amplitude_check'] = {
                "min_threshold": min_threshold,
                "max_threshold": max_threshold,
                "min_val": min_val,
                "max_val": max_val,
                "within_range": min_threshold <= min_val <= max_val <= max_threshold
            }
            """
            # Low Amplitude Segments
            low_amplitude_threshold = 0.005
            low_amplitude_ratio = (pd.Series(adaptively_smoothed_signal).abs() < low_amplitude_threshold).mean()
            signal_quality['Low_amp_check'] = {
                "low_amplitude_threshold": low_amplitude_threshold,
                "low_amplitude_ratio": low_amplitude_ratio,
                "exceeds_threshold": low_amplitude_ratio > 0.2  # 20% threshold
            }
            """
            # Baseline Stability
            window_size = 64
            rolling_std = pd.Series(cleaned_signal_elgendi).rolling(window=window_size, center=True).std()
            baseline_drift = rolling_std.mean()
            baseline_drift_threshold = 0.02
            signal_quality['Baseline_drift_check'] = {
                "baseline_drift": baseline_drift,
                "baseline_drift_threshold": baseline_drift_threshold,
                "within_threshold": baseline_drift <= baseline_drift_threshold
            }
            
            # Signal to Noise Ratio (SNR)
            signal_power = np.mean(cleaned_signal_elgendi ** 2)
            noise_power = np.var(np.diff(cleaned_signal_elgendi))
            snr = 10 * np.log10(signal_power / noise_power)
            snr_threshold = 3
            signal_quality['SNR'] = {
                "signal_power": signal_power,
                "noise_power": noise_power,
                "snr": snr,
                "snr_threshold": snr_threshold,
                "meets_threshold": snr >= snr_threshold
            }
            
            # Zero Crossing Rate (ZCR)
            zero_crossings = np.where(np.diff(np.sign(cleaned_signal_elgendi)))[0]
            zcr = len(zero_crossings) / len(cleaned_signal_elgendi)
            zcr_threshold = 0.03
            signal_quality['Zero_crossing_rate'] = {
                "zcr": zcr,
                "zcr_threshold": zcr_threshold,
                "meets_threshold": zcr <= zcr_threshold
            }
            
            # Power Spectral Density (PSD)
            freqs, psd = welch(cleaned_signal_elgendi, fs=sampling_rate)
            low_freq_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 3)])
            high_freq_power = np.sum(psd[freqs > 3])
            psd_ratio = low_freq_power / high_freq_power if high_freq_power > 0 else np.inf
            psd_threshold_ratio = 0.7
            signal_quality['PSD'] = {
                "low_freq_power": low_freq_power,
                "high_freq_power": high_freq_power,
                "psd_ratio": psd_ratio,
                "psd_threshold_ratio": psd_threshold_ratio,
                "meets_threshold": psd_ratio >= psd_threshold_ratio
            }
            
            # Skewness and Kurtosis
            signal_skewness = skew(cleaned_signal_elgendi)
            signal_kurtosis = kurtosis(cleaned_signal_elgendi)
            skewness_threshold = (-1.5, 1.5)
            kurtosis_threshold = (1, 10)
            signal_quality['Skewness'] = {
                "signal_skewness": signal_skewness,
                "skewness_threshold": skewness_threshold,
                "within_range": skewness_threshold[0] <= signal_skewness <= skewness_threshold[1]
            }
            signal_quality['Kurtosis'] = {
                "signal_kurtosis": signal_kurtosis,
                "kurtosis_threshold": kurtosis_threshold,
                "within_range": kurtosis_threshold[0] <= signal_kurtosis <= kurtosis_threshold[1]
            }
            
            """
            # Print Summary of Quality Assessment
            print("\nDetailed Signal Quality Assessment:")
            
            # Print peak-to-peak intervals separately
            print("\nPeak-to-Peak Intervals (first 5 shown):", peak_intervals[:5])
            print("\nPeak Interval Metrics:")
            for metric, details in signal_quality.items():
                if metric.startswith("Mean_peak_interval") or metric.startswith("Std_dev_peak_interval") or metric.startswith("Peak_interval_variability"):
                    print(f"  {metric}: {details['value']} ({details['description']})")
            
            # Print other metrics
            for metric, results in signal_quality.items():
                if not metric.startswith("Mean_peak_interval") and not metric.startswith("Std_dev_peak_interval") and not metric.startswith("Peak_interval_variability"):
                    print(f"\nMetric: {metric}")
                    for key, value in results.items():
                        print(f"  {key}: {value}")

            """
            
            ######################## HRV ANALYSIS ##############################
            
            # Calculate Peak-to-Peak Intervals
            peaks_info = nk.ppg_findpeaks(cleaned_signal_elgendi, sampling_rate=sampling_rate)
            peaks = peaks_info["PPG_Peaks"]
            peak_intervals = np.diff(peaks) / sampling_rate  # Intervals in seconds
            # Convert intervals to a DataFrame for easier manipulation
            
            smoothed_intervals = np.convolve(peak_intervals, np.ones(5)/5, mode='same')
            
            #peak reconstruction:
            # Initialize adjusted peaks array with the first original peak as a starting point
            adjusted_peaks = [peaks[0]]
            
            # Calculate subsequent peaks based on smoothed intervals
            for interval in smoothed_intervals:
                next_peak = adjusted_peaks[-1] + int(interval * sampling_rate)
                adjusted_peaks.append(next_peak)
            
            # Convert adjusted_peaks to a numpy array if needed
            adjusted_peaks = np.array(adjusted_peaks)
            
            hrv_metrics = nk.hrv(adjusted_peaks, sampling_rate=64, show=False)  # Set sampling_rate=1000 as data is in ms
            #print('Adjusted peaks:',hrv_metrics.to_dict())
            
            
            average_interval = np.mean(smoothed_intervals)
            bpm = 60 / average_interval
            """
            print('Average_interval:',average_interval)
            print('Mean NN:', hrv_metrics['HRV_MeanNN'])
            """
            hrv_metrics['HRV_BPM'] = bpm
            hrv_metrics['timestamp'] = signal_timestamp[0]
            """
            # Display HRV Metrics in a readable format
            print("\n--- HRV Metrics ---\n")
            
            # Iterate through each HRV metric calculated and print in a formatted way
            for metric, value in hrv_metrics.iloc[0].items():
                print(f"{metric}: {value}")
            """
            #add quality metrics to hrv_metrics
            window_hrv_metrics+= [hrv_metrics]
        if window_hrv_metrics !=[]:
            hrv_metrics_full+=[pd.concat(window_hrv_metrics,axis=0,ignore_index=True)]
        else:
            print('segment:', segment_id, 'is skipped.')
        print('segment:', segment_id, 'done.')
    hrv_results=pd.concat(hrv_metrics_full,axis=0,ignore_index=True)
    #hrv_results.to_csv('C:\\Users\\selaca\\Desktop\\hrv_results.csv')
    
    single_hrv= hrv_results.drop_duplicates(subset='timestamp', keep='first') 
    
    return single_hrv
    """
    #################### QUALITY CHECKS #######################################
    expected_ranges = {
    'HRV_MeanNN': (400, 1200),           # Mean NN Interval in ms
    'HRV_SDNN': (20, 100),               # SDNN in ms
    'HRV_SDANN1': (5, 80),               # SDANN over short segments in ms
    'HRV_SDNNI1': (5, 80),               # SDNN Index in ms
    'HRV_SDANN2': (5, 80),               # SDANN over larger segments in ms
    'HRV_SDNNI2': (5, 80),               # SDNNI over larger segments in ms
    'HRV_SDANN5': (5, 80),               # SDANN over longest segments in ms
    'HRV_SDNNI5': (5, 80),               # SDNN Index over longest segments in ms
    'HRV_RMSSD': (10, 80),               # RMSSD in ms
    'HRV_SDSD': (10, 80),                # SDSD in ms
    'HRV_CVNN': (0.1, 1.0),              # Coefficient of Variation for NN
    'HRV_CVSD': (0.1, 1.0),              # Coefficient of Variation for SD
    'HRV_MedianNN': (400, 1200),         # Median NN Interval in ms
    'HRV_MadNN': (10, 300),              # Median Absolute Deviation for NN
    'HRV_MCVNN': (0.1, 1.0),             # Modified Coefficient of Variation for NN
    'HRV_IQRNN': (50, 500),              # Interquartile Range for NN
    'HRV_SDRMSSD': (0.1, 2.0),           # Ratio of SDNN/RMSSD
    'HRV_Prc20NN': (300, 1200),          # 20th Percentile NN Interval in ms
    'HRV_Prc80NN': (300, 1200),          # 80th Percentile NN Interval in ms
    'HRV_pNN50': (0, 100),               # Percentage of NN intervals > 50 ms
    'HRV_pNN20': (0, 100),               # Percentage of NN intervals > 20 ms
    'HRV_MinNN': (300, 1200),            # Minimum NN Interval in ms
    'HRV_MaxNN': (300, 2500),            # Maximum NN Interval in ms
    'HRV_HTI': (0, 50),                  # HRV Triangular Index
    'HRV_TINN': (50, 600),               # TINN in ms
    'HRV_ULF': (0, 0.01),                # ULF Power
    'HRV_VLF': (0, 0.1),                 # VLF Power
    'HRV_LF': (0, 0.15),                 # LF Power
    'HRV_HF': (0, 0.4),                  # HF Power
    'HRV_VHF': (0, 0.05),                # VHF Power
    'HRV_TP': (0, 1.0),                  # Total Power
    'HRV_LFHF': (0, 5),                  # LF/HF Ratio
    'HRV_LFn': (0, 100),                 # Normalized LF Power
    'HRV_HFn': (0, 100),                 # Normalized HF Power
    'HRV_LnHF': (-10, 10),               # Log-transformed HF Power
    'HRV_SD1': (10, 50),                 # SD1 from Poincaré Plot in ms
    'HRV_SD2': (30, 150),                # SD2 from Poincaré Plot in ms
    'HRV_SD1SD2': (0, 1),                # Ratio of SD1 to SD2
    'HRV_S': (0, 1e6),                   # Area of Poincaré plot
    'HRV_CSI': (0, 10),                  # Cardiac Sympathetic Index
    'HRV_CVI': (0, 10),                  # Cardiac Vagal Index
    'HRV_CSI_Modified': (0, 5000),       # Modified CSI
    'HRV_PIP': (0, 1),                   # Poincaré plot Index
    'HRV_IALS': (0, 1),                  # Index of Asymmetry in NN intervals
    'HRV_PSS': (0, 1),                   # Point Symmetry Score
    'HRV_PAS': (0, 1),                   # Point Asymmetry Score
    'HRV_GI': (0, 100),                  # Geometric Index
    'HRV_SI': (0, 100),                  # Symmetry Index
    'HRV_AI': (0, 100),                  # Asymmetry Index
    'HRV_PI': (0, 100),                  # Poincaré Index
    'HRV_C1d': (0, 1),                   # First Poincaré descriptor in one direction
    'HRV_C1a': (0, 1),                   # First Poincaré descriptor in opposite direction
    'HRV_SD1d': (0, 300),                # SD1 in descending direction
    'HRV_SD1a': (0, 300),                # SD1 in ascending direction
    'HRV_C2d': (0, 1),                   # Second Poincaré descriptor in one direction
    'HRV_C2a': (0, 1),                   # Second Poincaré descriptor in opposite direction
    'HRV_SD2d': (0, 300),                # SD2 in descending direction
    'HRV_SD2a': (0, 300),                # SD2 in ascending direction
    'HRV_Cd': (0, 1),                    # Combined descriptor d
    'HRV_Ca': (0, 1),                    # Combined descriptor a
    'HRV_SDNNd': (0, 300),               # SDNN in descending direction
    'HRV_SDNNa': (0, 300),               # SDNN in ascending direction
    'HRV_DFA_alpha1': (0, 1.5),          # Detrended Fluctuation Analysis alpha1
    'HRV_MFDFA_alpha1_Width': (0, 1.5),  # Multifractal DFA alpha1 Width
    'HRV_MFDFA_alpha1_Peak': (0, 2.0),   # Multifractal DFA alpha1 Peak
    'HRV_MFDFA_alpha1_Mean': (0, 2.0),   # Multifractal DFA alpha1 Mean
    'HRV_MFDFA_alpha1_Max': (0, 2.0),    # Multifractal DFA alpha1 Max
    'HRV_MFDFA_alpha1_Delta': (-1, 1),   # Multifractal DFA alpha1 Delta
    'HRV_MFDFA_alpha1_Asymmetry': (-1, 1), # Multifractal DFA alpha1 Asymmetry
    'HRV_MFDFA_alpha1_Fluctuation': (0, 1), # Multifractal DFA alpha1 Fluctuation
    'HRV_MFDFA_alpha1_Increment': (0, 1),   # Multifractal DFA alpha1 Increment
    'HRV_DFA_alpha2': (0, 1.5),          # DFA alpha2
    'HRV_MFDFA_alpha2_Width': (0, 1.5),  # Multifractal DFA alpha2 Width
    'HRV_MFDFA_alpha2_Peak': (0, 2.0),   # Multifractal DFA alpha2 Peak
    'HRV_MFDFA_alpha2_Mean': (0, 2.0),   # Multifractal DFA alpha2 Mean
    'HRV_MFDFA_alpha2_Max': (0, 2.0),    # Multifractal DFA alpha2 Max
    'HRV_MFDFA_alpha2_Delta': (-1, 1),   # Multifractal DFA alpha2 Delta
    'HRV_MFDFA_alpha2_Asymmetry': (-1, 1), # Multifractal DFA alpha2 Asymmetry
    'HRV_MFDFA_alpha2_Fluctuation': (0, 1), # Multifractal DFA alpha2 Fluctuation
    'HRV_MFDFA_alpha2_Increment': (0, 1),   # Multifractal DFA alpha2 Increment
    'HRV_ApEn': (0, 2),                  # Approximate Entropy
    'HRV_SampEn': (0, 2),                # Sample Entropy
    'HRV_ShanEn': (0, 10),               # Shannon Entropy
    'HRV_FuzzyEn': (0, 2),               # Fuzzy Entropy
    'HRV_MSEn': (0, 2),                  # Multiscale Entropy
    'HRV_CMSEn': (0, 2),                 # Composite Multiscale Entropy
    'HRV_RCMSEn': (0, 2),                # Refined Composite Multiscale Entropy
    'HRV_CD': (0, 5),                    # Correlation Dimension
    'HRV_HFD': (0, 5),                   # Higuchi Fractal Dimension
    'HRV_KFD': (0, 5),                   # Katz Fractal Dimension
    'HRV_LZC': (0, 2),                   # Lempel-Ziv Complexity
    'HRV_BPM': (40, 200)                 # Heart Rate (BPM)
}
    # Initialize a dictionary to store results of out-of-range values
    out_of_range_summary = {}
    
    # Check each column against its expected range
    for metric, (min_val, max_val) in expected_ranges.items():
        # Identify rows where the values are out of the expected range
        out_of_range = single_hrv[(single_hrv[metric] < min_val) | (single_hrv[metric] > max_val)]
        
        # Store count and example out-of-range rows for review
        out_of_range_summary[metric] = {
            'out_of_range_count': len(out_of_range),
            'sample_out_of_range_values': out_of_range[[metric, 'timestamp']].head()  # show a few examples with timestamp
        }
    
    # Display summary report
    for metric, summary in out_of_range_summary.items():
        print(f"\nMetric: {metric}")
        print(f"Out-of-range count: {summary['out_of_range_count']}")
        print("Sample out-of-range values:")
        print(summary['sample_out_of_range_values'])

    # Convert both 'timestamp' columns to ensure they're both in UTC
    hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp']).dt.tz_localize(None)
    single_hrv['timestamp'] = pd.to_datetime(single_hrv['timestamp']).dt.tz_localize(None)

    # Step 1: Merge the DataFrames on 'timestamp'
    merged_df = pd.merge(hr_df[['timestamp', 'HR_value']], single_hrv[['timestamp', 'HRV_BPM']], on='timestamp', how='inner')
    
    # Step 2: Calculate the absolute difference between HR_Value and HRV_BPM
    merged_df['Difference'] = abs(merged_df['HR_value'] - merged_df['HRV_BPM'])
    
    # Step 3: Calculate statistics to analyze closeness
    mean_diff = merged_df['Difference'].mean()  # Mean absolute difference
    max_diff = merged_df['Difference'].max()    # Maximum difference
    min_diff = merged_df['Difference'].min()    # Minimum difference
    
    # Optional: Percentage of values within a specific threshold (e.g., within 5 BPM)
    threshold = 5
    within_threshold_percentage = (merged_df['Difference'] <= threshold).mean() * 100
    
    # Display results
    print("Mean Absolute Difference:", mean_diff)
    print("Max Difference:", max_diff)
    print("Min Difference:", min_diff)
    print(f"Percentage of differences within {threshold} BPM:", within_threshold_percentage)

    # Step 0: Convert 'IBI_duration_s' to milliseconds
    ibi_df['IBI_duration_ms'] = ibi_df['IBI_duration_s'] * 1000
    
    # Ensure 'timestamp' columns are timezone-naive
    ibi_df['timestamp'] = pd.to_datetime(ibi_df['timestamp']).dt.tz_localize(None)
    single_hrv['timestamp'] = pd.to_datetime(single_hrv['timestamp']).dt.tz_localize(None)
    
    ibi_df['IBI_duration_ms'] = pd.to_numeric(ibi_df['IBI_duration_s'], errors='coerce') * 1000

    # Convert 'HRV_MeanNN' to numeric in case it contains any non-numeric values
    single_hrv['HRV_MeanNN'] = pd.to_numeric(single_hrv['HRV_MeanNN'], errors='coerce')

    
    # Step 1: Merge the DataFrames on 'timestamp'
    merged_ibi_df = pd.merge(
        ibi_df[['timestamp', 'IBI_duration_ms']],
        single_hrv[['timestamp', 'HRV_MeanNN']],
        on='timestamp',
        how='inner'
    )
    
    # Step 2: Calculate the absolute difference between IBI_duration_ms and HRV_MeanNN
    merged_ibi_df['Difference'] = abs(merged_ibi_df['IBI_duration_ms'] - merged_ibi_df['HRV_MeanNN'])
    
    # Step 3: Calculate statistics to analyze closeness
    mean_diff_ibi = merged_ibi_df['Difference'].mean()  # Mean absolute difference
    max_diff_ibi = merged_ibi_df['Difference'].max()    # Maximum difference
    min_diff_ibi = merged_ibi_df['Difference'].min()    # Minimum difference
    
    # Optional: Percentage of values within a specific threshold (e.g., within 5 ms)
    threshold_ibi = 100
    within_threshold_percentage_ibi = (merged_ibi_df['Difference'] <= threshold_ibi).mean() * 100
    
    # Display results
    print("Mean Absolute Difference (IBI vs HRV_MeanNN):", mean_diff_ibi)
    print("Max Difference (IBI vs HRV_MeanNN):", max_diff_ibi)
    print("Min Difference (IBI vs HRV_MeanNN):", min_diff_ibi)
    print(f"Percentage of differences within {threshold_ibi} ms:", within_threshold_percentage_ibi)
    """
   
    
    



    
    
    
   
    
    
    
    

            
            