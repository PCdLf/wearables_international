import os
import shutil
import sys
import re
import zipfile
import pandas as pd
from zipfile import ZipFile
from fastavro import reader

def get_timestamp_column(start_time,sampling_freq,len_list):
    """ 
    -Recevies the starting time, sampling frequency and the length of the dataframe 
    from the associated dataframe.
    -Creates a timestamp column starting from the start_time with the sampling frequency sampling_freq 
    and stops when it reaches the length len_list.
    -Returns the created dataframe.
    """
    start_time_ns = start_time * 1000
    start_timestamp = pd.to_datetime(start_time_ns, unit='ns')

    # Calculate end_timestamp based on the length of the list and sampling frequency
    end_timestamp = start_timestamp + pd.to_timedelta(len_list / sampling_freq, unit='s')

    # Generate a range of timestamps from start to end with the given frequency
    timestamp_column = pd.date_range(start=start_timestamp, end=end_timestamp, freq=pd.to_timedelta(1 / sampling_freq, unit='s'))
    timestamp_df = pd.DataFrame({'timestamp': timestamp_column})
    
    # Convert 'timestamp' column back to Unix timestamp in seconds
    timestamp_df['unix_timestamp'] = timestamp_df['timestamp'].astype('int64') // 10**9

    return timestamp_df

def get_avro_content(avro_file_path):
    """ 
    Reads an Avro file and extracts relevant data into a dictionary of DataFrames.

    - Receives the full file path to an Avro file.
    - Reads the Avro file and extracts data for accelerometer, gyroscope, EDA, temperature, BVP, steps, and systolic peaks.
    - Returns a dictionary of DataFrames containing the parsed data.
    """
    avro_records = []

    # Read the Avro file
    with open(avro_file_path, 'rb') as avro_file:
        avro_reader = reader(avro_file)
        for record in avro_reader:
            acc_start = record['rawData']['accelerometer']['timestampStart']
            acc_sampling_freq = record['rawData']['accelerometer']['samplingFrequency']
            acc_x = record['rawData']['accelerometer']['x']
            acc_y = record['rawData']['accelerometer']['y']
            acc_z = record['rawData']['accelerometer']['z']
            acc_x_df = pd.DataFrame({'x': acc_x})
            acc_y_df = pd.DataFrame({'y': acc_y})
            acc_z_df = pd.DataFrame({'z': acc_z})
            if acc_x_df.empty or acc_y_df.empty or acc_z_df.empty:
                print(f'Accelerometer empty for: {avro_file_path}')
                acc_df = pd.DataFrame()
            else:
                timestamp_df = get_timestamp_column(acc_start, acc_sampling_freq, len_list=len(acc_x))
                acc_df = pd.concat([acc_x_df, acc_y_df, acc_z_df, timestamp_df], axis=1)

            gy_start = record['rawData']['gyroscope']['timestampStart']
            gy_sampling_freq = record['rawData']['gyroscope']['samplingFrequency']
            gy_x = record['rawData']['gyroscope']['x']
            gy_y = record['rawData']['gyroscope']['y']
            gy_z = record['rawData']['gyroscope']['z']
            gy_x_df = pd.DataFrame({'x': gy_x})
            gy_y_df = pd.DataFrame({'y': gy_y})
            gy_z_df = pd.DataFrame({'z': gy_z})
            if gy_x_df.empty or gy_y_df.empty or gy_z_df.empty:
                print(f'Gyroscope empty for: {avro_file_path}')
                gy_df = pd.DataFrame()
            else:
                timestamp_df = get_timestamp_column(gy_start, gy_sampling_freq, len_list=len(gy_x))
                gy_df = pd.concat([gy_x_df, gy_y_df, gy_z_df, timestamp_df], axis=1)

            eda_start = record['rawData']['eda']['timestampStart']
            eda_sampling_freq = record['rawData']['eda']['samplingFrequency']
            eda = record['rawData']['eda']['values']
            eda_df = pd.DataFrame({'eda': eda})
            if eda_df.empty:
                print(f'EDA empty for: {avro_file_path}')
                eda_df = pd.DataFrame()
            else:
                timestamp_df = get_timestamp_column(eda_start, eda_sampling_freq, len_list=len(eda))
                eda_df = pd.concat([eda_df, timestamp_df], axis=1)

            temp_start = record['rawData']['temperature']['timestampStart']
            temp_sampling_freq = record['rawData']['temperature']['samplingFrequency']
            temp = record['rawData']['temperature']['values']
            temp_df = pd.DataFrame({'temp': temp})
            if temp_df.empty:
                print(f'Temperature empty for: {avro_file_path}')
                temp_df = pd.DataFrame()
            else:
                timestamp_df = get_timestamp_column(temp_start, temp_sampling_freq, len_list=len(temp))
                temp_df = pd.concat([temp_df, timestamp_df], axis=1)

            bvp_start = record['rawData']['bvp']['timestampStart']
            bvp_sampling_freq = record['rawData']['bvp']['samplingFrequency']
            bvp = record['rawData']['bvp']['values']
            bvp_df = pd.DataFrame({'bvp': bvp})
            if bvp_df.empty:
                print(f'BVP empty for: {avro_file_path}')
                bvp_df = pd.DataFrame()
            else:
                timestamp_df = get_timestamp_column(bvp_start, bvp_sampling_freq, len_list=len(bvp))
                bvp_df = pd.concat([bvp_df, timestamp_df], axis=1)

            steps_start = record['rawData']['steps']['timestampStart']
            steps_sampling_freq = record['rawData']['steps']['samplingFrequency']
            steps = record['rawData']['steps']['values']
            steps_df = pd.DataFrame({'steps': steps})
            if steps_df.empty:
                print(f'Steps empty for: {avro_file_path}')
                steps_df = pd.DataFrame()
            else:
                timestamp_df = get_timestamp_column(steps_start, steps_sampling_freq, len_list=len(steps))
                steps_df = pd.concat([steps_df, timestamp_df], axis=1)

            systolic_peaks = record['rawData']['systolicPeaks']['peaksTimeNanos']
            systolic_peaks_df = pd.DataFrame({'systolic_peaks': systolic_peaks})

            avro_dicts = {
                'ACC': acc_df,
                'GY': gy_df,
                'EDA': eda_df,
                'TEMP': temp_df,
                'BVP': bvp_df,
                'steps': steps_df,
                'systolic_peaks': systolic_peaks_df
            }

            avro_records.append(avro_dicts)

    return avro_records

def read_e4_plus(root):
    """
    This is the original file which will be called to read the .avro files. It will receive a zip file with all the relevant
    avro files. Split the original path into 2: .zip part and the rest. The assumption is that, we have an example folder
    structure like the following:
        - Embrace Plus Example Data:
            -date1:
                -code1:
                    -digital_biomarkers (aggregated data)
                    -raw_data:
                        -v folder:
                            -1.avro
                            -2.avro
                            -3.avro              
            -date2:
                -code2:
                    -digital_biomarkers (aggregated data)
                    -raw_data:
                        -v folder:
                            -1.avro
                            -2.avro
                            -3.avro            
    After calling other helper functions, it will return the desired avro content.
    """
    avro_content = []

    # Walk through the folder structure to find all .avro files
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.lower().endswith(".avro"):
                avro_file_path = os.path.join(dirpath, file)
                #print(f"Processing Avro file: {avro_file_path}")
                avro_content += get_avro_content(avro_file_path)

    return avro_content



def read(root):
    #print('Python Script Loaded Successfully')
    avro_files=read_e4_plus(root)
    
    return avro_files

