import os
import shutil
import sys
import re
import numpy as np
import zipfile
import pandas as pd
from zipfile import ZipFile


def get_sampling_freq(df):
    """
    Calculates the sampling frequency for the given dataframe.
    Returns the calculated sampling frequency.
    """
    
    df['timestamp']= df['timestamp']
    
    df['timestamp_sec'] = df['timestamp'] 
    
    # Calculate time differences between consecutive samples in seconds
    df['time_diff_sec'] = df['timestamp_sec'].diff()
    
    # Calculate the average time difference in seconds
    average_time_diff_sec = df['time_diff_sec'].mean()
    
    # Invert the average time difference to get the sampling frequency (in samples per second)
    sampling_frequency_sec = 1 / average_time_diff_sec if average_time_diff_sec != 0 else np.nan
    
    return sampling_frequency_sec
    


def read_nowatch(root):
    """
    Reads the csv file from the root path, creates a dictionary of dataframes.
    The assumption is that, we have an example folder structure like the following:
        - NoWatch Example Data:
            -date1.zip:
                -  date1:
                    - 1.csv
                    - 2.csv
                    - 3.csv
            -date2.zip:
                -date2:
                    - 1.csv
                    - 2.csv
                    - 3.csv
    """
    content={}

    zip_files=os.listdir(root)
    zip_files = [(root+'\\') + s for s in zip_files]

    days={}
    for zip_file in zip_files:
        zf = ZipFile(zip_file)
        names = zf.namelist()
        
        measurements={}
        for i in names:
            split_parts = i.split('/')
            rest_of_the_strings = split_parts[-1]
            #print(rest_of_the_strings)
            df = pd.read_csv(zf.open(i))
            """
            sampling_freq=get_sampling_freq(df)
            print(sampling_freq)
            """
            if rest_of_the_strings == 'heartrate.csv' or rest_of_the_strings == 'heart_rate.csv':

                if 'HeartRate' in measurements.keys():
                    measurements['HeartRate'] += df                
                else:
                    measurements['HeartRate'] = df
                    
            if rest_of_the_strings == 'heartbeats.csv':
                if 'HeartBeats' in measurements.keys():
                    measurements['HeartBeats'] += df              
                else:
                    measurements['HeartBeats'] = df
                
            if rest_of_the_strings == 'resting_heart_rate.csv':
                if 'RestingHeartRate' in measurements.keys():
                    measurements['RestingHeartRate'] += df             
                else:
                    measurements['RestingHeartRate'] = df
                    
            if rest_of_the_strings == 'activitytype.csv' or rest_of_the_strings == 'activity_type.csv':
            
                if 'ActivityType' in measurements.keys():
                    measurements['ActivityType'] += df              
                else:
                    measurements['ActivityType'] = df
                    
            if rest_of_the_strings == 'cadence.csv' or rest_of_the_strings == 'activity_cadence.csv':
                
                if 'Cadence' in measurements.keys():
                    measurements['Cadence'] += df             
                else:
                    measurements['Cadence'] = df
            if rest_of_the_strings == 'activity_count.csv':

                if 'ActivityCount' in measurements.keys():
                    measurements['ActivityCount'] += df               
                else:
                    measurements['ActivityCount'] = df
                    
            if rest_of_the_strings == 'stresslevel.csv' or rest_of_the_strings == 'stress_level.csv':

                if 'StressLevel' in measurements.keys():
                    measurements['StressLevel'] += df          
                else:
                    measurements['StressLevel'] = df
                    
            if rest_of_the_strings == 'respirationrate.csv' or rest_of_the_strings == 'respiration_rate.csv':
            
                if 'RespirationRate' in measurements.keys():
                    measurements['RespirationRate'] += df            
                else:
                    measurements['RespirationRate'] = df
                    
            if rest_of_the_strings == 'cortisollevels.csv':

                if 'Cortisolevels' in measurements.keys():
                    measurements['Cortisolevels'] += df                
                else:
                    measurements['Cortisolevels'] = df
                    
            if rest_of_the_strings == 'temperature.csv':

                if 'Temperature' in measurements.keys():
                    measurements['Temperature'] +=df            
                else:
                    measurements['Temperature'] = df
                    
            if rest_of_the_strings == 'sleepsession.csv' or rest_of_the_strings == 'sleep_session.csv':

                if 'SleepSession' in measurements.keys():
                    measurements['SleepSession'] += df             
                else:
                    measurements['SleepSession'] = df
                    
            if rest_of_the_strings == 'userprofile.csv' or rest_of_the_strings == 'profile.csv':

                if 'UserProfile' in measurements.keys():
                    measurements['UserProfile'] +=df             
                else:
                    measurements['UserProfile'] =df
                    
            if rest_of_the_strings == 'spo2.csv':

                if 'Spo2' in measurements.keys():
                    measurements['Spo2'] += df                
                else:
                    measurements['Spo2'] = df
                    
            if rest_of_the_strings == 'system_event.csv':
                if 'SystemEvent' in measurements.keys():
                    measurements['SystemEvent'] += df            
                else:
                    measurements['SystemEvent'] = df

                 
                    
                    
        split_parts = i.split('/')
        day_code = split_parts[0]
        
        content[day_code]=measurements
        measurements={}
    return content



"""
EXAMPLE USAGE BELOW
"""
#print('Python Script Loaded Successfully')
root=r'C:\Users\acans\Desktop\NoWatch Example Data' 
nowatch_files=read_nowatch(root)