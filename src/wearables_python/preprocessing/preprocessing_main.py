import sys
import os
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers import read_plus
#from helpers import read_nowatch

from preprocessing_e4_plus import preprocessing_eda
from preprocessing_e4_plus import preprocessing_temp
from preprocessing_e4_plus import preprocessing_acc
from preprocessing_e4_plus import preprocessing_bvp

def combine_dataframes(data_list):
    """
    Combines DataFrames for each key across multiple dictionaries in a list.
    
    :param data_list: List of dictionaries where each dictionary contains DataFrames for specific keys.
    :return: A dictionary of combined DataFrames for each key.
    """
    combined_data = {}

    for key in data_list[0].keys():
        key_dataframes = [data[key] for data in data_list if key in data]
        combined_data[key] = pd.concat(key_dataframes, ignore_index=True)
    
    return combined_data


if __name__ == "__main__":
    print('Preprocessing')
    
    ######################## PREPROCESSING E4PLUS ##########################
    
    root = "C:\\Users\\selaca\\Desktop\\wearables_international-main\\src\\wearables_python\\sample_data\\2024-12-04"    
    e4_avro_files=read_plus.read(root)    
    dataframes=combine_dataframes(e4_avro_files)
    
    ### PREPROCESS EDA ###
    eda_df=dataframes['EDA'].drop(columns=['unix_timestamp']).dropna().reset_index(drop=True)
    eda_metrics_df=preprocessing_eda.process(eda_df)
    
    ### PREPROCESS TEMP ###
    temp_df=dataframes['TEMP'].drop(columns=['unix_timestamp']).dropna().reset_index(drop=True)
    temp_metrics_df=preprocessing_temp.process(temp_df)
    
    ### PREPROCESS ACC ###
    acc_df=dataframes['ACC'].drop(columns=['unix_timestamp']).dropna().reset_index(drop=True)
    acc_metrics_df=preprocessing_acc.process(acc_df)
    
    ### PREPROCESS BVP ###
    bvp_df=dataframes['BVP'].drop(columns=['unix_timestamp']).dropna().reset_index(drop=True)
    hrv_metrics_df=preprocessing_bvp.process(bvp_df)
    
    ######################## PREPROCESSING NOWATCH ##########################
    """
    root = "C:\\Users\\selaca\\Desktop\\wearables_international-main\\src\\wearables_python\\sample_data\\unprocessed_metric_data_Nowatch_20241204_0807_0820.txt"    
    nowatch_files=read_nowatch.read(root)    
    """
