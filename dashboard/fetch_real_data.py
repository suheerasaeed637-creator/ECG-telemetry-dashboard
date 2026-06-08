import wfdb
import pandas as pd
import numpy as np
import os

def fetch_and_save_record(record_name, output_filename, duration_seconds=300):
    """
    Fetches a record from the MIT-BIH Arrhythmia Database and saves it to a CSV.
    We grab the first 'duration_seconds' of data. The sampling rate is 360 Hz.
    """
    print(f"Fetching record {record_name}...")
    try:
        # MIT-BIH sampling rate is 360 Hz
        fs = 360
        samps = duration_seconds * fs
        
        # Download the record
        record = wfdb.rdrecord(record_name, pn_dir='mitdb', sampto=samps)
        
        # Typically the first channel is MLII, which is a good standard ECG lead
        ecg_signal = record.p_signal[:, 0]
        
        # Create a time array
        time = np.arange(len(ecg_signal)) / fs
        
        # Save to CSV
        df = pd.DataFrame({'Time': time, 'ECG': ecg_signal})
        df.to_csv(output_filename, index=False)
        print(f"Successfully saved {record_name} to {output_filename}")
        
    except Exception as e:
        print(f"Failed to fetch record {record_name}: {e}")

if __name__ == "__main__":
    # Ensure we are in the correct directory
    os.makedirs('data', exist_ok=True)
    
    # Let's fetch 3 different records:
    # 100: Normal sinus rhythm
    # 119: Ventricular ectopic beats
    # 200: Ventricular bigeminy and couplets
    records_to_fetch = [
        ('100', 'data/subject_100_normal.csv'),
        ('119', 'data/subject_119_ectopic.csv'),
        ('200', 'data/subject_200_arrhythmia.csv')
    ]
    
    for rec, out_file in records_to_fetch:
        fetch_and_save_record(rec, out_file, duration_seconds=600) # Fetch 10 mins of data
    
    print("Data fetching complete.")
