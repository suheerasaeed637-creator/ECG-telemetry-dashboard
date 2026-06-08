import pandas as pd
import neurokit2 as nk
import numpy as np
import os

records = [
    ("data/subject_100_normal.csv", "100_normal"),
    ("data/subject_119_ectopic.csv", "119_ectopic"),
    ("data/subject_200_arrhythmia.csv", "200_arrhythmia")
]

for path, name in records:
    if os.path.exists(path):
        df = pd.read_csv(path)
        fs = 360
        ecg_signal = df['ECG'].values
        
        ecg_fixed, is_inverted = nk.ecg_invert(ecg_signal, sampling_rate=fs)
        signals, info = nk.ecg_process(ecg_fixed, sampling_rate=fs, method='neurokit')
        clean_ecg = signals['ECG_Clean'].values
        
        peaks = info["ECG_R_Peaks"]
        window = int(0.05 * fs)
        
        snapped_to_neg = 0
        for p in peaks:
            start = max(0, p - window)
            end = min(len(clean_ecg), p + window)
            segment = clean_ecg[start:end]
            p_abs = start + np.argmax(np.abs(segment))
            
            # Check if the snapped point has a negative value
            if clean_ecg[p_abs] < 0:
                snapped_to_neg += 1
                
        print(f"Record: {name:<20} | Total peaks: {len(peaks):<4} | Snapped to negative value: {snapped_to_neg}")
