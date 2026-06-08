import pandas as pd
import neurokit2 as nk
import numpy as np
import plotly.graph_objects as go
import os

df = pd.read_csv("data/subject_100_normal.csv")
fs = 360
signals, info = nk.ecg_process(df['ECG'].values, sampling_rate=fs, method='pantompkins1985')
clean_ecg = signals['ECG_Clean'].values

# Let's see the first 5 peaks
print("First 5 original peaks from Pan-Tompkins:", info["ECG_R_Peaks"][:5])

window = int(0.05 * fs)
corrected_peaks_abs = []
for p in info["ECG_R_Peaks"]:
    start = max(0, p - window)
    end = min(len(clean_ecg), p + window)
    if end > start:
        local_max_idx = np.argmax(np.abs(clean_ecg[start:end]))
        corrected_peaks_abs.append(start + local_max_idx)

print("First 5 absolute snapped peaks:", corrected_peaks_abs[:5])

# Also check for subject 119 (which has PVCs)
if os.path.exists("data/subject_119_ectopic.csv"):
    df_119 = pd.read_csv("data/subject_119_ectopic.csv")
    signals_119, info_119 = nk.ecg_process(df_119['ECG'].values, sampling_rate=fs, method='pantompkins1985')
    clean_ecg_119 = signals_119['ECG_Clean'].values
    print("Subject 119 - First 5 original peaks:", info_119["ECG_R_Peaks"][:5])
    
    corrected_peaks_abs_119 = []
    for p in info_119["ECG_R_Peaks"]:
        start = max(0, p - window)
        end = min(len(clean_ecg_119), p + window)
        if end > start:
            local_max_idx = np.argmax(np.abs(clean_ecg_119[start:end]))
            corrected_peaks_abs_119.append(start + local_max_idx)
    print("Subject 119 - First 5 absolute snapped peaks:", corrected_peaks_abs_119[:5])
