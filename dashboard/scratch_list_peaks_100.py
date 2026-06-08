import pandas as pd
import neurokit2 as nk
import numpy as np

df = pd.read_csv("data/subject_100_normal.csv")
fs = 360
ecg_signal = df['ECG'].values

ecg_fixed, is_inverted = nk.ecg_invert(ecg_signal, sampling_rate=fs)
signals, info = nk.ecg_process(ecg_fixed, sampling_rate=fs, method='neurokit')
clean_ecg = signals['ECG_Clean'].values

peaks = info["ECG_R_Peaks"]
print("First 20 peak indices and their values:")
for i in range(min(20, len(peaks))):
    p = peaks[i]
    print(f"  Peak {i:2d}: index = {p:<6d} | value = {clean_ecg[p]:8.4f}")
