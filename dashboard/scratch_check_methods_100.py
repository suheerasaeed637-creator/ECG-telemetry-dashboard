import pandas as pd
import neurokit2 as nk
import os

df = pd.read_csv("data/subject_100_normal.csv")
fs = 360
ecg_signal = df['ECG'].values

methods = ['neurokit', 'pantompkins1985', 'hamilton2002', 'elgendi2010', 'kalidas2016']

for method in methods:
    try:
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs, method=method)
        peaks = info["ECG_R_Peaks"]
        print(f"\nMethod: {method}")
        for i in range(10):
            p = peaks[i]
            print(f"  Peak {i}: index = {p:<6d} | raw = {ecg_signal[p]:8.4f} | clean = {signals['ECG_Clean'].values[p]:8.4f}")
    except Exception as e:
        print(f"Method {method} failed: {e}")
