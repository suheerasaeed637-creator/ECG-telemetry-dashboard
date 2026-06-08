import wfdb
import numpy as np
import neurokit2 as nk
import pandas as pd

# Load first 10 seconds (10 * 360 = 3600 samples)
fs = 360
samps = 10 * fs

record = wfdb.rdrecord('100', pn_dir='mitdb', sampto=samps)
ann = wfdb.rdann('100', 'atr', pn_dir='mitdb', sampto=samps)

true_peaks = ann.sample
print("True (official) peaks from database:")
for idx, p in enumerate(true_peaks):
    print(f"  Peak {idx}: index = {p:<6d} | raw = {record.p_signal[p, 0]:8.4f}")

# Process with neurokit2
ecg_signal = record.p_signal[:, 0]
signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs, method='neurokit')
nk_peaks = info["ECG_R_Peaks"]

print("\nNeuroKit2 detected peaks:")
for idx, p in enumerate(nk_peaks):
    if idx < len(true_peaks):
        print(f"  Peak {idx}: index = {p:<6d} | raw = {ecg_signal[p]:8.4f}")
