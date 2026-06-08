import pandas as pd
import neurokit2 as nk
import numpy as np

# Load normal record (subject 100)
df = pd.read_csv("data/subject_100_normal.csv")
fs = 360
ecg_signal = df['ECG'].values

# Run process
ecg_fixed, is_inverted = nk.ecg_invert(ecg_signal, sampling_rate=fs)
signals, info = nk.ecg_process(ecg_fixed, sampling_rate=fs, method='neurokit')
clean_ecg = signals['ECG_Clean'].values

# Let's inspect the first 3 peaks
peaks = info["ECG_R_Peaks"]
window = int(0.05 * fs) # 18 samples for 360 Hz

print("Window size:", window)
for i in range(3):
    p = peaks[i]
    start = max(0, p - window)
    end = min(len(clean_ecg), p + window)
    
    segment = clean_ecg[start:end]
    p_pos = start + np.argmax(segment)
    p_abs = start + np.argmax(np.abs(segment))
    
    print(f"\nPeak {i}: original detected index = {p}")
    print(f"  Snapping with max positive: index = {p_pos}, value = {clean_ecg[p_pos]:.4f}")
    print(f"  Snapping with max absolute: index = {p_abs}, value = {clean_ecg[p_abs]:.4f}")
    print("  Segment values around peak:")
    for offset, val in enumerate(segment):
        idx = start + offset
        marker = ""
        if idx == p: marker += " [orig]"
        if idx == p_pos: marker += " [pos]"
        if idx == p_abs: marker += " [abs]"
        print(f"    Index {idx:4d}: {val:8.4f}{marker}")
