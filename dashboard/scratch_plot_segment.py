import pandas as pd
import neurokit2 as nk
import numpy as np

df = pd.read_csv("data/subject_100_normal.csv")
fs = 360
ecg_signal = df['ECG'].values
signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs, method='neurokit')
peaks = info["ECG_R_Peaks"]

peaks_in_range = [p for p in peaks if 1700 <= p <= 3100]
print("Detected peaks in range 1700-3100:", peaks_in_range)

# Find local maxima in the raw signal in this range
# We expect peaks around 1809, 2045 (or 2092?), 2403 (or 2375?), 2706 (or 2686?), 2997
print("\nChecking raw signal around the detected peak indices:")
for p in peaks_in_range:
    # Print the raw values in a window of 20 samples around the peak
    segment = ecg_signal[p-10:p+10]
    local_max_idx = p - 10 + np.argmax(segment)
    print(f"  Detected index: {p:<4d} (raw={ecg_signal[p]:.4f}) | Local raw max index: {local_max_idx:<4d} (raw={ecg_signal[local_max_idx]:.4f})")
