import pandas as pd
import neurokit2 as nk
import numpy as np

df = pd.read_csv("data/subject_200_arrhythmia.csv")
fs = 360
ecg_signal = df['ECG'].values

# original process (without manual invert check)
signals_orig, info_orig = nk.ecg_process(ecg_signal, sampling_rate=fs, method='neurokit')
clean_orig = signals_orig['ECG_Clean'].values

# fixed process
ecg_fixed, is_inverted = nk.ecg_invert(ecg_signal, sampling_rate=fs)
signals_fixed, info_fixed = nk.ecg_process(ecg_fixed, sampling_rate=fs, method='neurokit')
clean_fixed = signals_fixed['ECG_Clean'].values

print("Is Inverted according to ecg_invert:", is_inverted)
print("Original Raw Signal range:", np.min(ecg_signal), np.max(ecg_signal))
print("Fixed Raw Signal range:", np.min(ecg_fixed), np.max(ecg_fixed))

print("\n--- Original peaks info ---")
for i in range(5):
    p = info_orig["ECG_R_Peaks"][i]
    print(f"Peak {i} at {p}: value in clean_orig = {clean_orig[p]:.4f}")

print("\n--- Fixed peaks info ---")
for i in range(5):
    p = info_fixed["ECG_R_Peaks"][i]
    print(f"Peak {i} at {p}: value in clean_fixed = {clean_fixed[p]:.4f}")
