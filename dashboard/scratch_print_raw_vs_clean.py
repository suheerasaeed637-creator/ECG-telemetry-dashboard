import pandas as pd
import neurokit2 as nk

df = pd.read_csv("data/subject_100_normal.csv")
fs = 360
ecg_signal = df['ECG'].values

signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs, method='neurokit')
clean_ecg = signals['ECG_Clean'].values

print("Indices 2030 to 2060:")
for idx in range(2030, 2060):
    print(f"  Index {idx:4d}: raw = {ecg_signal[idx]:8.4f} | clean = {clean_ecg[idx]:8.4f}")
