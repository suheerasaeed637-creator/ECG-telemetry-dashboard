import pandas as pd
import numpy as np

df = pd.read_csv("data/subject_100_normal.csv")
raw = df['ECG'].values

# Let's see the maximum value in segments of 100 samples from index 1800 to 3200
for start in range(1800, 3200, 100):
    end = start + 100
    seg = raw[start:end]
    print(f"Segment [{start}:{end}] | min: {np.min(seg):8.4f} | max: {np.max(seg):8.4f} | argmax_index: {start + np.argmax(seg)}")
