# ECG‑HRV Telemetry Dashboard

A Python‑based dashboard for real‑time ECG signal acquisition, filtering, and Heart Rate Variability (HRV) analysis.  
This project integrates time‑domain, frequency‑domain, and non‑linear HRV metrics into an interactive telemetry interface.

---

## Overview
This dashboard processes ECG signals to extract HRV parameters and visualize autonomic responses.  
It supports:
- Band‑pass and notch filtering for clean ECG signals  
- R‑peak detection using Pan–Tompkins algorithm  
- HRV computation (SDNN, RMSSD, LF/HF ratio, SD1, SD2)  
- Interactive plots for ECG waveform, RR‑tachogram, power spectrum, and Poincaré map  

---
## Methodology
Signal Acquisition: ECG data from PhysioNet datasets

Filtering: 0.5–40 Hz band‑pass + 50 Hz notch

Feature Extraction: R‑peak detection → RR‑intervals → HRV metrics

Visualization: Streamlit/Dash interface with dynamic plots

---
## Quick Access
Scan the QR code below to open the GitHub repository:
![QR Code](qr.png)

