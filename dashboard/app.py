import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import neurokit2 as nk
import scipy.signal as signal
import os

# --- Configuration & Styling ---
st.set_page_config(layout="wide", page_title="ECG Telemetry Dashboard", page_icon="🫀")

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    .metric-container {
        background: rgba(30, 34, 45, 0.6);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    
    .dataframe {
        font-size: 14px;
        color: #b0c4de !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df

import json

def get_record_metadata(file_path):
    base, _ = os.path.splitext(file_path)
    meta_path = base + "_meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"fs": 360}  # default to 360 Hz if no metadata exists

def save_record_metadata(file_path, metadata):
    base, _ = os.path.splitext(file_path)
    meta_path = base + "_meta.json"
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        return True
    except Exception:
        return False

# --- Signal Processing ---
@st.cache_data
def process_ecg(ecg_signal, fs=360):
    # Automatically check and correct for signal inversion (leads placed backwards)
    ecg_fixed, is_inverted = nk.ecg_invert(ecg_signal, sampling_rate=fs)
    
    # Clean ECG and find peaks using the robust NeuroKit algorithm (highly reliable)
    signals, info = nk.ecg_process(ecg_fixed, sampling_rate=fs, method='neurokit')
    
    # Snap R-peaks to the absolute local maximum/minimum (largest deflection) to ensure perfect visual alignment
    clean_ecg = signals['ECG_Clean'].values
    window = int(0.05 * fs) # 50 ms window
    corrected_peaks = []
    for p in info["ECG_R_Peaks"]:
        start = max(0, p - window)
        end = min(len(clean_ecg), p + window)
        if end > start:
            # Snap to the point with the largest absolute deviation from baseline
            local_max_idx = np.argmax(np.abs(clean_ecg[start:end]))
            corrected_peaks.append(start + local_max_idx)
    
    info["ECG_R_Peaks"] = np.array(corrected_peaks)
    
    # Calculate global HRV
    hrv_indices = nk.hrv(info, sampling_rate=fs)
    
    # Extract RR intervals in ms
    r_peaks = info["ECG_R_Peaks"]
    r_peaks_time = r_peaks / fs
    rr_intervals = np.diff(r_peaks_time) * 1000
    
    return signals, info, hrv_indices, rr_intervals, r_peaks_time[1:]

def calculate_segmentwise_hrv(rr_intervals, r_peaks_time, window_size_beats=50, step=10):
    segments = []
    times = []
    
    for i in range(0, len(rr_intervals) - window_size_beats, step):
        window = rr_intervals[i:i+window_size_beats]
        
        # Calculate time-domain metrics
        sdnn = np.std(window, ddof=1)
        rmssd = np.sqrt(np.mean(np.square(np.diff(window))))
        hr = 60000 / np.mean(window)
        
        segments.append({"SDNN": sdnn, "RMSSD": rmssd, "HR": hr})
        # Use the middle time of the window
        times.append(r_peaks_time[i + window_size_beats // 2])
        
    return pd.DataFrame(segments), times

def calculate_psd(rr_intervals, fs=4.0):
    # Interpolate RR intervals to an evenly spaced time series for PSD
    time = np.cumsum(rr_intervals) / 1000.0
    time = time - time[0]
    
    even_time = np.arange(0, time[-1], 1.0/fs)
    even_rr = np.interp(even_time, time, rr_intervals)
    
    # Detrend
    even_rr = signal.detrend(even_rr)
    
    # Welch's method
    freqs, psd = signal.welch(even_rr, fs=fs, nperseg=256, noverlap=128)
    return freqs, psd

# --- Dashboard Layout ---
st.title("The Telemetry Dashboard")

# Sidebar
st.sidebar.header("Data Selection")
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# --- Upload Custom Dataset Expander ---
with st.sidebar.expander("📤 Add Custom Dataset", expanded=False):
    uploaded_file = st.file_uploader("Upload ECG CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            # Read first few lines to preview
            uploaded_file.seek(0)
            preview_df = pd.read_csv(uploaded_file, nrows=5)
            st.markdown("**Preview of uploaded columns:**")
            st.dataframe(preview_df.head(3), use_container_width=True)
            
            # Select columns
            cols = list(preview_df.columns)
            ecg_col_default = 0
            # Try to find a column with "ecg", "signal", "lead"
            for i, col in enumerate(cols):
                if any(x in col.lower() for x in ["ecg", "signal", "lead", "mlii"]):
                    ecg_col_default = i
                    break
            ecg_col = st.selectbox("ECG Column", cols, index=ecg_col_default)
            
            time_option = st.selectbox("Time Column", ["Generate from Sampling Rate", "Select Column"])
            time_col = None
            if time_option == "Select Column":
                time_col_default = 0
                for i, col in enumerate(cols):
                    if "time" in col.lower():
                        time_col_default = i
                        break
                time_col = st.selectbox("Time Column Name", cols, index=time_col_default)
            
            # Estimate fs if Time column is selected
            estimated_fs = 360
            if time_col is not None:
                try:
                    uploaded_file.seek(0)
                    uploaded_full = pd.read_csv(uploaded_file, usecols=[time_col])
                    diffs = np.diff(uploaded_full[time_col].dropna().values)
                    mean_diff = np.mean(diffs)
                    if mean_diff > 0:
                        estimated_fs = int(round(1.0 / mean_diff))
                except Exception:
                    pass
            
            fs_input = st.number_input("Sampling Rate (Hz)", min_value=10, max_value=10000, value=estimated_fs, step=1)
            
            # Record Name
            default_name = os.path.splitext(uploaded_file.name)[0]
            record_name = st.text_input("Dataset Record Name", value=default_name)
            
            if st.button("Save & Load Dataset"):
                if not record_name.strip():
                    st.error("Please enter a valid record name.")
                else:
                    # Sanitize record name
                    sanitized_name = "".join([c if c.isalnum() or c in ['-', '_'] else '_' for c in record_name.strip()])
                    dest_csv = os.path.join(data_dir, f"{sanitized_name}.csv")
                    
                    # Read full data
                    uploaded_file.seek(0)
                    full_df = pd.read_csv(uploaded_file)
                    
                    if ecg_col not in full_df.columns:
                        st.error(f"ECG column '{ecg_col}' not found in file.")
                    elif time_col is not None and time_col not in full_df.columns:
                        st.error(f"Time column '{time_col}' not found in file.")
                    else:
                        with st.spinner("Saving dataset..."):
                            # Prepare standard dataframe
                            clean_df = pd.DataFrame()
                            if time_col is not None:
                                clean_df['Time'] = full_df[time_col]
                            else:
                                clean_df['Time'] = np.arange(len(full_df)) / fs_input
                            
                            clean_df['ECG'] = full_df[ecg_col]
                            
                            # Save CSV
                            clean_df.to_csv(dest_csv, index=False)
                            
                            # Save metadata
                            meta_data = {"fs": fs_input}
                            save_record_metadata(dest_csv, meta_data)
                            
                            # Set as active selection in session state
                            filename = f"{sanitized_name}.csv"
                            st.session_state['selected_file'] = filename
                            st.success(f"Saved {filename} successfully!")
                            st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Load list of available datasets (excluding meta JSON files)
available_files = []
if os.path.exists(data_dir):
    available_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

if not available_files:
    st.sidebar.warning("No data found. Please run `fetch_real_data.py` first, or upload a dataset above.")
    st.stop()

# Auto-select newly saved dataset from session state if available
default_index = 0
if 'selected_file' in st.session_state and st.session_state['selected_file'] in available_files:
    default_index = available_files.index(st.session_state['selected_file'])

selected_file = st.sidebar.selectbox("Select ECG Record", available_files, index=default_index)
df = load_data(os.path.join(data_dir, selected_file))

if df is None:
    st.error("Failed to load data.")
    st.stop()

# Retrieve metadata sampling rate
metadata = get_record_metadata(os.path.join(data_dir, selected_file))
fs = metadata.get("fs", 360)

st.sidebar.info(f"⚡ **Sampling Rate:** {fs} Hz")

with st.spinner("Processing ECG and calculating HRV metrics..."):
    signals, info, hrv_indices, rr_intervals, rr_times = process_ecg(df['ECG'].values, fs=fs)

# Top Row: ECG Waveform
st.markdown("### ECG Waveform Viewer")
fig_ecg = go.Figure()

# Show 10 seconds to mimic a standard ECG strip
view_len = min(10 * fs, len(df))
time_view = df['Time'][:view_len]
ecg_raw_view = signals['ECG_Raw'][:view_len].values
ecg_clean_view = signals['ECG_Clean'][:view_len].values

# Plot the Raw ECG trace faintly
fig_ecg.add_trace(go.Scatter(x=time_view, y=ecg_raw_view, mode='lines', name='Raw ECG', line=dict(color='rgba(136, 132, 216, 0.4)', width=1)))

# Plot the Cleaned ECG trace
fig_ecg.add_trace(go.Scatter(x=time_view, y=ecg_clean_view, mode='lines', name='Cleaned ECG', line=dict(color='#00C49F', width=2)))

# Add R-peaks
r_peaks_in_view = [p for p in info["ECG_R_Peaks"] if p < view_len]
fig_ecg.add_trace(go.Scatter(
    x=[df['Time'][p] for p in r_peaks_in_view], 
    y=[signals['ECG_Clean'][p] for p in r_peaks_in_view],
    mode='markers', name='R-Peaks', marker=dict(color='#FF4B4B', size=10, symbol='circle', line=dict(color='white', width=1))
))

fig_ecg.update_layout(
    height=350, margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Time (s)"), 
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Amplitude"),
    font=dict(color='#e0e6ed'),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_ecg, use_container_width=True)


# Middle Row: Metrics, Poincaré, PSD
col1, col2, col3 = st.columns([1.2, 1.5, 1.5])

with col1:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("#### HRV Measures")
    
    # Format HRV indices table
    if not hrv_indices.empty:
        metrics = {
            "Mean RR (ms)": hrv_indices['HRV_MeanNN'].values[0],
            "HR (bpm)": 60000 / hrv_indices['HRV_MeanNN'].values[0],
            "SDNN (ms)": hrv_indices['HRV_SDNN'].values[0],
            "RMSSD (ms)": hrv_indices['HRV_RMSSD'].values[0],
            "pNN50 (%)": hrv_indices['HRV_pNN50'].values[0],
            "LF (ms²)": hrv_indices['HRV_LF'].values[0],
            "HF (ms²)": hrv_indices['HRV_HF'].values[0],
            "LF/HF ratio": hrv_indices['HRV_LFHF'].values[0],
            "SD1 (ms)": hrv_indices['HRV_SD1'].values[0],
            "SD2 (ms)": hrv_indices['HRV_SD2'].values[0]
        }
        
        # Add sample entropy if available
        if 'HRV_SampEn' in hrv_indices.columns:
            metrics["SampEn"] = hrv_indices['HRV_SampEn'].values[0]
            
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        metrics_df['Value'] = metrics_df['Value'].map('{:.2f}'.format)
        
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    else:
        st.write("Not enough data to compute global HRV.")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("#### Global Return Map (RR Poincare)")
    
    if len(rr_intervals) > 1:
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        
        fig_poincare = px.scatter(
            x=rr_n, y=rr_n1, 
            labels={'x': 'RR_n (ms)', 'y': 'RR_n+1 (ms)'},
            marginal_x="histogram", marginal_y="histogram",
            color_discrete_sequence=['#00C49F']
        )
        fig_poincare.update_traces(marker=dict(size=4, opacity=0.6), name="RR Intervals", showlegend=True, selector=dict(type='scatter'))
        
        # Add Poincaré fitting ellipse (SD1 / SD2)
        try:
            if not hrv_indices.empty:
                sd1 = hrv_indices['HRV_SD1'].values[0]
                sd2 = hrv_indices['HRV_SD2'].values[0]
                mean_rr = hrv_indices['HRV_MeanNN'].values[0]
                
                # Generate points for the 45-degree rotated ellipse
                theta = np.linspace(0, 2 * np.pi, 200)
                x_ell = (sd2 * np.cos(theta) - sd1 * np.sin(theta)) / np.sqrt(2) + mean_rr
                y_ell = (sd2 * np.cos(theta) + sd1 * np.sin(theta)) / np.sqrt(2) + mean_rr
                
                # Add ellipse trace
                fig_poincare.add_trace(go.Scatter(
                    x=x_ell, y=y_ell,
                    mode='lines',
                    name='HRV Ellipse (SD1/SD2)',
                    line=dict(color='#FFBB28', width=2, dash='dash'),
                    showlegend=True
                ))
                
                # Add Identity Line (y = x)
                min_val = min(np.min(rr_n), np.min(rr_n1))
                max_val = max(np.max(rr_n), np.max(rr_n1))
                fig_poincare.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    name='Identity Line (y=x)',
                    line=dict(color='rgba(255, 255, 255, 0.2)', width=1.5, dash='dot'),
                    showlegend=True
                ))
        except Exception:
            pass

        fig_poincare.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e6ed'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_poincare, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("#### Spectrum of RR Tachogram")
    
    if len(rr_intervals) > 60: # Need enough data for welch
        freqs, psd = calculate_psd(rr_intervals)
        
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', line=dict(color='#0088FE', width=2), showlegend=False))
        
        # Shade LF (0.04 - 0.15 Hz) and HF (0.15 - 0.4 Hz)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
        
        fig_psd.add_trace(go.Scatter(x=freqs[lf_mask], y=psd[lf_mask], fill='tozeroy', mode='none', fillcolor='rgba(0, 196, 159, 0.4)', name='LF'))
        fig_psd.add_trace(go.Scatter(x=freqs[hf_mask], y=psd[hf_mask], fill='tozeroy', mode='none', fillcolor='rgba(255, 170, 0, 0.4)', name='HF'))
        
        fig_psd.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title='Frequency (Hz)', range=[0, 0.5]), yaxis=dict(title='Power (ms²/Hz)'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e6ed'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_psd, use_container_width=True)
    else:
        st.write("Not enough data to calculate PSD.")
    st.markdown("</div>", unsafe_allow_html=True)

# Bottom Row: Global Phase Space & Segmentwise
bcol1, bcol2 = st.columns([1, 1.5])

with bcol1:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("### Global ECG Return Map (3D Phase Space)")
    
    # Use first 10 seconds of cleaned ECG for phase space reconstruction
    tau = int(fs * 0.04) # 40ms delay
    ecg_ps = signals['ECG_Clean'].values[:10*fs]
    
    if len(ecg_ps) > 2*tau:
        x_ps = ecg_ps[:-2*tau]
        y_ps = ecg_ps[tau:-tau]
        z_ps = ecg_ps[2*tau:]
        
        fig_phase = go.Figure(data=[go.Scatter3d(
            x=x_ps, y=y_ps, z=z_ps,
            mode='lines',
            line=dict(
                color=np.arange(len(x_ps)),
                colorscale='Viridis',
                width=3
            ),
            opacity=0.8
        )])
        
        fig_phase.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(
                xaxis_title='ECG(t)',
                yaxis_title='ECG(t+τ)',
                zaxis_title='ECG(t+2τ)',
                xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
                yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
                zaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e6ed')
        )
        st.plotly_chart(fig_phase, use_container_width=True)
    else:
        st.write("Not enough data to compute 3D phase space.")
    st.markdown("</div>", unsafe_allow_html=True)

with bcol2:
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("### Segmentwise HRV analysis")
    if len(rr_intervals) > 100:
        seg_df, seg_times = calculate_segmentwise_hrv(rr_intervals, rr_times)
        
        fig_seg = go.Figure()
        
        # Scale variables for visual comparison on same plot
        fig_seg.add_trace(go.Scatter(x=seg_times, y=seg_df['HR'], mode='lines', name='HR (bpm)', line=dict(color='#FF8042')))
        fig_seg.add_trace(go.Scatter(x=seg_times, y=seg_df['SDNN'], mode='lines', name='SDNN (ms)', line=dict(color='#00C49F')))
        fig_seg.add_trace(go.Scatter(x=seg_times, y=seg_df['RMSSD'], mode='lines', name='RMSSD (ms)', line=dict(color='#FFBB28')))
        
        fig_seg.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title='Time (s)'), yaxis=dict(title='Value'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e6ed'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_seg, use_container_width=True)
    else:
        st.write("Not enough data for segmentwise analysis.")
    st.markdown("</div>", unsafe_allow_html=True)


