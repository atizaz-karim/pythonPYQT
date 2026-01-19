import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk
from scipy.signal import find_peaks
from scipy.fft import fft, ifft

def load_data(file_path):
    """Loads the heart_disease.csv dataset into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

# ==========================================
# FILTERING & SIGNAL PROCESSING
# ==========================================

def apply_moving_average(df, column, window_size=5):
    """
    Applies a simple moving average to smooth out noise in health metrics.
    Useful for visualizing trends in heart rate or blood pressure.
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found for Moving Average.")
        return df
    
    df_filtered = df.copy()
    temp_series = df_filtered[column].fillna(method='ffill').fillna(method='bfill')
    df_filtered[f'{column}_smoothed'] = temp_series.rolling(window=window_size, center=True).mean()
    df_filtered[f'{column}_smoothed'] = df_filtered[f'{column}_smoothed'].fillna(method='ffill').fillna(method='bfill')
    
    return df_filtered

def apply_threshold_filter(df, column, min_val=None, max_val=None):
    """
    Filters data based on specific health thresholds to remove noise or outliers.
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found for Threshold Filtering.")
        return df
    
    filtered_df = df.copy()
    if min_val is not None:
        filtered_df = filtered_df[filtered_df[column] >= min_val]
    if max_val is not None:
        filtered_df = filtered_df[filtered_df[column] <= max_val]
        
    return filtered_df

def fft_denoise_signal(data, threshold_percent=0.1):
    """
    NEW: Cleans a signal by zeroing out low-amplitude noise in the frequency domain.
    This is used for the 'Cleaned Signal' visualization in the GUI.
    """
    if data is None or len(data) == 0:
        return data

    n = len(data)
    f_hat = fft(data) # Move to frequency domain
    psd = np.abs(f_hat) / n # Power spectrum
    
    limit = np.max(psd) * threshold_percent
    indices = psd > limit # Mask for significant frequencies
    f_hat_clean = f_hat * indices
    
    return np.real(ifft(f_hat_clean)) # Back to time domain

# ==========================================
# ADVANCED ECG & EEG ANALYSIS (UPDATED)
# ==========================================

def analyze_ecg_signal(signal_data, sampling_rate=1000):
    """
    UPDATED: Processes ECG signal using NeuroKit2 (cleaning/peaks) 
    and SciPy (peak verification).
    """
    # High-level pipeline: cleaning, R-peak detection, and HR calculation
    signals, info = nk.ecg_process(signal_data, sampling_rate=sampling_rate)
    
    # Manual peak detection for verification/custom logic
    peaks, _ = find_peaks(signal_data, distance=sampling_rate*0.6, height=np.mean(signal_data))
    
    return signals, info, peaks

def analyze_eeg_signal(signal_data, sampling_rate=1000):
    """
    UPDATED: Cleans EEG signal and extracts brainwave frequency bands (Alpha, Beta, etc.).
    """
    eeg_cleaned = nk.eeg_clean(signal_data, sampling_rate=sampling_rate)
    
    # Extract Power Spectral Density for bands
    bands = nk.eeg_power(eeg_cleaned, sampling_rate=sampling_rate)
    
    return eeg_cleaned, bands

# ==========================================
# DATA QUALITY & DESCRIPTIVE STATS (ORIGINAL)
# ==========================================

def check_missing_values(df):
    """Calculates and returns missing value counts and percentages."""
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percentage
    })
    return missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

def display_data_info(df):
    """Displays data types and unique value counts for each column."""
    return pd.DataFrame({
        'Data Type': df.dtypes,
        'Unique Values': df.nunique()
    })

def get_descriptive_stats(df):
    """Calculates and returns descriptive statistics for numerical columns."""
    return df.select_dtypes(include=np.number).describe()

# ==========================================
# VISUALIZATION & CORRELATION (ORIGINAL)
# ==========================================

def compute_correlations(df):
    """Computes Pearson correlation coefficients between numerical metrics."""
    return df.select_dtypes(include=np.number).corr()

def get_fft_analysis(series):
    """Computes FFT for a specific signal series for spectrum analysis."""
    data = series.dropna().values
    n = len(data)
    if n == 0: return None, None
    freq = np.fft.rfftfreq(n, d=1.0)
    fft_values = np.abs(np.fft.rfft(data))
    return freq, fft_values

def plot_numerical_distributions(df):
    """Generates and displays histograms for numerical columns."""
    numerical_cols = df.select_dtypes(include=np.number).columns
    if len(numerical_cols) == 0: return
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols):
        plt.subplot(len(numerical_cols)//3 + 1, 3, i + 1)
        sns.histplot(df[col].dropna(), kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df):
    """Generates and displays bar plots for categorical columns."""
    categorical_cols = df.select_dtypes(include='object').columns
    if len(categorical_cols) == 0: return
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_cols):
        plt.subplot(len(categorical_cols)//3 + 1, 3, i + 1)
        sns.countplot(x=col, data=df, palette='viridis')
        plt.title(f'Frequency of {col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()