import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Fill NaN values to ensure the rolling window doesn't break
    temp_series = df_filtered[column].fillna(method='ffill').fillna(method='bfill')
    
    # Calculate rolling mean (centered provides better alignment for time-series)
    df_filtered[f'{column}_smoothed'] = temp_series.rolling(window=window_size, center=True).mean()
    
    # Fill edge cases caused by the window centering
    df_filtered[f'{column}_smoothed'] = df_filtered[f'{column}_smoothed'].fillna(method='ffill').fillna(method='bfill')
    
    return df_filtered



def apply_threshold_filter(df, column, min_val=None, max_val=None):
    """
    Filters data based on specific health thresholds to remove noise or outliers.
    Example: Removing physiological impossibilities or focusing on high-risk zones.
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

# ==========================================
# DATA QUALITY & DESCRIPTIVE STATS
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
    data_info_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Unique Values': df.nunique()
    })
    return data_info_df

def get_descriptive_stats(df):
    """Calculates and returns descriptive statistics for numerical columns."""
    numerical_df = df.select_dtypes(include=np.number)
    return numerical_df.describe()

# ==========================================
# VISUALIZATION & CORRELATION
# ==========================================

def compute_correlations(df):
    """Computes Pearson correlation coefficients between numerical health metrics."""
    numerical_df = df.select_dtypes(include=np.number)
    return numerical_df.corr()



def plot_numerical_distributions(df):
    """Generates and displays histograms for numerical columns."""
    numerical_cols = df.select_dtypes(include=np.number).columns
    if len(numerical_cols) == 0:
        return

    num_features = len(numerical_cols)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols * 5, num_rows * 4))
    for i, col in enumerate(numerical_cols):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(df[col].dropna(), kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df):
    """Generates and displays bar plots for categorical columns."""
    categorical_cols = df.select_dtypes(include='object').columns
    if len(categorical_cols) == 0:
        return

    num_features = len(categorical_cols)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols * 5, num_rows * 4))
    for i, col in enumerate(categorical_cols):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.countplot(x=col, data=df, palette='viridis')
        plt.title(f'Frequency of {col}')
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    plt.show()

# ==========================================
# SIGNAL ANALYSIS (NEW)
# ==========================================

def get_fft_analysis(series):
    """
    Computes FFT for a specific signal series.
    This is a utility function for the spectrum analysis tab.
    """
    data = series.dropna().values
    n = len(data)
    if n == 0:
        return None, None
        
    # Compute Power Spectrum
    freq = np.fft.rfftfreq(n, d=1.0)
    fft_values = np.abs(np.fft.rfft(data))
    
    return freq, fft_values