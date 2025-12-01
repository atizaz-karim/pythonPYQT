import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Loads the heart_disease.csv dataset into a pandas DataFrame."""
    return pd.read_csv(file_path)

def check_missing_values(df):
    """Calculates and returns missing value counts and percentages for each column."""
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percentage
    })

    return missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

def display_data_info(df):
    """Displays data types and unique value counts for each column in the DataFrame."""
    data_types = df.dtypes
    unique_counts = df.nunique()

    data_info_df = pd.DataFrame({
        'Data Type': data_types,
        'Unique Values': unique_counts
    })
    return data_info_df

def get_descriptive_stats(df):
    """Calculates and returns descriptive statistics for numerical columns in the DataFrame."""
    numerical_df = df.select_dtypes(include=np.number)
    return numerical_df.describe()

def plot_numerical_distributions(df):
    """Generates and displays histograms for numerical columns to visualize their distributions."""
    numerical_cols = df.select_dtypes(include=np.number).columns

    num_features = len(numerical_cols)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols * 5, num_rows * 4))

    for i, col in enumerate(numerical_cols):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df):
    """Generates and displays bar plots for categorical columns to visualize their frequency distributions."""
    categorical_cols = df.select_dtypes(include='object').columns

    num_features = len(categorical_cols)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols * 5, num_rows * 4))

    for i, col in enumerate(categorical_cols):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.countplot(x=col, data=df, palette='viridis', hue=col, legend=False)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
