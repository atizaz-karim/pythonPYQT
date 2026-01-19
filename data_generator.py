import pandas as pd
import io
import re

def get_clean_ecg_data(filename):
    with open(filename, 'r', encoding='latin-1') as f:
        text = f.read()
    
    # Extract CSV part from RTF
    # It seems to be a list of lines. We look for the header.
    match = re.search(r'report_id,Name,.*', text)
    if not match:
        return None
    
    csv_part = text[match.start():]
    # Clean up RTF formatting characters like \ and } and \n
    csv_part = csv_part.replace('\\\n', '')
    csv_part = csv_part.replace('\\', '')
    csv_part = re.sub(r'\}$', '', csv_part.strip())
    
    df = pd.read_csv(io.StringIO(csv_part))
    return df

ecg_df = get_clean_ecg_data('ecg.csv')
print("ECG Data Columns:", ecg_df.columns.tolist())
print("ECG Data Head:\n", ecg_df)

# Features Raw
features_df = pd.read_csv('features_raw.csv')
print("\nFeatures Data Shape:", features_df.shape)