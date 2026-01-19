import pandas as pd
from datetime import datetime
import os

def create_data_file():
    all_records = []
    
    # 1. Get EEG segments from the Seizure Dataset
    if os.path.exists('Epileptic Seizure Recognition Data Set.csv'):
        df_seizure = pd.read_csv('Epileptic Seizure Recognition Data Set.csv')
        for _, row in df_seizure.iterrows():
            # Merging 178 points into one string for your GUI
            eeg_str = ",".join([str(row[f'X{i}']) for i in range(1, 179)])
            all_records.append({
                'patient id': str(row['ID']),
                'ecg': '0', 
                'eeg': eeg_str,
                'date recorded': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    # 2. Get the long signal from Features Raw (Patient 99)
    if os.path.exists('features_raw.csv'):
        df_feat = pd.read_csv('features_raw.csv')
        long_signal = ",".join(df_feat['Fp1'].astype(str).tolist())
        all_records.append({
            'patient id': '99',
            'ecg': '0',
            'eeg': long_signal,
            'date recorded': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    # 3. SAVE TO FILE (This creates the actual CSV)
    if all_records:
        output_df = pd.DataFrame(all_records)
        # Force the columns to the 4 you requested
        output_df = output_df[['patient id', 'ecg', 'eeg', 'date recorded']]
        output_df.to_csv('data.csv', index=False)
        print("Done! Check your folder for data.csv")
    else:
        print("Error: No files found to process.")

if __name__ == "__main__":
    create_data_file()
    