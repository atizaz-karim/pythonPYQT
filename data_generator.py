import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_patient_history_data():
    try:
        print("Loading source files...")
        heart_df = pd.read_csv('heart_disease.csv')
        ecg_df = pd.read_csv('ecg_1d_timeseries_prediction.csv', sep=';')
        raw_ecg = ecg_df['ecg_value'].values
        
        num_unique_patients = 100
        entries_per_patient = 5  # Each patient will have 5 records on different dates
        points_per_signal = 1000 # 1000 points to ensure correlation analysis works
        
        all_rows = []
        mean_ecg = np.mean(raw_ecg)
        
        print(f"Creating {entries_per_patient} entries for {num_unique_patients} unique patients...")
        
        for i in range(num_unique_patients):
            # Get base data for the patient
            patient_base = heart_df.iloc[i]
            p_name = f"Patient_{i+1}"
            
            # Base values to generate realistic variations over time
            bp_base = float(patient_base['Blood Pressure'])
            chol_base = float(patient_base['Cholesterol Level'])
            bmi_base = float(patient_base['BMI'])
            
            for entry_idx in range(entries_per_patient):
                # 1. Generate a unique Date (e.g., 30 days apart)
                date_recorded = (datetime(2023, 1, 1) + timedelta(days=entry_idx*30)).strftime('%Y-%m-%d')
                
                # 2. ECG Signal (unique slice for every entry)
                start_idx = (i * 500 + entry_idx * 1000) % (len(raw_ecg) - points_per_signal)
                ecg_seg = (raw_ecg[start_idx : start_idx + points_per_signal] - mean_ecg) / 100.0
                ecg_str = ",".join([f"{v:.3f}" for v in ecg_seg])
                
                # 3. Heart Rate Signal (Synthetic wave around a base BPM)
                hr_base = np.random.randint(65, 85)
                hr_sig = hr_base + np.sin(np.linspace(0, 10, points_per_signal)) + np.random.normal(0, 0.5, points_per_signal)
                hr_str = ",".join([f"{v:.1f}" for v in hr_sig])
                
                # 4. Blood Pressure Signal (Base value + variation)
                bp_sig = bp_base + np.random.normal(0, 2.0, points_per_signal)
                bp_str = ",".join([f"{v:.1f}" for v in bp_sig])
                
                # 5. Cholesterol Signal (Base value + variation)
                chol_sig = chol_base + np.random.normal(0, 1.0, points_per_signal)
                chol_str = ",".join([f"{v:.1f}" for v in chol_sig])
                
                # Construct the combined row
                row = {
                    'Name': p_name,
                    'Gender': patient_base['Gender'],
                    'Age': patient_base['Age'],
                    'Date_Recorded': date_recorded,
                    'ECG Signal': ecg_str,
                    'Heart Rate': hr_str,
                    'Blood Pressure': bp_str,
                    'Cholesterol Level': chol_str,
                    'BMI': bmi_base,
                    'Triglyceride Level': patient_base['Triglyceride Level'],
                    'Heart Disease Status': patient_base['Heart Disease Status']
                }
                all_rows.append(row)
        
        final_df = pd.DataFrame(all_rows)
        output_file = 'merged_patient_ecg_data.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\nSuccess! '{output_file}' created with {len(final_df)} records.")
        print("Each patient now has a history of 5 entries with full signal data.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_patient_history_data()