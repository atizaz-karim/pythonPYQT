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
        entries_per_patient = 5  # 5 history records per patient
        points_per_signal = 1000 
        
        all_rows = []
        mean_ecg = np.mean(raw_ecg)
        
        for i in range(num_unique_patients):
            patient_base = heart_df.iloc[i]
            p_name = f"Patient_{i+1}"
            
            # Base values for realistic variations
            bp_base = float(patient_base['Blood Pressure'])
            chol_base = float(patient_base['Cholesterol Level'])
            bmi_base = float(patient_base['BMI'])
            
            for entry_idx in range(entries_per_patient):
                # 1. Unique Date
                date_recorded = (datetime(2023, 1, 1) + timedelta(days=entry_idx*30)).strftime('%Y-%m-%d')
                
                # 2. ECG Signal (Keep as 1000-point signal)
                start_idx = (i * 500 + entry_idx * 1000) % (len(raw_ecg) - points_per_signal)
                ecg_seg = (raw_ecg[start_idx : start_idx + points_per_signal] - mean_ecg) / 100.0
                ecg_str = ",".join([f"{v:.3f}" for v in ecg_seg])
                
                # 3. Health Metrics (Single numbers for analysis and clean table)
                # We add a small random variation to each visit
                hr_val = float(np.random.randint(65, 85))
                bp_val = bp_base + np.random.uniform(-5, 5)
                chol_val = chol_base + np.random.uniform(-10, 10)
                
                all_rows.append({
                    'Name': p_name,
                    'Gender': patient_base['Gender'],
                    'Age': patient_base['Age'],
                    'Date_Recorded': date_recorded,
                    'ECG Signal': ecg_str,
                    'Heart Rate': round(hr_val, 1),
                    'Blood Pressure': round(bp_val, 1),
                    'Cholesterol Level': round(chol_val, 1),
                    'BMI': round(bmi_base, 2),
                    'Triglyceride Level': patient_base['Triglyceride Level'],
                    'Heart Disease Status': patient_base['Heart Disease Status']
                })
        
        pd.DataFrame(all_rows).to_csv('merged_patient_ecg_data.csv', index=False)
        print("Success! Multi-entry CSV created with single-value metrics.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_patient_history_data()