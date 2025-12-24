from PyQt5.QtCore import QThread, pyqtSignal
import sqlite3
import pandas as pd

class InsertDataThread(QThread):
    progress = pyqtSignal(str)

    def __init__(self, db_path, df_source):
        super().__init__()
        self.db_path = db_path
        self.df_source = df_source

    def run(self):
        self.progress.emit("Inserting data into database...")
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            selected_columns = [
                'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
                'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level', 'Heart Disease Status'
            ]
            df = self.df_source[selected_columns].copy()
            df.rename(columns={
                'Blood Pressure': 'Blood_Pressure',
                'Cholesterol Level': 'Cholesterol_Level',
                'Sleep Hours': 'Sleep_Hours',
                'Triglyceride Level': 'Triglyceride_Level',
                'Fasting Blood Sugar': 'Fasting_Blood_Sugar',
                'CRP Level': 'CRP_Level',
                'Homocysteine Level': 'Homocysteine_Level',
                'Heart Disease Status': 'Heart_Disease_Status'
            }, inplace=True)

            for col in ['Age', 'Blood_Pressure', 'Cholesterol_Level', 'BMI', 'Sleep_Hours',
                        'Triglyceride_Level', 'Fasting_Blood_Sugar', 'CRP_Level', 'Homocysteine_Level']:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)

            for col in ['Gender', 'Heart_Disease_Status']:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)

            data_to_insert = [tuple(row) for row in df.itertuples(index=False)]

            columns_str = ', '.join(df.columns)
            placeholders = ', '.join(['?' for _ in df.columns])
            insert_sql = f"INSERT INTO patient_health_metrics ({columns_str}) VALUES ({placeholders})"

            cursor.executemany(insert_sql, data_to_insert)
            conn.commit()
            cursor.close()
            conn.close()

            self.progress.emit(f"Successfully inserted {len(data_to_insert)} rows into patient_health_metrics!")

        except Exception as e:
            self.progress.emit(f"Error inserting data: {e}")
