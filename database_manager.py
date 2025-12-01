import sqlite3
import pandas as pd
import numpy as np

class DatabaseManager:
    def __init__(self, db_name='health_metrics.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        print(f"Successfully connected to {db_name} and created a cursor object.")

    def create_tables(self):
        create_patient_metrics_table = '''
        CREATE TABLE IF NOT EXISTS patient_health_metrics (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            Age REAL,
            Gender TEXT,
            Blood_Pressure REAL,
            Cholesterol_Level REAL,
            BMI REAL,
            Sleep_Hours REAL,
            Triglyceride_Level REAL,
            Fasting_Blood_Sugar REAL,
            CRP_Level REAL,
            Homocysteine_Level REAL,
            Heart_Disease_Status TEXT
        );
        '''

        create_medical_image_table = '''
        CREATE TABLE IF NOT EXISTS medical_image_metadata (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            image_type TEXT,
            capture_date TEXT,
            FOREIGN KEY (patient_id) REFERENCES patient_health_metrics(patient_id)
        );
        '''
        self.cursor.execute(create_patient_metrics_table)
        self.cursor.execute(create_medical_image_table)
        self.conn.commit()
        print("Tables 'patient_health_metrics' and 'medical_image_metadata' created successfully.")

    def insert_patient_data(self, df_source):
        selected_columns = [
            'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
            'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level', 'Heart Disease Status'
        ]

        patient_data_df = df_source[selected_columns].copy()

        patient_data_df.rename(columns={
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
            if patient_data_df[col].isnull().any():
                median_val = patient_data_df[col].median()
                patient_data_df[col] = patient_data_df[col].fillna(median_val)

        for col in ['Gender', 'Heart_Disease_Status']:
            if patient_data_df[col].isnull().any():
                mode_val = patient_data_df[col].mode()[0]
                patient_data_df[col] = patient_data_df[col].fillna(mode_val)

        patient_data_for_db = [tuple(row) for row in patient_data_df.itertuples(index=False)]

        columns_str = ', '.join(patient_data_df.columns)
        placeholders = ', '.join(['?' for _ in patient_data_df.columns])
        insert_sql = f"INSERT INTO patient_health_metrics ({columns_str}) VALUES ({placeholders})"

        self.cursor.executemany(insert_sql, patient_data_for_db)
        self.conn.commit()
        print(f"Successfully inserted {len(patient_data_for_db)} rows into patient_health_metrics.")

    def get_patient_data(self, min_age=None, max_age=None, gender=None, heart_disease_status=None):
        base_query = "SELECT * FROM patient_health_metrics"
        conditions = []
        parameters = []

        if min_age is not None:
            conditions.append("Age >= ?")
            parameters.append(min_age)
        if max_age is not None:
            conditions.append("Age <= ?")
            parameters.append(max_age)
        if gender is not None:
            conditions.append("Gender = ?")
            parameters.append(gender)
        if heart_disease_status is not None:
            conditions.append("Heart_Disease_Status = ?")
            parameters.append(heart_disease_status)

        if conditions:
            query = base_query + " WHERE " + " AND ".join(conditions)
        else:
            query = base_query

        self.cursor.execute(query, tuple(parameters))
        data = self.cursor.fetchall()

        column_names = [description[0] for description in self.cursor.description]

        return pd.DataFrame(data, columns=column_names)

    def update_patient_data(self, patient_id, **kwargs):
        update_fields = []
        parameters = []

        for key, value in kwargs.items():
            db_column_name = key.replace(' ', '_')
            update_fields.append(f"{db_column_name} = ?")
            parameters.append(value)

        if not update_fields:
            print("No update fields provided.")
            return

        parameters.append(patient_id)
        set_clause = ", ".join(update_fields)
        update_sql = f"UPDATE patient_health_metrics SET {set_clause} WHERE patient_id = ?"

        self.cursor.execute(update_sql, tuple(parameters))
        if self.cursor.rowcount > 0:
            self.conn.commit()
            print(f"Successfully updated patient_id {patient_id}.")
        else:
            print(f"No record found for patient_id {patient_id} or no changes made.")

    def delete_patient_data(self, patient_id):
        delete_sql = "DELETE FROM patient_health_metrics WHERE patient_id = ?"

        self.cursor.execute(delete_sql, (patient_id,))
        if self.cursor.rowcount > 0:
            self.conn.commit()
            print(f"Successfully deleted patient_id {patient_id}.")
        else:
            print(f"No record found for patient_id {patient_id} or no changes made.")

    def close_connection(self):
        self.conn.close()
        print("Database connection closed.")