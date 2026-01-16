import sqlite3
import pandas as pd
import numpy as np

class DatabaseManager:
    def __init__(self, db_name='health_metrics.db'):
        """Initializes connection and enables WAL mode for high performance."""
        self.conn = sqlite3.connect(db_name)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.cursor = self.conn.cursor()
        print(f"Successfully connected to {db_name} in WAL mode.")

    def create_tables(self):
        """Creates the relational tables with a full schema for all medical metrics."""
        # 1. Table for unique patients (One-to-Many relationship starts here)
        create_patients_table = '''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT UNIQUE,
            Gender TEXT
        );
        '''
        # 2. Table for multiple reports linked to a patient
        create_health_reports_table = '''
        CREATE TABLE IF NOT EXISTS patient_health_metrics (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            Age REAL,
            Blood_Pressure REAL,
            Cholesterol_Level REAL,
            BMI REAL,
            Sleep_Hours REAL,
            Triglyceride_Level REAL,
            Fasting_Blood_Sugar REAL,
            CRP_Level REAL,
            Homocysteine_Level REAL,
            Heart_Disease_Status TEXT,
            ECG_Signal TEXT,
            EEG_Signal TEXT,
            Date_Recorded TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
        '''
        self.cursor.execute(create_patients_table)
        self.cursor.execute(create_health_reports_table)
        self.conn.commit()
        print("Tables ensured successfully with full schema.")

    def insert_patient_data(self, df_source):
        """
        Cleans data and performs relational insertion. 
        Links metrics to specific patients and allows multiple entries per patient.
        """
        column_mapping = {
            'Name': 'Name',
            'Gender': 'Gender',
            'ECG Signal': 'ECG_Signal',
            'EEG Signal': 'EEG_Signal',
            'Blood Pressure': 'Blood_Pressure',
            'Cholesterol Level': 'Cholesterol_Level',
            'Sleep Hours': 'Sleep_Hours',
            'Triglyceride Level': 'Triglyceride_Level',
            'Fasting Blood Sugar': 'Fasting_Blood_Sugar',
            'CRP Level': 'CRP_Level',
            'Homocysteine Level': 'Homocysteine_Level',
            'Heart Disease Status': 'Heart_Disease_Status',
            'Date Recorded': 'Date_Recorded'
        }

        df = df_source.copy()

        for space_name, underscore_name in column_mapping.items():
            if space_name in df.columns:
                df.rename(columns={space_name: underscore_name}, inplace=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
            
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])

        rows_inserted = 0
        for _, row in df.iterrows():
            try:
                name = str(row.get('Name', '')).strip()
                gender = row.get('Gender', 'Unknown')
                
                if not name:
                    continue

                # Relational Lookup: Check if patient exists
                self.cursor.execute("SELECT patient_id FROM patients WHERE Name = ?", (name,))
                result = self.cursor.fetchone()

                if result:
                    patient_id = result[0]
                else:
                    self.cursor.execute("INSERT INTO patients (Name, Gender) VALUES (?, ?)", (name, gender))
                    patient_id = self.cursor.lastrowid

                metrics_data = {
                    'patient_id': patient_id,
                    'Age': row.get('Age'),
                    'Blood_Pressure': row.get('Blood_Pressure'),
                    'Cholesterol_Level': row.get('Cholesterol_Level'),
                    'BMI': row.get('BMI'),
                    'Sleep_Hours': row.get('Sleep_Hours'),
                    'Triglyceride_Level': row.get('Triglyceride_Level'),
                    'Fasting_Blood_Sugar': row.get('Fasting_Blood_Sugar'),
                    'CRP_Level': row.get('CRP_Level'),
                    'Homocysteine_Level': row.get('Homocysteine_Level'),
                    'Heart_Disease_Status': row.get('Heart_Disease_Status'),
                    'ECG_Signal': row.get('ECG_Signal'),
                    'EEG_Signal': row.get('EEG_Signal'),
                    'Date_Recorded': row.get('Date_Recorded')
                }

                metrics_data = {k: v for k, v in metrics_data.items() if v is not None}
                columns = ', '.join(metrics_data.keys())
                placeholders = ', '.join(['?' for _ in metrics_data])
                values = tuple(metrics_data.values())
                
                insert_sql = f"INSERT INTO patient_health_metrics ({columns}) VALUES ({placeholders})"
                self.cursor.execute(insert_sql, values)
                rows_inserted += 1

            except Exception as e:
                print(f"Error inserting row for {row.get('Name')}: {e}")

        self.conn.commit()
        print(f"Relational insertion complete. Processed {rows_inserted} entries.")

    def get_patient_data(self, limit=50, offset=0):
        """Fetches joined data from patients and metrics tables for the general table view."""
        query = """
        SELECT m.report_id, p.Name, p.Gender, m.*
        FROM patients p
        JOIN patient_health_metrics m ON p.patient_id = m.patient_id
        ORDER BY m.Date_Recorded DESC
        LIMIT ? OFFSET ?
        """
        try:
            df = pd.read_sql_query(query, self.conn, params=(limit, offset))
            if 'patient_id' in df.columns:
                df = df.loc[:, ~df.columns.duplicated()]
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def get_single_patient_history(self, patient_id):
        """
        Fetches all historical health records for a specific patient ID.
        Used for Time-Series trends and individual correlation analysis.
        """
        query = """
        SELECT m.*, p.Name 
        FROM patient_health_metrics m
        JOIN patients p ON m.patient_id = p.patient_id
        WHERE m.patient_id = ?
        ORDER BY m.Date_Recorded ASC
        """
        try:
            return pd.read_sql_query(query, self.conn, params=(patient_id,))
        except Exception as e:
            print(f"Error fetching history for Patient ID {patient_id}: {e}")
            return pd.DataFrame()

    def get_total_count(self):
        """Returns the total number of health records in the database."""
        try:
            self.cursor.execute("SELECT COUNT(*) FROM patient_health_metrics")
            return self.cursor.fetchone()[0]
        except:
            return 0

    def update_patient_data(self, patient_id, **kwargs):
        """Updates specific fields for a patient record."""
        update_fields = []
        parameters = []
        for key, value in kwargs.items():
            db_column_name = key.replace(' ', '_')
            update_fields.append(f"{db_column_name} = ?")
            parameters.append(value)

        if not update_fields: return
        parameters.append(patient_id)
        set_clause = ", ".join(update_fields)
        update_sql = f"UPDATE patient_health_metrics SET {set_clause} WHERE patient_id = ?"
        self.cursor.execute(update_sql, tuple(parameters))
        self.conn.commit()

    def delete_patient_data(self, patient_id):
        """Deletes a patient record by ID."""
        try:
            delete_sql = "DELETE FROM patient_health_metrics WHERE patient_id = ?"
            self.cursor.execute(delete_sql, (patient_id,))
            self.conn.commit()
            return self.cursor.rowcount
        except Exception as e:
            print(f"Database error during deletion: {e}")
            return -1

    def close_connection(self):
        """Closes the database safely."""
        self.conn.close()
        print("Database connection closed.")