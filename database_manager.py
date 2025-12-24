import sqlite3
import pandas as pd
import numpy as np

class DatabaseManager:
    def __init__(self, db_name='health_metrics.db'):
        """Initializes connection and enables WAL mode for high performance."""
        self.conn = sqlite3.connect(db_name)
        
        # PERFORMANCE TWEAK: Enable Write-Ahead Logging.
        # This allows background threads to write while you read/delete from the UI.
        self.conn.execute('PRAGMA journal_mode=WAL;')
        
        self.cursor = self.conn.cursor()
        print(f"Successfully connected to {db_name} in WAL mode.")

    def create_tables(self):
        """Ensures the required tables exist in the database."""
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
        print("Tables ensured successfully.")

    def insert_patient_data(self, df_source):
        """Cleans data and performs bulk insertion. Handles both Space and Underscore names."""
        
        # Mapping of CSV names (with spaces) to SQL names (with underscores)
        column_mapping = {
            'Blood Pressure': 'Blood_Pressure',
            'Cholesterol Level': 'Cholesterol_Level',
            'Sleep Hours': 'Sleep_Hours',
            'Triglyceride Level': 'Triglyceride_Level',
            'Fasting Blood Sugar': 'Fasting_Blood_Sugar',
            'CRP Level': 'CRP_Level',
            'Homocysteine Level': 'Homocysteine_Level',
            'Heart Disease Status': 'Heart_Disease_Status'
        }

        # Create a copy to avoid modifying the original GUI dataframe
        df = df_source.copy()

        # FLEXIBLE COLUMN CHECK:
        # If the input has spaces (CSV), rename them to underscores (SQL).
        # If it already has underscores (Retrieved from DB), it skips renaming.
        for space_name, underscore_name in column_mapping.items():
            if space_name in df.columns:
                df.rename(columns={space_name: underscore_name}, inplace=True)

        # The exact columns the database table expects
        required_sql_columns = [
            'Age', 'Gender', 'Blood_Pressure', 'Cholesterol_Level', 'BMI', 
            'Sleep_Hours', 'Triglyceride_Level', 'Fasting_Blood_Sugar', 
            'CRP_Level', 'Homocysteine_Level', 'Heart_Disease_Status'
        ]

        # Final check to ensure we only try to insert what the DB can handle
        available_cols = [c for c in required_sql_columns if c in df.columns]
        df = df[available_cols]

        # Handling Missing Values (Imputation)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=['object']).columns:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])

        # Batch Insertion
        data_to_insert = [tuple(row) for row in df.itertuples(index=False)]
        columns_str = ', '.join(df.columns)
        placeholders = ', '.join(['?' for _ in df.columns])
        insert_sql = f"INSERT INTO patient_health_metrics ({columns_str}) VALUES ({placeholders})"

        self.cursor.executemany(insert_sql, data_to_insert)
        self.conn.commit()
        print(f"Batch inserted {len(data_to_insert)} rows.")

    def get_patient_data(self, min_age=None, max_age=None, gender=None, 
                         heart_disease_status=None, limit=50, offset=0):
        """Fetches a specific 'page' of data using LIMIT and OFFSET."""
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

        query = base_query
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # PAGINATION: Prevents UI lag
        query += " LIMIT ? OFFSET ?"
        parameters.extend([limit, offset])

        self.cursor.execute(query, tuple(parameters))
        data = self.cursor.fetchall()
        column_names = [description[0] for description in self.cursor.description]

        return pd.DataFrame(data, columns=column_names)

    def get_total_count(self):
        """Returns total records for UI pagination math."""
        self.cursor.execute("SELECT COUNT(*) FROM patient_health_metrics")
        result = self.cursor.fetchone()
        return result[0] if result else 0

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
        delete_sql = "DELETE FROM patient_health_metrics WHERE patient_id = ?"
        self.cursor.execute(delete_sql, (patient_id,))
        self.conn.commit()

    def close_connection(self):
        """Closes the database safely."""
        self.conn.close()
        print("Database connection closed.")