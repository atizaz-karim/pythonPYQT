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
        self.create_tables()

    def create_tables(self):
        """Creates the relational tables with a full schema including image support."""
        # 1. Table for unique patients
        create_patients_table = '''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT UNIQUE,
            Gender TEXT
        );
        '''
        # 2. Table for health metrics including BLOB for image storage
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
            Image_Data BLOB,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
        '''
        self.cursor.execute(create_patients_table)
        self.cursor.execute(create_health_reports_table)
        self.conn.commit()
        print("Tables ensured successfully with full schema (including Image BLOB).")

    def insert_patient_data(self, df_source):
        """Processes and inserts a DataFrame of patient data into the relational structure."""
        column_mapping = {
            'Name': 'Name', 'Gender': 'Gender', 'ECG Signal': 'ECG_Signal',
            'EEG Signal': 'EEG_Signal', 'Blood Pressure': 'Blood_Pressure',
            'Cholesterol Level': 'Cholesterol_Level', 'Sleep Hours': 'Sleep_Hours',
            'Triglyceride Level': 'Triglyceride_Level', 'Fasting Blood Sugar': 'Fasting_Blood_Sugar',
            'CRP Level': 'CRP_Level', 'Homocysteine Level': 'Homocysteine_Level',
            'Heart Disease Status': 'Heart_Disease_Status', 'Date Recorded': 'Date_Recorded'
        }

        df = df_source.copy()
        for space_name, underscore_name in column_mapping.items():
            if space_name in df.columns:
                df.rename(columns={space_name: underscore_name}, inplace=True)

        rows_inserted = 0
        for i, row in df.iterrows():
            try:
                name = str(row.get('Name', '')).strip()
                if not name or name.lower() == 'nan':
                    name = f"Patient_{i+1}"
                
                gender = row.get('Gender', 'Unknown')

                # Handle relational patient lookup/creation
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

                # Clean and Insert
                metrics_data = {k: v for k, v in metrics_data.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
                columns = ', '.join(metrics_data.keys())
                placeholders = ', '.join(['?' for _ in metrics_data])
                values = tuple(metrics_data.values())
                
                self.cursor.execute(f"INSERT INTO patient_health_metrics ({columns}) VALUES ({placeholders})", values)
                rows_inserted += 1

            except Exception as e:
                print(f"Error inserting row {i}: {e}")

        self.conn.commit()
        print(f"Relational insertion complete. Processed {rows_inserted} entries.")

    def insert_manual_record(self, metrics_data):
        try:
            # Relational Patient Logic
            if 'Name' in metrics_data:
                name = metrics_data.pop('Name')
                gender = metrics_data.pop('Gender', 'Unknown')
                
                self.cursor.execute("SELECT patient_id FROM patients WHERE Name = ?", (name,))
                result = self.cursor.fetchone()
                if not result:
                    self.cursor.execute("INSERT INTO patients (Name, Gender) VALUES (?, ?)", (name, gender))
                    patient_id = self.cursor.lastrowid
                else:
                    patient_id = result[0]
                metrics_data['patient_id'] = patient_id

            # CRITICAL: Convert raw bytes to SQLite Binary
            if 'Image_Data' in metrics_data and metrics_data['Image_Data'] is not None:
                metrics_data['Image_Data'] = sqlite3.Binary(metrics_data['Image_Data'])

            # Build and execute query
            columns = ', '.join(metrics_data.keys())
            placeholders = ', '.join(['?' for _ in metrics_data])
            query = f"INSERT INTO patient_health_metrics ({columns}) VALUES ({placeholders})"
            
            self.cursor.execute(query, tuple(metrics_data.values()))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Database Error: {e}")
            return False

    def get_patient_data(self, limit=50, offset=0):
        """Fetches data for the main table view."""
        query = """
        SELECT m.report_id, p.Name, p.Gender, m.*
        FROM patients p
        JOIN patient_health_metrics m ON p.patient_id = m.patient_id
        ORDER BY m.Date_Recorded DESC
        LIMIT ? OFFSET ?
        """
        try:
            df = pd.read_sql_query(query, self.conn, params=(limit, offset))
            # Remove duplicate patient_id column if present from JOIN
            return df.loc[:, ~df.columns.duplicated()]
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def update_patient_data(self, patient_id, **kwargs):
        """Updates specific fields in a patient record."""
        update_fields = []
        parameters = []
        for key, value in kwargs.items():
            db_column_name = key.replace(' ', '_')
            update_fields.append(f"{db_column_name} = ?")
            parameters.append(value)

        if not update_fields: return
        parameters.append(patient_id)
        set_clause = ", ".join(update_fields)
        self.cursor.execute(f"UPDATE patient_health_metrics SET {set_clause} WHERE patient_id = ?", tuple(parameters))
        self.conn.commit()

    def convert_to_binary(self, file_path):
        """Helper to convert image file to binary BLOB."""
        try:
            with open(file_path, 'rb') as file:
                return file.read()
        except Exception as e:
            print(f"Binary conversion error: {e}")
            return None

    def save_image_to_db(self, report_id, image_bytes):
        """Updates an existing record with processed image data."""
        try:
            sql = "UPDATE patient_health_metrics SET Image_Data = ? WHERE report_id = ?"
            self.cursor.execute(sql, (sqlite3.Binary(image_bytes), report_id))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Failed to save image to DB: {e}")
            return False

    def retrieve_image_from_db(self, report_id):
        """Fetches image BLOB for a specific report."""
        try:
            self.cursor.execute("SELECT Image_Data FROM patient_health_metrics WHERE report_id = ?", (report_id,))
            row = self.cursor.fetchone()
            return row[0] if row and row[0] else None
        except Exception as e:
            print(f"Error retrieving image: {e}")
            return None

    def search_patient(self, query):
        """Searches by ID or Name."""
        try:
            if query.isdigit():
                sql = "SELECT p.Name, p.Gender, m.* FROM patients p JOIN patient_health_metrics m ON p.patient_id = m.patient_id WHERE p.patient_id = ?"
                params = (int(query),)
            else:
                sql = "SELECT p.Name, p.Gender, m.* FROM patients p JOIN patient_health_metrics m ON p.patient_id = m.patient_id WHERE p.Name LIKE ?"
                params = (f"%{query}%",)
            return pd.read_sql_query(sql, self.conn, params=params)
        except Exception as e:
            print(f"Search error: {e}")
            return pd.DataFrame()

    def get_total_count(self):
        self.cursor.execute("SELECT COUNT(*) FROM patient_health_metrics")
        return self.cursor.fetchone()[0]

    def close_connection(self):
        self.conn.close()
        print("Database connection closed.")