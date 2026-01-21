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
        """Creates the relational tables and handles migrations for new columns."""
        # 1. Table for unique patients
        create_patients_table = '''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT UNIQUE,
            Gender TEXT
        );
        '''
        # 2. Table for health metrics
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
            ECG_FFT_Magnitude TEXT,
            Correlation_Data TEXT,
            EEG_Signal TEXT,
            Date_Recorded TEXT,
            Image_Data BLOB,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
        '''
        self.cursor.execute(create_patients_table)
        self.cursor.execute(create_health_reports_table)
        
        # --- MIGRATION LOGIC ---
        # If the DB existed before we added Correlation_Data, ALTER the table
        try:
            self.cursor.execute("SELECT Correlation_Data FROM patient_health_metrics LIMIT 1")
        except sqlite3.OperationalError:
            print("Migrating database: Adding Correlation_Data column...")
            self.cursor.execute("ALTER TABLE patient_health_metrics ADD COLUMN Correlation_Data TEXT")
            self.conn.commit()
            
        self.conn.commit()
        print("Tables ensured successfully with full schema.")

    def update_correlation_data(self, patient_id, corr_string):
        """Updates the most recent health report for a patient with correlation results."""
        try:
            # Targets the most recent report for this specific patient
            sql = """
                UPDATE patient_health_metrics 
                SET Correlation_Data = ? 
                WHERE report_id = (
                    SELECT MAX(report_id) FROM patient_health_metrics WHERE patient_id = ?
                )
            """
            self.cursor.execute(sql, (corr_string, patient_id))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Database correlation update error: {e}")
            return False

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
        """
        Updates specific fields in a patient record.
        Handles both the 'patients' identity table and the 'metrics' table.
        """
        if not kwargs: return 0
        
        # Define which columns belong to which table
        patient_identity_fields = ['Name', 'Gender']
        p_updates = {k: v for k, v in kwargs.items() if k in patient_identity_fields}
        m_updates = {k: v for k, v in kwargs.items() if k not in patient_identity_fields}
        
        rows_affected = 0
        try:
            # Update Identity table (Name, Gender)
            if p_updates:
                sets = ", ".join([f"{k} = ?" for k in p_updates.keys()])
                self.cursor.execute(f"UPDATE patients SET {sets} WHERE patient_id = ?", 
                                   (*p_updates.values(), patient_id))
                rows_affected += self.cursor.rowcount
            
            # Update Metrics table (Age, BP, etc.)
            if m_updates:
                sets = ", ".join([f"{k} = ?" for k in m_updates.keys()])
                # This updates all reports for this specific patient
                self.cursor.execute(f"UPDATE patient_health_metrics SET {sets} WHERE patient_id = ?", 
                                   (*m_updates.values(), patient_id))
                rows_affected += self.cursor.rowcount
            
            self.conn.commit()
            return rows_affected
        except Exception as e:
            print(f"Database Update error: {e}")
            if hasattr(self, 'conn'): self.conn.rollback()
            return 0

    def delete_patient_data(self, patient_id):
        """
        Deletes all health records and the patient identity.
        Returns the number of patient records removed (0 if not found).
        """
        try:
            # 1. Delete dependent health metrics first
            self.cursor.execute("DELETE FROM patient_health_metrics WHERE patient_id = ?", (patient_id,))
            
            # 2. Delete the primary patient record
            self.cursor.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
            
            # 3. Capture how many patients were actually removed (0 or 1)
            rows_deleted = self.cursor.rowcount
            
            self.conn.commit()
            return rows_deleted # Returns 1 if deleted, 0 if patient didn't exist

        except Exception as e:
            print(f"Database Error during Deletion: {e}")
            if hasattr(self, 'conn'):
                self.conn.rollback()
            return 0

        except Exception as e:
            print(f"Database Error during Deletion: {e}")
            if hasattr(self, 'conn'):
                self.conn.rollback()
            return 0

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
        
    def get_patient_images(self, patient_id):
        """Returns a list of (report_id, Date_Recorded) for a specific patient."""
        try:
            # Only select rows where Image_Data is not NULL
            sql = "SELECT report_id, Date_Recorded FROM patient_health_metrics WHERE patient_id = ? AND Image_Data IS NOT NULL"
            self.cursor.execute(sql, (patient_id,))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error fetching images for patient {patient_id}: {e}")
            return []
        
    def update_fft_data(self, patient_id, fft_string):
        """Updates the most recent health report for a patient with FFT data."""
        try:
            # We target the most recent report for this patient
            sql = """
                UPDATE patient_health_metrics 
                SET ECG_FFT_Magnitude = ? 
                WHERE report_id = (
                    SELECT MAX(report_id) FROM patient_health_metrics WHERE patient_id = ?
                )
            """
            self.cursor.execute(sql, (fft_string, patient_id))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Database update error: {e}")
            return False
    
    def get_all_records_for_patient(self, patient_id):
        """Fetches every record for a specific patient, regardless of pagination."""
        query = """
        SELECT m.*, p.Name, p.Gender 
        FROM patient_health_metrics m
        JOIN patients p ON m.patient_id = p.patient_id
        WHERE m.patient_id = ?
        ORDER BY m.Date_Recorded ASC
        """
        return pd.read_sql_query(query, self.conn, params=(patient_id,))

    def get_total_count(self):
        self.cursor.execute("SELECT COUNT(*) FROM patient_health_metrics")
        return self.cursor.fetchone()[0]
    

    def close_connection(self):
        self.conn.close()
        print("Database connection closed.")