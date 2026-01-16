# main.py (MODIFIED)
import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from data_analyzer import load_data
from gui_app import HealthcareApp  
from database_manager import DatabaseManager
from insert_thread import InsertDataThread

def load_qss(app, qss_path="styles.qss"):
    """Load QSS stylesheet if available (no crash if missing)."""
    try:
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                qss = f.read()
            app.setStyleSheet(qss)
            print(f"Loaded stylesheet: {qss_path}")
        else:
            print(f"Stylesheet not found at {qss_path}, skipping.")
    except Exception as e:
        print(f"Failed to load stylesheet: {e}")

def safe_load_csv(path):
    """Load CSV with helpful errors."""
    try:
        df = load_data(path) 
        print(f"Loaded dataset with {len(df)} rows.")
        return df
    except FileNotFoundError:
        err = f"CSV file not found: {path}"
        print(err)
        raise 
    except Exception as e:
        print(f"Error loading CSV '{path}': {e}")
        raise

if __name__ == "__main__":
    app = QApplication(sys.argv)

    load_qss(app, "styles.qss")

    csv_path = os.path.join(os.path.dirname(__file__), "heart_disease.csv")
    try:
        df = safe_load_csv(csv_path)
    except Exception as e:
        QMessageBox.critical(None, "Fatal Error", f"Failed to load CSV: {csv_path}\nError: {e}")
        sys.exit(1)

    db_path = os.path.join(os.path.dirname(__file__), "health_metrics.db")
    db_manager = DatabaseManager(db_name=db_path)
    try:
        db_manager.create_tables()
        print("Database tables ensured.")
    except Exception as e:
        print(f"Error creating tables: {e}")
        QMessageBox.warning(None, "Database Error", f"Could not create tables: {e}")

    main_window = HealthcareApp(df, db_manager=db_manager) 
    main_window.show()

    def update_status(message: str):
        try:
            if hasattr(main_window, "status_label") and main_window.status_label is not None:
                main_window.status_label.setText(message)
        except Exception:
            pass
        print(message)

    try:
        insert_thread = InsertDataThread(db_path, df)
        insert_thread.progress.connect(update_status) 
    except Exception as e:
        print(f"Failed to start insert thread: {e}")
        QMessageBox.warning(None, "Insert Error", "Failed to start background data insertion.")

    sys.exit(app.exec_())