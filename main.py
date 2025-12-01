# main.py (MODIFIED)
import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from data_analyzer import load_data
from gui_app import HealthcareApp  # <--- CORRECTED: Changed HeartDiseaseApp to HealthcareApp
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
        # data_analyzer.load_data is assumed to handle the core loading logic
        df = load_data(path) 
        print(f"Loaded dataset with {len(df)} rows.")
        return df
    except FileNotFoundError:
        err = f"CSV file not found: {path}"
        print(err)
        # Re-raise to be caught by the main block for the QMessageBox
        raise 
    except Exception as e:
        print(f"Error loading CSV '{path}': {e}")
        raise

if __name__ == "__main__":
    # Single QApplication instance
    app = QApplication(sys.argv)

    # Load stylesheet (optional)
    load_qss(app, "styles.qss")

    # Load dataset
    csv_path = os.path.join(os.path.dirname(__file__), "heart_disease.csv")
    try:
        df = safe_load_csv(csv_path)
    except Exception as e:
        # Display critical error and exit if data cannot be loaded
        QMessageBox.critical(None, "Fatal Error", f"Failed to load CSV: {csv_path}\nError: {e}")
        sys.exit(1)

    # Initialize database and tables
    db_path = os.path.join(os.path.dirname(__file__), "health_metrics.db")
    db_manager = DatabaseManager(db_name=db_path)
    try:
        db_manager.create_tables()
        print("Database tables ensured.")
    except Exception as e:
        print(f"Error creating tables: {e}")
        QMessageBox.warning(None, "Database Error", f"Could not create tables: {e}")

    # Create and show main GUI (pass df and db_manager)
    # The constructor is now HealthcareApp(df, db_manager)
    main_window = HealthcareApp(df, db_manager=db_manager) 
    main_window.show()

    # Helper to update status label (connected to the GUI's status_label)
    def update_status(message: str):
        try:
            # Check if the main_window and status_label attribute exist
            if hasattr(main_window, "status_label") and main_window.status_label is not None:
                main_window.status_label.setText(message)
        except Exception:
            # Silent failure if the status bar/label is not yet ready
            pass
        print(message)

    # Start background insertion thread AFTER GUI is shown
    try:
        # Note: InsertDataThread logic is outside of this file, assume it works.
        insert_thread = InsertDataThread(db_path, df)
        # progress.connect must be used if InsertDataThread emits a signal
        insert_thread.progress.connect(update_status) 
        insert_thread.start()
    except Exception as e:
        print(f"Failed to start insert thread: {e}")
        QMessageBox.warning(None, "Insert Error", "Failed to start background data insertion.")

    # Run the Qt event loop
    sys.exit(app.exec_())