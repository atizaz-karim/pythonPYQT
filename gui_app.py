from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QTabWidget, QFileDialog, QTableWidget,
    QTableWidgetItem, QScrollArea, QSlider, QLineEdit, QCheckBox,
    QSpinBox, QMessageBox, QGridLayout, QInputDialog, QDoubleSpinBox,
    QFrame, QGroupBox, QHeaderView
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QDateTimeEdit
from PyQt5.QtCore import QDateTime

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Add this line near your other imports
from data_analyzer import fft_denoise_signal, analyze_ecg_signal, analyze_eeg_signal

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

class DatabaseManager:
    def __init__(self): pass
    def get_patient_data(self): return pd.DataFrame()
    def insert_patient_data(self, df): pass
    def update_patient_data(self, patient_id, **kwargs): pass
    def delete_patient_data(self, patient_id): pass

class HealthcareApp(QMainWindow):
    def __init__(self, df, db_manager=None):

        super().__init__()
        self.df = df
        self.db_manager = db_manager
        self.current_page = 0
        self.rows_per_page = 50
        self.filtered_df = df.copy()
        self.cv_image = None
        self.processed_cv_image = None

        self.setWindowTitle("Healthcare Data and Medical Image Processing Tool")
        self.setGeometry(50, 50, 1400, 800)
        
        # 2. Setup the Scroll Area
        self.main_scroll = QScrollArea() 
        self.main_scroll.setObjectName("MainScrollArea") 
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setFrameShape(QFrame.NoFrame)

        # 3. Create the container widget
        container_widget = QWidget()
        self.main_scroll.setWidget(container_widget)
        
        # FIX: Remove the duplicate line that uses the wrong name
        self.setCentralWidget(self.main_scroll) 

        # 4. DEFINE main_layout HERE
        main_layout = QHBoxLayout(container_widget)

        # Now the following lines will work because main_layout is defined above
        self.sidebar = QWidget()
        self.sidebar.setObjectName("SidebarWidget")
        self.sidebar.setFixedWidth(250)
        self.sidebar_layout = QVBoxLayout(self.sidebar)

        self.stacked_widget = QTabWidget()
        self.stacked_widget.tabBar().setVisible(False) 

        self.stacked_widget.addTab(self.create_data_management_panel(), "Data Loading and Management")
        self.stacked_widget.addTab(self.create_analysis_panel(), "Health Data Analysis")
        self.stacked_widget.addTab(self.create_spectrum_panel(), "Signal Analysis")
        self.stacked_widget.addTab(self.create_image_processing_panel(), "Medical Image Processing")
        self.stacked_widget.addTab(self.create_data_visualization_panel(), "Data Visualization")

        tab_names = ["Patient Data Management", "Health Data Analysis", "Spectrum Analysis",
                     "Image Processing", "Data Visualization"]

        for i, name in enumerate(tab_names):
            btn = QPushButton(name)
            btn.setObjectName(f"SidebarBtn_{i}") 
            btn.setToolTip(f"Switch to the {name} section.")
            btn.clicked.connect(lambda checked, index=i: self.stacked_widget.setCurrentIndex(index))
            self.sidebar_layout.addWidget(btn)

        self.sidebar_layout.addStretch(1)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stacked_widget)

        self._setup_menu_bar()

        if self.db_manager:
            self.db_retrieve_data()
        else:
            self.populate_table(self.df.head(self.rows_per_page))

    def _setup_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")

        load_data_action = file_menu.addAction("Load Data (Ctrl+L)")
        load_data_action.setShortcut("Ctrl+L")
        load_data_action.triggered.connect(lambda: self.stacked_widget.setCurrentIndex(0)) 
        load_data_action.setToolTip("Load data from CSV or connect to a database.")
        
        save_results_action = file_menu.addAction("Save Results (Ctrl+S)")
        save_results_action.setShortcut("Ctrl+S")
        save_results_action.setToolTip("Save current processed data or visualization output.")
        
        file_menu.addSeparator()
        file_menu.addAction("Exit").triggered.connect(self.close)

        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("Documentation").setToolTip("View detailed tool documentation.")
        help_menu.addAction("Report Issue").setToolTip("Provide feedback or report a bug.")
        
    def populate_table(self, df):
        if not hasattr(self, 'table_widget'):
            return

        self.table_widget.setRowCount(0) 
        self.table_widget.clearContents()

        if df is None or df.empty:
            self.table_widget.setColumnCount(0)
            return

        # Columns to exclude from the dataset view
        columns_to_exclude = [
            'Sleep_Hours', 'Sleep Hours',
            'Triglyceride_Level', 'Triglyceride Level',
            'CRP_Level', 'CRP Level',
            'Homocysteine_Level', 'Homocysteine Level'
        ]
        
        # Filter out excluded columns
        display_df = df.drop(columns=[col for col in columns_to_exclude if col in df.columns], errors='ignore')

        self.table_widget.setRowCount(len(display_df))
        self.table_widget.setColumnCount(len(display_df.columns))
        self.table_widget.setHorizontalHeaderLabels(display_df.columns)

        for i in range(len(display_df)):
            for j, col in enumerate(display_df.columns):
                val = display_df.iloc[i, j]
                
                # --- NEW: Check for ECG/EEG Signal Columns ---
                if col in ['ECG_Signal', 'EEG_Signal']:
                    val_str = str(val).strip() if val is not None else ""
                    if val_str and val_str != "nan" and len(val_str) > 0:
                        # Count number of data points by splitting commas
                        points = len(val_str.split(','))
                        # Create a preview label: e.g., "📈 500 pts [0.12, 0.45...]"
                        display_text = f"{points} pts [{val_str[:12]}...]"
                    else:
                        display_text = "No Signal"
                else:
                    # Default handling for regular columns
                    display_text = str(val) if val is not None else ""

                self.table_widget.setItem(i, j, QTableWidgetItem(display_text))

        # UI Fix: Don't resize Signal columns to full contents (they are too long)
        header = self.table_widget.horizontalHeader()
        for i, col_name in enumerate(display_df.columns):
            if "Signal" in col_name:
                self.table_widget.setColumnWidth(i, 160) # Keep signal columns compact
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        self.table_widget.setToolTip("Scrollable view of the loaded dataset. Signals are shown as previews.")

# --- Panel 1: Data Management (CRUD) ---
    def create_data_management_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("Data Loading and Management")
        title.setObjectName("PanelTitle") 
        layout.addWidget(title)

        source_group = QWidget()
        source_layout = QHBoxLayout(source_group)
        source_layout.setAlignment(Qt.AlignLeft)
        
        self.load_csv_btn = QPushButton("Load CSV File (Local)")
        self.load_csv_btn.setObjectName("LoadCSVButton")
        self.load_csv_btn.setToolTip("Browse and load data from a local CSV file.")
        self.load_csv_btn.clicked.connect(self.load_csv)

        # self.db_connect_btn = QPushButton("Refresh Table")
        # self.db_connect_btn.setObjectName("RefreshTableButton")
        # self.db_connect_btn.setToolTip("Retrieves all patient records from the database and loads them into the table.")
        # self.db_connect_btn.clicked.connect(self.db_retrieve_data) 
        
        source_layout.addWidget(self.load_csv_btn)
        # source_layout.addWidget(self.db_connect_btn)
        source_layout.addStretch()
        layout.addWidget(source_group)

# --- New Manual Entry Section ---
        manual_entry_group = QGroupBox("Manual Patient Entry & Diagnostics")
        manual_layout = QGridLayout(manual_entry_group)
        manual_layout.setSpacing(15) 
        manual_layout.setContentsMargins(20, 20, 20, 20)

        # Row 0: Name and Age
        manual_layout.addWidget(QLabel("Patient Name:"), 0, 0)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Full Name")
        manual_layout.addWidget(self.name_input, 0, 1)

        manual_layout.addWidget(QLabel("Age:"), 0, 2)
        self.age_input = QSpinBox()
        self.age_input.setRange(0, 120)
        self.age_input.setSuffix(" yrs")
        manual_layout.addWidget(self.age_input, 0, 3)

        # Row 1: Gender and Blood Pressure
        manual_layout.addWidget(QLabel("Gender:"), 1, 0)
        self.gender_input = QComboBox()
        self.gender_input.addItems(["Male", "Female", "Other"])
        manual_layout.addWidget(self.gender_input, 1, 1)

        manual_layout.addWidget(QLabel("Blood Pressure (sys):"), 1, 2)
        self.bp_input = QDoubleSpinBox()
        self.bp_input.setRange(50, 250)
        self.bp_input.setSuffix(" mmHg")
        manual_layout.addWidget(self.bp_input, 1, 3)

        # Row 2: Cholesterol and BMI
        manual_layout.addWidget(QLabel("Cholesterol:"), 2, 0)
        self.chol_input = QDoubleSpinBox()
        self.chol_input.setRange(100, 400)
        self.chol_input.setSuffix(" mg/dL")
        manual_layout.addWidget(self.chol_input, 2, 1)

        manual_layout.addWidget(QLabel("BMI:"), 2, 2)
        self.bmi_input = QDoubleSpinBox()
        self.bmi_input.setRange(10, 60)
        manual_layout.addWidget(self.bmi_input, 2, 3)

        # Row 3: Blood Sugar and Status
        manual_layout.addWidget(QLabel("Fasting Blood Sugar:"), 3, 0)
        self.sugar_input = QDoubleSpinBox()
        self.sugar_input.setRange(50, 300)
        self.sugar_input.setSuffix(" mg/dL")
        manual_layout.addWidget(self.sugar_input, 3, 1)

        manual_layout.addWidget(QLabel("Heart Disease Status:"), 3, 2)
        self.status_input = QComboBox()
        self.status_input.addItems(["Negative", "Positive"])
        manual_layout.addWidget(self.status_input, 3, 3)

        # Row 4: ECG and EEG Data (Now clearly in Row 4)
        manual_layout.addWidget(QLabel("ECG Signal (CSV):"), 4, 0)
        self.ecg_input = QLineEdit()
        self.ecg_input.setPlaceholderText("e.g., 0.1, 0.5, 0.8...")
        manual_layout.addWidget(self.ecg_input, 4, 1)

        manual_layout.addWidget(QLabel("EEG Signal (CSV):"), 4, 2)
        self.eeg_input = QLineEdit()
        self.eeg_input.setPlaceholderText("e.g., -20, 15, 30...")
        manual_layout.addWidget(self.eeg_input, 4, 3)

        # Row 5: Date of Record (Moved to Row 5 to avoid overlap)
        manual_layout.addWidget(QLabel("Date of Record:"), 5, 0)
        # Ensure QDateTimeEdit and QDateTime are imported from PyQt5.QtWidgets/QtCore
        self.date_input = QDateTimeEdit(QDateTime.currentDateTime())
        self.date_input.setCalendarPopup(True) 
        manual_layout.addWidget(self.date_input, 5, 1)
        # Assign ID for CSS
        self.date_input.setObjectName("dateInput")

        manual_layout.addWidget(QLabel("Medical Image:"), 5, 2)
        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("Select image path...")
        self.image_path_input.setReadOnly(True) # Path updated via button
        
        self.browse_img_btn = QPushButton("Browse")
        self.browse_img_btn.clicked.connect(self.browse_patient_image)
        
        image_hb = QHBoxLayout()
        image_hb.addWidget(self.image_path_input)
        image_hb.addWidget(self.browse_img_btn)
        manual_layout.addLayout(image_hb, 5, 3)

        # Row 6: Submit Button (Moved to Row 6)
        self.add_patient_btn = QPushButton("Save Patient Record")
        self.add_patient_btn.setMinimumHeight(40)
        self.add_patient_btn.setObjectName("saveButton")
        self.add_patient_btn.clicked.connect(self.manual_db_insert)
        
        # Grid parameters: (widget, row, column, rowSpan, columnSpan)
        manual_layout.addWidget(self.add_patient_btn, 6, 0, 1, 4)

        layout.addWidget(manual_entry_group)
        
        db_ops_group = QWidget()
        db_ops_group.setObjectName("DbOpsGroup")
        db_ops_layout = QGridLayout(db_ops_group)
        
        db_ops_layout.addWidget(QLabel("### Database CRUD Operations"), 0, 0, 1, 4)
        
        db_ops_layout.addWidget(QLabel("Patient ID:"), 1, 0)
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setObjectName("PatientIDInput")
        self.patient_id_input.setPlaceholderText("Enter Patient ID for Update/Delete")
        self.patient_id_input.setFixedWidth(200)
        self.patient_id_input.setToolTip("Specify the unique ID of the record to modify or delete.")
        db_ops_layout.addWidget(self.patient_id_input, 1, 1)

        # self.insert_btn = QPushButton("Insert Current Data")
        # self.insert_btn.setObjectName("InsertButton") 
        # self.insert_btn.setToolTip("Inserts the currently loaded DataFrame into the DB (bulk insert).")
        # self.insert_btn.clicked.connect(self.db_insert_data)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search Name or ID...")
        self.search_input.setFixedWidth(180)
        db_ops_layout.addWidget(self.search_input, 1, 2)

        self.search_btn = QPushButton("Search Patient")
        self.search_btn.setObjectName("SearchButton")
        # Connect this to your search logic
        self.search_btn.clicked.connect(self.db_search_patient) 
        db_ops_layout.addWidget(self.search_btn, 1, 3)

        self.update_btn = QPushButton("Update Record")
        self.update_btn.setObjectName("UpdateButton")
        self.update_btn.setToolTip("Updates the record specified by Patient ID (requires prompt).")
        self.update_btn.clicked.connect(self.db_update_prompt) 
        
        self.delete_btn = QPushButton("Delete Record")
        self.delete_btn.setObjectName("DeleteButton") 
        self.delete_btn.setToolTip("Removes the record specified by Patient ID permanently.")
        self.delete_btn.clicked.connect(self.db_delete_record)

        # db_ops_layout.addWidget(self.insert_btn, 2, 0)
        db_ops_layout.addWidget(self.update_btn, 2, 1)
        db_ops_layout.addWidget(self.delete_btn, 2, 2)

        self.clear_btn = QPushButton("Clear Search")
        self.clear_btn.clicked.connect(self.db_retrieve_data)
        db_ops_layout.addWidget(self.clear_btn, 2, 3)

        db_ops_layout.setColumnStretch(4, 1) 
        layout.addWidget(db_ops_group)

        layout.addWidget(db_ops_group)

        layout.addWidget(QLabel("Loaded Dataset / Database View:"))
        self.table_widget = QTableWidget()
        self.table_widget.setMinimumHeight(450)
        layout.addWidget(self.table_widget)
        self.populate_table(self.df) 
        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("<")
        self.prev_btn.setObjectName("PreviousPageButton")
        self.next_btn = QPushButton(">")
        self.next_btn.setObjectName("NextPageButton")
        self.page_label = QLabel("Page 1")
        self.prev_btn.setEnabled(False)
        
        self.prev_btn.clicked.connect(self.load_previous_page)
        self.next_btn.clicked.connect(self.load_next_page)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.page_label)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn)
        layout.addLayout(nav_layout)
        layout.addStretch()

        return panel
    def browse_patient_image(self):
        """Opens file dialog to select an image for the record."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Patient Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path_input.setText(file_path)

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                # 1. Read CSV
                new_df = pd.read_csv(file_path)
                
                # 2. Automatically save and refresh
                if self.db_manager:
                    self.db_manager.insert_patient_data(new_df)
                    self.current_page = 0
                    self.db_retrieve_data() # This refreshes the table view (canvas)
                    QMessageBox.information(self, "Success", f"Data from {os.path.basename(file_path)} saved and table updated.")
                else:
                    self.df = new_df
                    self.populate_table(self.df)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def _update_analysis_dropdowns(self):
        """Refreshes the dropdown menus while filtering out non-analytical columns like Patient ID."""
        if self.df is None or self.df.empty:
            return

        try:
            numerical_cols = [
                col for col in self.df.select_dtypes(include=np.number).columns.tolist()
                if 'ID' not in col.upper()
            ]

            if hasattr(self, 'ts_analysis_column'):
                current_ts = self.ts_analysis_column.currentText()
                self.ts_analysis_column.clear()
                self.ts_analysis_column.addItems(numerical_cols if numerical_cols else [])
                if current_ts in numerical_cols:
                    self.ts_analysis_column.setCurrentText(current_ts)
                elif len(numerical_cols) > 0:
                    self.ts_analysis_column.setCurrentIndex(0)

            # 2. Update Correlation Metric 1
            if hasattr(self, 'metric1_dropdown'):
                current_m1 = self.metric1_dropdown.currentText()
                self.metric1_dropdown.clear()
                self.metric1_dropdown.addItems(numerical_cols if numerical_cols else [])
                if current_m1 in numerical_cols:
                    self.metric1_dropdown.setCurrentText(current_m1)
                elif len(numerical_cols) > 0:
                    self.metric1_dropdown.setCurrentIndex(0)

            # 3. Update Correlation Metric 2
            if hasattr(self, 'metric2_dropdown'):
                current_m2 = self.metric2_dropdown.currentText()
                self.metric2_dropdown.clear()
                self.metric2_dropdown.addItems(numerical_cols if numerical_cols else [])
                if current_m2 in numerical_cols:
                    self.metric2_dropdown.setCurrentText(current_m2)
                # Try to select the second available metric for a better initial scatter plot
                elif len(numerical_cols) > 1:
                    self.metric2_dropdown.setCurrentIndex(1)
                elif len(numerical_cols) > 0:
                    self.metric2_dropdown.setCurrentIndex(0)
                    
        except Exception as e:
            print(f"Error updating analysis dropdowns: {e}")

    def _update_viz_dropdowns(self):
        """Refreshes visualization dropdowns with numeric columns and biomedical signals."""
        if self.df is None or self.df.empty:
            return
            
        try:
            # FIX: Define numerical_cols by selecting all numeric types from the dataframe
            numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()

            # Filter specifically for biomedical signals (ECG/EEG)
            biomed_cols = [col for col in self.df.columns if 'EEG' in col.upper() or 'ECG' in col.upper()]
            
            # 1. Update Signal Selection Dropdown
            if hasattr(self, 'signal_dropdown'):
                current_signal = self.signal_dropdown.currentText()
                self.signal_dropdown.clear()
                # Use biomed_cols if found, otherwise fall back to all numerical columns
                items = biomed_cols if biomed_cols else numerical_cols
                self.signal_dropdown.addItems(items if items else ["No Data"])
                if current_signal in items:
                    self.signal_dropdown.setCurrentText(current_signal)

            # 2. Update Scatter Plot X-Axis Dropdown
            if hasattr(self, 'scatter_x_combo'):
                current_x = self.scatter_x_combo.currentText()
                self.scatter_x_combo.clear()
                self.scatter_x_combo.addItems(numerical_cols if numerical_cols else [])
                if current_x in numerical_cols:
                    self.scatter_x_combo.setCurrentText(current_x)
                elif len(numerical_cols) > 0:
                    self.scatter_x_combo.setCurrentIndex(0)
            
            # 3. Update Scatter Plot Y-Axis Dropdown
            if hasattr(self, 'scatter_y_combo'):
                current_y = self.scatter_y_combo.currentText()
                self.scatter_y_combo.clear()
                self.scatter_y_combo.addItems(numerical_cols if numerical_cols else [])
                if current_y in numerical_cols:
                    self.scatter_y_combo.setCurrentText(current_y)
                elif len(numerical_cols) > 1:
                    self.scatter_y_combo.setCurrentIndex(1)
                elif len(numerical_cols) > 0:
                    self.scatter_y_combo.setCurrentIndex(0)
            
            # 4. Update Time-Series Column Dropdown
            if hasattr(self, 'ts_column_combo'):
                current_ts = self.ts_column_combo.currentText()
                self.ts_column_combo.clear()
                self.ts_column_combo.addItems(numerical_cols if numerical_cols else [])
                if current_ts in numerical_cols:
                    self.ts_column_combo.setCurrentText(current_ts)
                elif len(numerical_cols) > 0:
                    self.ts_column_combo.setCurrentIndex(0)
            
            # 5. Update FFT Spectrum Column Dropdown
            if hasattr(self, 'fft_column_combo'):
                current_fft = self.fft_column_combo.currentText()
                self.fft_column_combo.clear()
                self.fft_column_combo.addItems(numerical_cols if numerical_cols else [])
                if current_fft in numerical_cols:
                    self.fft_column_combo.setCurrentText(current_fft)
                elif len(numerical_cols) > 0:
                    self.fft_column_combo.setCurrentIndex(0)
                    
        except Exception as e:
            # Print the error to the console for debugging
            print(f"Error updating visualization dropdowns: {e}")

    # --- DB HANDLERS (Unchanged logic) ---
    def _check_db_manager(self):
        if not self.db_manager:
            QMessageBox.critical(self, "DB Error", "Database Manager is not initialized. Check your main.py setup.")
            return False
        return True
    def db_retrieve_data(self):
        if not self._check_db_manager(): return
        try:
            offset = self.current_page * self.rows_per_page
            
            # 1. Fetch fresh data from DB
            retrieved_df = self.db_manager.get_patient_data(limit=self.rows_per_page, offset=offset) 
            
            # 2. CRITICAL FIX: Overwrite self.df instead of appending/merging
            # This ensures the view starts fresh with only what is currently in the DB
            self.df = retrieved_df.copy()
            self.filtered_df = self.df.copy()
            
            # 3. Clear and Re-populate the table widget
            self.populate_table(self.df)
            
            # Update UI components
            self._update_viz_dropdowns()
            self._update_analysis_dropdowns()
            
            total_records = self.db_manager.get_total_count()
            total_pages = (total_records // self.rows_per_page) + (1 if total_records % self.rows_per_page > 0 else 0)
            self.page_label.setText(f"Page {self.current_page + 1} of {max(1, total_pages)}")
            
            self.prev_btn.setEnabled(self.current_page > 0)
            self.next_btn.setEnabled(offset + len(retrieved_df) < total_records)
            
            self.status_label.setText(f"Table Refreshed: Showing {len(retrieved_df)} records.")
            
        except Exception as e:
            QMessageBox.critical(self, "Refresh Error", f"Failed to retrieve data: {e}")

    def manual_db_insert(self):
        """Captures UI input, including medical images, and sends it to the relational database."""
        try:
            # 1. Basic validation
            name_val = self.name_input.text().strip()
            if not name_val:
                QMessageBox.warning(self, "Input Error", "Please enter a patient name.")
                return

            # 2. Extract values from UI widgets
            gender_val = self.gender_input.currentText()
            record_date = self.date_input.dateTime().toString("yyyy-MM-dd HH:mm:ss")

            # 3. Handle Image File Processing
            image_blob = None
            image_path = self.image_path_input.text().strip()
            
            if image_path and os.path.exists(image_path):
                try:
                    # Convert image file to raw bytes (BLOB)
                    with open(image_path, 'rb') as file:
                        image_blob = file.read()
                except Exception as e:
                    QMessageBox.warning(self, "Image Error", f"Could not read image file: {e}")
            
            # 4. Create the data dictionary for the DB Manager
            # Note: We include 'Name' and 'Gender' so the DB Manager can handle 
            # the relational link to the 'patients' table.
            report_data = {
                "Name": name_val,
                "Gender": gender_val,
                "Age": self.age_input.value(),
                "Blood_Pressure": self.bp_input.value(),
                "Cholesterol_Level": self.chol_input.value(),
                "BMI": self.bmi_input.value(),
                "Fasting_Blood_Sugar": self.sugar_input.value(),
                "Heart_Disease_Status": self.status_input.currentText(),
                "ECG_Signal": self.ecg_input.text().strip(),
                "EEG_Signal": self.eeg_input.text().strip(),
                "Date_Recorded": record_date,
                "Image_Data": image_blob  # This maps to the BLOB column
            }

            # 5. Insert via specialized manual record method
            success = self.db_manager.insert_manual_record(report_data)
            
            if success:
                # 6. UI Updates and Cleanup
                self.db_retrieve_data() # Refresh the table to show new entry
                
                # Clear fields for next entry
                self.name_input.clear()
                self.image_path_input.clear()
                self.ecg_input.clear()
                self.eeg_input.clear()
                
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"Added record for {name_val} with image.")
                
                QMessageBox.information(self, "Success", f"Patient record for {name_val} saved successfully.")
            else:
                QMessageBox.critical(self, "DB Error", "Failed to save record. Check database connection.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add patient record: {e}")

    def db_insert_data(self):
        if not self._check_db_manager(): return
        if self.df.empty:
            QMessageBox.warning(self, "Insert Error", "No data loaded to insert.")
            return

        try:
            self.db_manager.insert_patient_data(self.df)
            QMessageBox.information(self, "Success", f"Successfully inserted {len(self.df)} rows into the DB.")
        except Exception as e:
            QMessageBox.critical(self, "DB Insertion Error", f"Failed to insert data: {e}")
    def db_search_patient(self):
        """Filters the database view for a specific name or ID."""
        search_term = self.search_input.text().strip()
        if not search_term:
            QMessageBox.warning(self, "Input Error", "Please enter a Name or ID.")
            return

        # Logic to filter: 
        # If using local DataFrame (self.df):
        if hasattr(self, 'df'):
            # Check if ID (exact match) or Name (contains string)
            if search_term.isdigit():
                # Adjust column name to match your CSV/DB headers (e.g., 'patient_id')
                filtered = self.df[self.df.iloc[:, 0].astype(str) == search_term]
            else:
                # Adjust 'Name' to match your column name
                filtered = self.df[self.df['Name'].str.contains(search_term, case=False, na=False)]
            
            if not filtered.empty:
                self.populate_table(filtered)
                self.status_label.setText(f"Found {len(filtered)} records.")
            else:
                QMessageBox.information(self, "No Results", "No matching patient found.")
    def db_update_prompt(self):
        if not self._check_db_manager(): return

        # 1. Get the Patient ID from the input box
        patient_id_str = self.patient_id_input.text().strip()
        if not patient_id_str.isdigit():
            QMessageBox.warning(self, "Input Error", "Please enter a valid Patient ID (number) to update.")
            return
        patient_id = int(patient_id_str)
        
        # DYNAMIC FIX: Get all column names from the current dataframe
        # Filter out 'patient_id' and 'report_id' as they shouldn't be manually edited
        if self.df is not None and not self.df.empty:
            fields = [col for col in self.df.columns if 'id' not in col.lower()]
        else:
            QMessageBox.warning(self, "No Data", "No data available to determine column names.")
            return

        # 3. Ask the user which field they want to change
        field, ok = QInputDialog.getItem(self, "Update Record", "Select Field to Update:", fields, 0, False)
        
        if ok and field:
            # 4. Open a text input dialog for the new value
            new_value, ok = QInputDialog.getText(self, "Update Record", f"Enter new value for {field}:")
            
            if ok and new_value:
                try:
                    # Create the update dictionary
                    update_kwargs = {field: new_value}
                    
                    # Call the DB manager to execute the UPDATE SQL
                    self.db_manager.update_patient_data(patient_id, **update_kwargs)
                    
                    QMessageBox.information(self, "Success", f"Patient {patient_id} updated successfully.")
                    
                    # 5. Refresh the table to see changes immediately
                    self.db_retrieve_data()
                    
                except Exception as e:
                    QMessageBox.critical(self, "DB Update Error", f"Failed to update record: {e}")

    def db_delete_record(self):
        if not self._check_db_manager(): return

        patient_id_str = self.patient_id_input.text().strip()
        if not patient_id_str.isdigit():
            QMessageBox.warning(self, "Input Error", "Please enter a valid Patient ID (number) to delete.")
            return
        
        patient_id = int(patient_id_str)
        
        reply = QMessageBox.question(self, 'Confirm Deletion',
                                     f"Are you sure you want to PERMANENTLY delete Patient ID {patient_id}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                # Capture the rowcount returned by the manager
                rows_affected = self.db_manager.delete_patient_data(patient_id)
                
                if rows_affected > 0:
                    # SUCCESS: ID existed and was removed
                    QMessageBox.information(self, "Success", f"Patient ID {patient_id} deleted successfully.")
                    self.patient_id_input.clear()
                    self.db_retrieve_data() 
                elif rows_affected == 0:
                    # FAILURE: ID does not exist in the table
                    QMessageBox.warning(self, "Not Found", f"Delete failed: No patient found with ID {patient_id}.")
                else:
                    # ERROR: A database exception occurred (-1)
                    QMessageBox.critical(self, "Error", "A database error occurred during deletion.")
                
            except Exception as e:
                QMessageBox.critical(self, "DB Deletion Error", f"Failed to delete record: {e}")

# --- Panel 2: Analysis ---
    def create_analysis_panel(self):
        panel = QWidget()
        main_layout = QVBoxLayout(panel)

        if self.df is not None and not self.df.empty:
           numerical_cols = [col for col in self.df.select_dtypes(include=np.number).columns 
                              if 'ID' not in col.upper()]
        else:
            numerical_cols = ['Age', 'Blood_Pressure', 'Cholesterol_Level', 'BMI']

        title = QLabel("Health Data Analysis")
        title.setObjectName("PanelTitle")
        main_layout.addWidget(title)
        
        patient_sel_layout = QHBoxLayout()
        patient_sel_layout.addWidget(QLabel("Select Patient ID:"))
        self.analysis_patient_id = QComboBox()
        self.analysis_patient_id.setEditable(True)
        # Automatically refresh data when a new patient is selected
        self.analysis_patient_id.currentTextChanged.connect(self.update_analysis_for_patient)
        patient_sel_layout.addWidget(self.analysis_patient_id)
        
        self.refresh_patient_list_btn = QPushButton("Refresh ID List")
        self.refresh_patient_list_btn.setObjectName("refreshIdButton")
        self.refresh_patient_list_btn.clicked.connect(self._refresh_patient_ids)
        patient_sel_layout.addWidget(self.refresh_patient_list_btn)
        
        patient_sel_layout.addStretch()
        main_layout.addLayout(patient_sel_layout)

        controls_group = QHBoxLayout()

        filtering_widget = QWidget()
        filtering_widget.setObjectName("FilterGroup")
        filtering_group = QVBoxLayout(filtering_widget)
        
        filter_header = QLabel("Data Filtering & Smoothing")
        filtering_group.addWidget(filter_header)

        ma_layout = QHBoxLayout()
        ma_layout.addWidget(QLabel("MA Window:"))
        self.ma_window = QSpinBox()
        self.ma_window.setRange(2, 100)
        self.ma_window.setValue(5)
        ma_layout.addWidget(self.ma_window)
        filtering_group.addLayout(ma_layout)

        thresh_ctrl_layout = QHBoxLayout()
        thresh_ctrl_layout.addWidget(QLabel("Min Threshold:"))
        self.thresh_val = QDoubleSpinBox()
        self.thresh_val.setRange(0, 5000)
        thresh_ctrl_layout.addWidget(self.thresh_val)
        filtering_group.addLayout(thresh_ctrl_layout)
        
        filtering_group.addWidget(QLabel("Outlier Removal (IQR Factor):"))
        self.outlier_slider = QSlider(Qt.Horizontal)
        self.outlier_slider.setRange(10, 50) 
        self.outlier_slider.setValue(15) 
        filtering_group.addWidget(self.outlier_slider)

        filtering_group.addWidget(QLabel("Time-series Column:"))
        ts_controls = QHBoxLayout()
        self.ts_analysis_column = QComboBox()
        self.ts_analysis_column.addItems(numerical_cols)
        self.ts_analysis_column.setObjectName("AnalysisTSCombo")
        self.ts_raw_checkbox = QCheckBox("Show Raw")
        self.ts_raw_checkbox.setChecked(True)
        ts_controls.addWidget(self.ts_analysis_column)
        ts_controls.addWidget(self.ts_raw_checkbox)
        filtering_group.addLayout(ts_controls)
        
        self.apply_filter_btn = QPushButton("Apply & Plot")
        self.apply_filter_btn.setObjectName("ApplyPlotButton")
        self.apply_filter_btn.clicked.connect(self.apply_filters_and_plot)
        self.reset_filter_btn = QPushButton("Reset")
        self.reset_filter_btn.setObjectName("ResetButton")
        self.reset_filter_btn.clicked.connect(self.reset_filters)
        
        filter_btn_layout = QHBoxLayout()
        filter_btn_layout.addWidget(self.apply_filter_btn)
        filter_btn_layout.addWidget(self.reset_filter_btn)
        filtering_group.addLayout(filter_btn_layout)
        
        controls_group.addWidget(filtering_widget)
        
        corr_widget = QWidget()
        corr_widget.setObjectName("CorrGroup")
        corr_group = QVBoxLayout(corr_widget)
        corr_header = QLabel("Correlation Analysis")
        corr_group.addWidget(corr_header)
        
        self.metric1_dropdown = QComboBox()
        self.metric1_dropdown.addItems(numerical_cols if numerical_cols else [])
        self.metric1_dropdown.setObjectName("AnalysisMetric1Combo")
        self.metric2_dropdown = QComboBox()
        self.metric2_dropdown.addItems(numerical_cols if numerical_cols else [])
        self.metric2_dropdown.setObjectName("AnalysisMetric2Combo")
        corr_group.addWidget(QLabel("Metric 1:"))
        corr_group.addWidget(self.metric1_dropdown)
        corr_group.addWidget(QLabel("Metric 2:"))
        corr_group.addWidget(self.metric2_dropdown)

        self.compute_corr_btn = QPushButton("Scatter Plot")
        self.compute_corr_btn.setObjectName("ScatterPlotButton")
        self.compute_corr_btn.clicked.connect(self.compute_correlation_plot)
        self.show_heatmap_btn = QPushButton("Heatmap")
        self.show_heatmap_btn.setObjectName("HeatmapButton")
        self.show_heatmap_btn.clicked.connect(self.show_heatmap)
        
        corr_btn_layout = QHBoxLayout()
        corr_btn_layout.addWidget(self.compute_corr_btn)
        corr_btn_layout.addWidget(self.show_heatmap_btn)
        corr_group.addLayout(corr_btn_layout)
        
        controls_group.addWidget(corr_widget)
        main_layout.addLayout(controls_group)

        self.analysis_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        self.analysis_canvas.setMinimumHeight(700) 

        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setWidget(self.analysis_canvas)
        main_layout.addWidget(analysis_scroll)
        
        self.analysis_ax = self.analysis_canvas.figure.add_subplot(111)

        self.analysis_status_label = QLabel("Ready for analysis.")
        main_layout.addWidget(self.analysis_status_label)

        return panel

    def _refresh_patient_ids(self):
        """Populates the dropdown with available Patient IDs from the main dataframe."""
        if self.df is not None and 'patient_id' in self.df.columns:
            ids = sorted(self.df['patient_id'].unique().astype(str))
            self.analysis_patient_id.clear()
            self.analysis_patient_id.addItems(ids)

    def update_analysis_for_patient(self):
        """Filters the active dataset to only include the selected patient."""
        selected_id = self.analysis_patient_id.currentText()
        if not selected_id or self.df is None:
            return

        try:
            # Filter global dataframe for the specific patient
            self.patient_df = self.df[self.df['patient_id'].astype(str) == selected_id].copy()
            
            if self.patient_df.empty:
                self.analysis_status_label.setText(f"No records found for Patient ID: {selected_id}")
            else:
                self.analysis_status_label.setText(f"Loaded {len(self.patient_df)} records for Patient ID: {selected_id}")
                # Automatically update plots if needed
        except Exception as e:
            print(f"Filtering error: {e}")

    def apply_filters_and_plot(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data before applying filters.")
            return

        # --- NEW: Filter by selected Patient ID if available ---
        selected_id = self.analysis_patient_id.currentText().strip()
        if selected_id and 'patient_id' in self.df.columns:
            # Filter global df to only include the specific patient's history
            target_df = self.df[self.df['patient_id'].astype(str) == selected_id].copy()
            patient_title = f" (Patient: {selected_id})"
        else:
            target_df = self.df.copy()
            patient_title = ""

        if target_df.empty:
            QMessageBox.warning(self, "No Data", "No records found for the selected patient.")
            return

        factor = self.outlier_slider.value() / 10.0
        ma_window = self.ma_window.value()
        col = self.ts_analysis_column.currentText()
        
        # Apply filtering logic to the (potentially single-patient) dataset
        self.filtered_df = target_df.copy()

        Q1 = self.filtered_df[col].quantile(0.25)
        Q3 = self.filtered_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        self.filtered_df = self.filtered_df[
            (self.filtered_df[col] >= lower_bound) & 
            (self.filtered_df[col] <= upper_bound)
        ]

        self.filtered_df['MA'] = self.filtered_df[col].rolling(window=ma_window, min_periods=1).mean()

        self.analysis_ax.clear()
        
        if self.ts_raw_checkbox.isChecked():
            # Plot raw history for the specific patient
            self.analysis_ax.plot(target_df[col].values, color='#BDC3C7', alpha=0.4, label='Raw History')
        
        self.analysis_ax.plot(self.filtered_df['MA'].values, color='#3498DB', linewidth=2, label='Filtered Trend')
        
        self.analysis_ax.set_title(f"Analysis: {col}{patient_title}")
        self.analysis_ax.set_xlabel("Visit / Record Index")
        self.analysis_ax.set_ylabel("Value")
        self.analysis_ax.legend()
        self.analysis_ax.grid(True, linestyle='--', alpha=0.5)
        
        self.analysis_ax.set_box_aspect(0.6)
        self.analysis_canvas.figure.tight_layout()
        self.analysis_canvas.draw()
        
        self.analysis_status_label.setText(
            f"Rows: {len(self.filtered_df)} | Patient Focus: {selected_id if selected_id else 'All'}"
        )
        
    def reset_filters(self):
        # Reset underlying data
        self.filtered_df = self.df.copy()

        # Reset controls to their initial defaults
        self.outlier_slider.setValue(15)
        self.ma_window.setValue(5)
        self.thresh_val.setValue(0.0)

        if self.df.select_dtypes(include='number').columns.size > 0:
            # Reset time-series column selector
            self.ts_analysis_column.setCurrentIndex(0)

            # Reset correlation metric dropdowns
            self.metric1_dropdown.setCurrentIndex(0)
            self.metric2_dropdown.setCurrentIndex(0)

        # Reset checkbox state
        self.ts_raw_checkbox.setChecked(True)

        # Clear plot and status label back to default
        self.analysis_canvas.figure.clear()
        self.analysis_ax = self.analysis_canvas.figure.add_subplot(111)
        self.analysis_canvas.draw()
        self.analysis_status_label.setText("All filters reset. Ready for analysis.")

    def compute_correlation_plot(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data before computing correlation.")
            return

        col1 = self.metric1_dropdown.currentText()
        col2 = self.metric2_dropdown.currentText()
        
        if not col1 or not col2:
            QMessageBox.warning(self, "Invalid Selection", "Please select both metrics for correlation analysis.")
            return

        # --- SINGLE PATIENT LOGIC START ---
        # Get selected patient ID from the input field
        selected_id = self.analysis_patient_id.currentText().strip()
        
        if selected_id and 'patient_id' in self.df.columns:
            # Filter specifically for the chosen patient to analyze their unique trends
            working_df = self.df[self.df['patient_id'].astype(str) == selected_id].copy()
            analysis_scope = f"Patient {selected_id}"
        else:
            # Fallback to the current filtered view or full dataset
            working_df = self.filtered_df if (self.filtered_df is not None and not self.filtered_df.empty) else self.df
            analysis_scope = "Global Dataset"
        # --- SINGLE PATIENT LOGIC END ---

        if working_df.empty:
            QMessageBox.warning(self, "No Data", f"No records found for {analysis_scope}.")
            return

        if col1 not in working_df.columns or col2 not in working_df.columns:
            available_cols = list(working_df.columns)
            QMessageBox.warning(self, "Invalid Columns", 
                              f"One or both columns ('{col1}', '{col2}') do not exist in the dataset.\n"
                              f"Available columns: {', '.join(available_cols[:10])}")
            return

        try:
            if not pd.api.types.is_numeric_dtype(working_df[col1]) or not pd.api.types.is_numeric_dtype(working_df[col2]):
                QMessageBox.warning(self, "Invalid Columns", "Both columns must be numeric for correlation analysis.")
                return

            valid_data = working_df[[col1, col2]].dropna()
            
            # Check if there are enough points for a correlation (need at least 2)
            if len(valid_data) < 2:
                QMessageBox.warning(self, "Insufficient Data", f"Not enough data points for {analysis_scope} to compute correlation.")
                return

            self.analysis_ax.clear()
            
            # Use a colored scatter plot for better visibility in analysis
            valid_data.plot.scatter(x=col1, y=col2, ax=self.analysis_ax, color='#2ECC71', alpha=0.7)
            
            corr_value = valid_data[col1].corr(valid_data[col2])
            
            # Update title to reflect the single patient focus
            self.analysis_ax.set_title(f'{analysis_scope}: {col1} vs {col2}\n(Correlation = {corr_value:.2f})')
            self.analysis_ax.grid(True, linestyle='--', alpha=0.6)
            
            self.analysis_canvas.draw()
            self.analysis_status_label.setText(f"Scope: {analysis_scope} | Corr: {corr_value:.2f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compute correlation: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_heatmap(self):
        """Displays a density heatmap between the two selected metrics with a trend line."""
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        # Get specific metric selections
        col1 = self.metric1_dropdown.currentText()
        col2 = self.metric2_dropdown.currentText()

        if not col1 or not col2 or col1 == col2:
            QMessageBox.warning(self, "Selection Error", "Please select two different metrics to correlate.")
            return

        # --- SINGLE PATIENT LOGIC START ---
        # Get selected patient ID from the input field
        selected_id = self.analysis_patient_id.currentText().strip()
        
        if selected_id and 'patient_id' in self.df.columns:
            # Filter specifically for the chosen patient's history
            source_df = self.df[self.df['patient_id'].astype(str) == selected_id].copy()
            analysis_scope = f"Patient {selected_id}"
        else:
            # Fallback to the global dataset
            source_df = self.df.copy()
            analysis_scope = "Global Dataset"
        # --- SINGLE PATIENT LOGIC END ---

        try:
            # Clear the previous plot completely
            self.analysis_canvas.figure.clear()
            self.analysis_ax = self.analysis_canvas.figure.add_subplot(111)

            # Extract data and drop missing values from the chosen source
            plot_df = source_df[[col1, col2]].dropna()

            if plot_df.empty:
                QMessageBox.warning(self, "No Data", f"No matching data points found for {analysis_scope}.")
                return
            
            # For a single patient, a heatmap requires a bit more data to look meaningful
            if len(plot_df) < 3:
                QMessageBox.warning(self, "Insufficient History", 
                                  f"{analysis_scope} only has {len(plot_df)} records. "
                                  "Need at least 3 records to generate a density heatmap.")
                return

            # --- DENSITY HEATMAP (Hexbin) ---
            # gridsize controls the hexagon size; for single patient data, we use a smaller grid
            grid_size = 12 if len(plot_df) < 20 else 18
            
            hb = self.analysis_ax.hexbin(
                plot_df[col1], 
                plot_df[col2], 
                gridsize=grid_size, 
                cmap='YlOrRd', 
                mincnt=1,
                alpha=0.8
            )
            
            # Add a colorbar
            cb = self.analysis_canvas.figure.colorbar(hb, ax=self.analysis_ax)
            cb.set_label('Record Frequency (Count)')

            # --- REGRESSION LINE (Trend Line) ---
            sns.regplot(
                x=col1, 
                y=col2, 
                data=plot_df, 
                ax=self.analysis_ax, 
                scatter=False,  # Use Hexbin for density, hide dots
                color='#2980B9', # Professional blue trend line
                line_kws={'linewidth': 2.5, 'label': 'Linear Trend'}
            )

            # Calculate correlation coefficient for this specific scope
            corr_val = plot_df[col1].corr(plot_df[col2])

            # Formatting the plot
            self.analysis_ax.set_title(f"{analysis_scope}: {col1} vs {col2}\nCorrelation (r) = {corr_val:.2f}", 
                                      fontsize=11, fontweight='bold')
            self.analysis_ax.set_xlabel(col1, fontweight='bold')
            self.analysis_ax.set_ylabel(col2, fontweight='bold')
            self.analysis_ax.legend()
            self.analysis_ax.grid(True, linestyle=':', alpha=0.5)

            self.analysis_canvas.figure.tight_layout()
            self.analysis_canvas.draw()
            self.analysis_status_label.setText(f"Heatmap generated for {analysis_scope}.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Heatmap failed: {str(e)}")

    def create_spectrum_panel(self):
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        main_layout.setSpacing(10)

        title = QLabel("Spectrum Analysis")
        title.setObjectName("PanelTitle")
        main_layout.addWidget(title)

        # --- Patient Selection Header ---
        patient_header = QGroupBox("Single Patient Focus")
        patient_header_layout = QHBoxLayout(patient_header)
        
        patient_header_layout.addWidget(QLabel("Select Patient ID:"))
        self.spectrum_patient_id = QComboBox()
        self.spectrum_patient_id.setEditable(True)
        # Connect to the update logic
        self.spectrum_patient_id.currentTextChanged.connect(self._update_spectrum_for_patient)
        patient_header_layout.addWidget(self.spectrum_patient_id, 1)
        
        refresh_ids_btn = QPushButton("Refresh List")
        refresh_ids_btn.clicked.connect(self._refresh_spectrum_ids)
        patient_header_layout.addWidget(refresh_ids_btn)
        
        main_layout.addWidget(patient_header) # Use main_layout, not layout
        
        # --- Signal Loading & Selection Group ---
        signal_loading_group = QGroupBox("Signal Loading & Selection")
        signal_loading_layout = QHBoxLayout(signal_loading_group)
        signal_loading_layout.setSpacing(15)
        
        signal_loading_layout.addWidget(QLabel("Biomedical Signal:"))
        self.signal_dropdown = QComboBox()
        self.signal_dropdown.setObjectName("BiomedicalSignalDropdown")
        
        # Define signal targets
        biomed_targets = ["ECG", "EEG"]
        biomed_options = []

        if self.df is not None and not self.df.empty:
            # Check which of the target signals exist in the dataframe (case-insensitive)
            biomed_options = [col for col in self.df.columns if any(t in col.upper() for t in biomed_targets)]
        
        # If the dataset doesn't have them, use defaults so the app doesn't crash
        if not biomed_options:
            biomed_options = biomed_targets
            
        # Now this line will work because 'biomed_options' is defined above
        self.signal_dropdown.addItems(biomed_options)
        # --- END OF FIX ---

        self.signal_dropdown.setMinimumWidth(200)
        signal_loading_layout.addWidget(self.signal_dropdown)

        self.signal_dropdown.addItems(biomed_options)
        self.signal_dropdown.setMinimumWidth(200)
        
        self.plot_signal_btn = QPushButton("Load & Display Signal")
        self.plot_signal_btn.setObjectName("LoadDisplaySignalButton")
        self.plot_signal_btn.setToolTip("Loads and displays the selected signal in a scrollable plot.")
        self.plot_signal_btn.clicked.connect(self.plot_raw_signal)
        signal_loading_layout.addWidget(self.plot_signal_btn)
        signal_loading_layout.addStretch()
        
        main_layout.addWidget(signal_loading_group)

        raw_signal_group = QGroupBox("Raw Signal Display (Scrollable)")
        raw_signal_layout = QVBoxLayout(raw_signal_group)
        
        self.raw_signal_canvas = FigureCanvas(Figure(figsize=(12, 4)))
        self.raw_signal_canvas.setMinimumHeight(300)
        raw_signal_scroll = QScrollArea()
        raw_signal_scroll.setWidgetResizable(True)
        raw_signal_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        raw_signal_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        raw_signal_scroll.setWidget(self.raw_signal_canvas)
        raw_signal_layout.addWidget(raw_signal_scroll)
        self.raw_signal_ax = self.raw_signal_canvas.figure.add_subplot(111)
        
        main_layout.addWidget(raw_signal_group)

        fft_controls_group = QGroupBox("FFT Spectrum Analysis")
        fft_controls_layout = QVBoxLayout(fft_controls_group)
        
        segment_controls = QHBoxLayout()
        segment_controls.setSpacing(20)
        
        start_segment_layout = QVBoxLayout()
        start_segment_layout.addWidget(QLabel("Segment Start (Sample):"))
        self.segment_start_slider = QSlider(Qt.Horizontal)
        self.segment_start_slider.setRange(0, 100)
        self.segment_start_slider.setValue(0)
        self.segment_start_slider.setSingleStep(1)
        self.segment_start_slider.setToolTip("Select the starting sample index for FFT analysis")
        self.segment_start_label = QLabel("0")
        self.segment_start_label.setMinimumWidth(50)
        self.segment_start_slider.valueChanged.connect(lambda v: self.segment_start_label.setText(str(v)))
        start_segment_layout.addWidget(self.segment_start_slider)
        start_segment_layout.addWidget(self.segment_start_label)
        segment_controls.addLayout(start_segment_layout)
        
        end_segment_layout = QVBoxLayout()
        end_segment_layout.addWidget(QLabel("Segment End (Sample):"))
        self.segment_end_slider = QSlider(Qt.Horizontal)
        self.segment_end_slider.setRange(10, 200)
        self.segment_end_slider.setValue(100)
        self.segment_end_slider.setSingleStep(1)
        self.segment_end_slider.setToolTip("Select the ending sample index for FFT analysis")
        self.segment_end_label = QLabel("100")
        self.segment_end_label.setMinimumWidth(50)
        self.segment_end_slider.valueChanged.connect(lambda v: self.segment_end_label.setText(str(v)))
        end_segment_layout.addWidget(self.segment_end_slider)
        end_segment_layout.addWidget(self.segment_end_label)
        segment_controls.addLayout(end_segment_layout)
        
        fft_controls_layout.addLayout(segment_controls)
        
        fft_button_layout = QHBoxLayout()
        self.fft_btn = QPushButton("Compute FFT Spectrum")
        self.fft_btn.setObjectName("ComputeFFTButton")
        self.fft_btn.setToolTip("Calculates and displays the Fast Fourier Transform (Power Spectrum) for the selected segment.")
        self.fft_btn.clicked.connect(self.plot_fft)
        fft_button_layout.addWidget(self.fft_btn)
        fft_button_layout.addStretch()
        fft_controls_layout.addLayout(fft_button_layout)
        
        main_layout.addWidget(fft_controls_group)

        visualization_controls_group = QGroupBox("Visualization Controls")
        viz_controls_layout = QHBoxLayout(visualization_controls_group)
        viz_controls_layout.setSpacing(15)
        
        freq_range_layout = QVBoxLayout()
        freq_range_layout.addWidget(QLabel("Frequency Range (Hz):"))
        freq_range_hbox = QHBoxLayout()
        freq_range_hbox.addWidget(QLabel("Min:"))
        self.freq_min_input = QDoubleSpinBox()
        self.freq_min_input.setRange(0, 1000)
        self.freq_min_input.setValue(0)
        self.freq_min_input.setDecimals(2)
        self.freq_min_input.setSingleStep(0.1)
        freq_range_hbox.addWidget(self.freq_min_input)
        freq_range_hbox.addWidget(QLabel("Max:"))
        self.freq_max_input = QDoubleSpinBox()
        self.freq_max_input.setRange(0, 1000)
        self.freq_max_input.setValue(1.0)
        self.freq_max_input.setDecimals(2)
        self.freq_max_input.setSingleStep(0.1)
        freq_range_hbox.addWidget(self.freq_max_input)
        freq_range_layout.addLayout(freq_range_hbox)
        viz_controls_layout.addLayout(freq_range_layout)
        
        amp_range_layout = QVBoxLayout()
        amp_range_layout.addWidget(QLabel("Amplitude Range:"))
        amp_range_hbox = QHBoxLayout()
        amp_range_hbox.addWidget(QLabel("Min:"))
        self.amp_min_input = QDoubleSpinBox()
        self.amp_min_input.setRange(-1000, 1000)
        self.amp_min_input.setValue(0)
        self.amp_min_input.setDecimals(2)
        amp_range_hbox.addWidget(self.amp_min_input)
        amp_range_hbox.addWidget(QLabel("Max:"))
        self.amp_max_input = QDoubleSpinBox()
        self.amp_max_input.setRange(-1000, 1000)
        self.amp_max_input.setValue(100)
        self.amp_max_input.setDecimals(2)
        amp_range_hbox.addWidget(self.amp_max_input)
        amp_range_layout.addLayout(amp_range_hbox)
        viz_controls_layout.addLayout(amp_range_layout)
        
        self.apply_zoom_btn = QPushButton("Apply Zoom")
        self.apply_zoom_btn.setObjectName("ApplyZoomButton")
        self.apply_zoom_btn.setToolTip("Apply the specified axis limits to zoom into the frequency spectrum.")
        self.apply_zoom_btn.clicked.connect(self.apply_fft_zoom)
        viz_controls_layout.addWidget(self.apply_zoom_btn)
        
        self.reset_zoom_btn = QPushButton("Reset View")
        self.reset_zoom_btn.setObjectName("ResetViewButton")
        self.reset_zoom_btn.setToolTip("Reset the view to show the full spectrum.")
        self.reset_zoom_btn.clicked.connect(self.reset_fft_zoom)
        viz_controls_layout.addWidget(self.reset_zoom_btn)
        
        main_layout.addWidget(visualization_controls_group)

        fft_display_group = QGroupBox("FFT Power Spectrum")
        fft_display_layout = QVBoxLayout(fft_display_group)
        
        self.spectrum_canvas = FigureCanvas(Figure(figsize=(12, 5)))
        self.spectrum_canvas.setMinimumHeight(350)
        fft_display_layout.addWidget(self.spectrum_canvas)
        self.spectrum_ax = self.spectrum_canvas.figure.add_subplot(111)
        
        main_layout.addWidget(fft_display_group)

        return panel
        
    def _refresh_spectrum_ids(self):
        """Populates the spectrum ID list from the main dataframe."""
        if self.df is not None and 'patient_id' in self.df.columns:
            # Get unique IDs, sort them, and convert to string for the ComboBox
            ids = sorted(self.df['patient_id'].unique().astype(str))
            self.spectrum_patient_id.clear()
            self.spectrum_patient_id.addItems(ids)

    def _update_spectrum_for_patient(self):
        """Triggered when the patient ID selection changes."""
        # Optional: Clear the graph when changing patients
        self.figure.clear()
        self.canvas.draw()

    def _get_selected_patient_signal(self, col_name):
        """Helper to extract signal data for the specific patient selected."""
        selected_id = self.spectrum_patient_id.currentText().strip()
        if not selected_id or self.df is None:
            return None

        # Filter global df for the specific patient
        patient_data = self.df[self.df['patient_id'].astype(str) == selected_id]
        
        if patient_data.empty:
            return None
            
        # Extract signal string from the most recent record
        signal_str = str(patient_data.iloc[-1][col_name])
        try:
            # Parse CSV string into numeric array
            return np.array([float(x) for x in signal_str.split(',') if x.strip()])
        except Exception:
            return None

    def plot_raw_signal(self):
        # 1. Check if global data exists
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load a dataset containing ECG/EEG data first.")
            return

        # 2. Get the selected Patient ID
        selected_patient_id = self.spectrum_patient_id.currentText().strip()
        if not selected_patient_id:
            QMessageBox.warning(self, "Selection Required", "Please select or enter a Patient ID first.")
            return

        # 3. Get the signal column selection
        col = self.signal_dropdown.currentText()
        if not col:
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid Biomedical Signal column.")
            return

        try:
            # 4. Filter data for the SPECIFIC patient
            patient_data = self.df[self.df['patient_id'].astype(str) == selected_patient_id]
            
            if patient_data.empty:
                QMessageBox.warning(self, "No Patient Found", f"No records found for Patient ID: {selected_patient_id}")
                return

            # Get the signal data from the most recent record
            raw_signal_data = patient_data.iloc[-1][col]

            # 5. Handle Signal Conversion (FIXED PARSING)
            if isinstance(raw_signal_data, str):
                # Strip brackets/quotes if they exist, then split by comma
                clean_str = raw_signal_data.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
                signal = np.array([float(x) for x in clean_str.split(',') if x.strip()])
            else:
                signal = np.atleast_1d(raw_signal_data).astype(float)

            if len(signal) == 0:
                QMessageBox.warning(self, "Empty Signal", f"The signal for patient {selected_patient_id} contains no data.")
                return

            # 6. UI Plotting (FIXED SCALING)
            self.raw_signal_ax.clear()
            
            # Use a thinner line (0.8) to make the wave look more detailed
            self.raw_signal_ax.plot(signal, color='#27AE60', linewidth=0.8, label="Raw Signal")
            
            # IMPORTANT: Set the X-limit to the actual length of the data
            # This prevents the "single line" compression issue
            self.raw_signal_ax.set_xlim(0, len(signal))
            
            self.raw_signal_ax.set_title(f"Patient {selected_patient_id}: Raw {col} Signal", fontsize=12, fontweight='bold')
            self.raw_signal_ax.set_xlabel("Samples (Time)", fontsize=10)
            self.raw_signal_ax.set_ylabel("Amplitude", fontsize=10)
            self.raw_signal_ax.grid(True, linestyle=':', alpha=0.6)
            
            self.raw_signal_canvas.figure.tight_layout()
            self.raw_signal_canvas.draw()
            
            # 7. Update Sliders
            max_samples = len(signal)
            self.segment_start_slider.setRange(0, max_samples - 2)
            self.segment_end_slider.setRange(2, max_samples)
            
            # Reset sliders to show a reasonable starting window (e.g., first 1000 samples)
            default_end = min(1000, max_samples)
            self.segment_start_slider.setValue(0)
            self.segment_end_slider.setValue(default_end)
            
            if hasattr(self, 'segment_end_label'):
                self.segment_end_label.setText(str(default_end))

        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"An error occurred: {str(e)}")

    def plot_fft(self):
        """Triggered by 'Compute FFT Spectrum' button: Shows the CLEANED Signal."""
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data before computing FFT.")
            return

        selected_patient_id = self.spectrum_patient_id.currentText().strip()
        if not selected_patient_id:
            QMessageBox.warning(self, "Selection Required", "Please select a Patient ID.")
            return

        col = self.signal_dropdown.currentText()
        if not col or not any(key in col.upper() for key in ['ECG', 'EEG']):
            QMessageBox.warning(self, "Invalid Signal", "FFT Analysis is for ECG/EEG Signals.")
            return

        try:
            # 1. Filter data for the SPECIFIC patient
            patient_data = self.df[self.df['patient_id'].astype(str) == selected_patient_id]
            if patient_data.empty:
                QMessageBox.warning(self, "No Patient", f"No records found for Patient {selected_patient_id}")
                return

            # 2. Get signal and convert to numeric array (WITH STRING CLEANING)
            raw_signal_data = patient_data.iloc[-1][col]
            if isinstance(raw_signal_data, str):
                # Strip brackets/quotes that often break the split logic
                clean_str = raw_signal_data.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
                signal = np.array([float(x) for x in clean_str.split(',') if x.strip()])
            else:
                signal = np.atleast_1d(raw_signal_data).astype(float)

            # 3. Apply Segment logic from your sliders
            start_idx = self.segment_start_slider.value()
            end_idx = self.segment_end_slider.value()
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(signal), end_idx)
            
            signal_segment = signal[start_idx:end_idx]

            if len(signal_segment) < 2:
                QMessageBox.warning(self, "Range Error", "Selected segment is too small.")
                return

            # 4. Use the cleaning function from data_analyzer.py
            # This performs FFT -> Filter -> Inverse FFT to reconstruct a clean wave
            from data_analyzer import fft_denoise_signal
            cleaned_signal = fft_denoise_signal(signal_segment, threshold_percent=0.1)

            # 5. UI Plotting (Time Domain reconstruction)
            # Use self.raw_signal_ax if you want the cleaned signal to replace the raw one on the same graph
            # Or use self.spectrum_ax if you have a separate graph area for FFT
            self.spectrum_ax.clear()
            
            # Plot the Cleaned Signal in Red/Orange to distinguish from Raw Green
            self.spectrum_ax.plot(cleaned_signal, color='#E74C3C', label='FFT Cleaned Wave', linewidth=1.2)
            
            # IMPORTANT: Set the X-limit to the length of the segment to prevent "single line" view
            self.spectrum_ax.set_xlim(0, len(cleaned_signal))
            
            self.spectrum_ax.set_title(f"Cleaned Signal: Patient {selected_patient_id} ({col})", 
                                      fontsize=12, fontweight='bold')
            self.spectrum_ax.set_xlabel("Samples (Time)")
            self.spectrum_ax.set_ylabel("Amplitude")
            self.spectrum_ax.legend()
            self.spectrum_ax.grid(True, linestyle='--', alpha=0.5)
            
            self.spectrum_canvas.figure.tight_layout()
            self.spectrum_canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clean signal: {str(e)}")

    def apply_fft_zoom(self):
        if hasattr(self, 'spectrum_ax') and len(self.spectrum_ax.lines) > 0:
            try:
                freq_min = self.freq_min_input.value()
                freq_max = self.freq_max_input.value()
                amp_min = self.amp_min_input.value()
                amp_max = self.amp_max_input.value()
                
                if freq_min >= freq_max:
                    QMessageBox.warning(self, "Invalid Range", "Frequency min must be less than max.")
                    return
                if amp_min >= amp_max:
                    QMessageBox.warning(self, "Invalid Range", "Amplitude min must be less than max.")
                    return
                
                self.spectrum_ax.set_xlim(freq_min, freq_max)
                self.spectrum_ax.set_ylim(amp_min, amp_max)
                self.spectrum_canvas.draw()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to apply zoom: {str(e)}")
        else:
            QMessageBox.warning(self, "No Plot", "Please compute FFT spectrum first.")

    def reset_fft_zoom(self):
        if hasattr(self, 'spectrum_ax') and len(self.spectrum_ax.lines) > 0:
            try:
                self.spectrum_ax.relim()
                self.spectrum_ax.autoscale()
                self.spectrum_canvas.draw()
                
                freq_min, freq_max = self.spectrum_ax.get_xlim()
                amp_min, amp_max = self.spectrum_ax.get_ylim()
                
                self.freq_min_input.setValue(freq_min)
                self.freq_max_input.setValue(freq_max)
                self.amp_min_input.setValue(amp_min)
                self.amp_max_input.setValue(amp_max)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to reset zoom: {str(e)}")
        else:
            QMessageBox.warning(self, "No Plot", "Please compute FFT spectrum first.")

    # --- Panel 4: Image Processing ---
    def create_image_processing_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. Title
        title = QLabel("Medical Image Processing & Database Sync")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        # 2. DB Fetch Section (Horizontal Bar)
        db_fetch_layout = QHBoxLayout()
        db_fetch_group = QGroupBox("Fetch From Database")
        db_fetch_inner = QHBoxLayout(db_fetch_group)

        self.id_fetch_input = QLineEdit()
        self.id_fetch_input.setPlaceholderText("Enter Patient ID...")
        self.id_fetch_input.setFixedWidth(150)

        self.load_db_img_btn = QPushButton("Fetch Latest Image")
        self.load_db_img_btn.setObjectName("FetchImageButton")
        self.load_db_img_btn.clicked.connect(self.load_image_by_patient_id)

        db_fetch_inner.addWidget(QLabel("Patient ID:"))
        db_fetch_inner.addWidget(self.id_fetch_input)
        db_fetch_inner.addWidget(self.load_db_img_btn)
        db_fetch_inner.addStretch()
        
        layout.addWidget(db_fetch_group)

        # 3. Processing Controls Row
        controls_layout = QHBoxLayout()
        
        # Upload from Local File
        self.load_image_btn = QPushButton("Upload Local File")
        self.load_image_btn.setObjectName("UploadImageButton")
        self.load_image_btn.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_image_btn)

        # Grayscale
        self.grayscale_btn = QPushButton("Grayscale")
        self.grayscale_btn.clicked.connect(self.convert_to_grayscale)
        controls_layout.addWidget(self.grayscale_btn)

        # Smoothing Filters
        blur_layout = QVBoxLayout()
        blur_layout.addWidget(QLabel("Smoothing:"))
        self.blur_dropdown = QComboBox()
        self.blur_dropdown.addItems(["Gaussian Blur (15x15)", "Median Filter (5x5)"])
        blur_layout.addWidget(self.blur_dropdown)
        self.apply_blur_btn = QPushButton("Apply Filter")
        self.apply_blur_btn.clicked.connect(self.apply_blur)
        blur_layout.addWidget(self.apply_blur_btn)
        controls_layout.addLayout(blur_layout)
        
        # Edges
        self.edge_btn = QPushButton("Edge Detection")
        self.edge_btn.clicked.connect(self.apply_edge_detection)
        controls_layout.addWidget(self.edge_btn)

        # Thresholding
        thresh_layout = QVBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold (0-255):"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(127)
        thresh_layout.addWidget(self.threshold_slider)
        self.apply_threshold_btn = QPushButton("Apply Threshold")
        self.apply_threshold_btn.clicked.connect(self.apply_threshold)
        thresh_layout.addWidget(self.apply_threshold_btn)
        controls_layout.addLayout(thresh_layout)
        
        layout.addLayout(controls_layout)

        # 4. Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # 5. Image Comparison Area (Side-by-Side)
        self.image_layout = QHBoxLayout()
        
        # Original Image
        original_container = QWidget()
        orig_v_layout = QVBoxLayout(original_container)
        orig_v_layout.addWidget(QLabel("Original / DB Image", alignment=Qt.AlignCenter))
        self.original_image_label = QLabel("No image loaded")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setFixedSize(500, 500)
        self.original_image_label.setStyleSheet("border: 2px dashed #7f8c8d; background: #2c3e50;")
        orig_v_layout.addWidget(self.original_image_label)
        self.image_layout.addWidget(original_container)

        # Processed Image
        processed_container = QWidget()
        proc_v_layout = QVBoxLayout(processed_container)
        proc_v_layout.addWidget(QLabel("Processed Result", alignment=Qt.AlignCenter))
        self.processed_image_label = QLabel("Result will appear here")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setFixedSize(500, 500)
        self.processed_image_label.setStyleSheet("border: 2px solid #27ae60; background: #2c3e50;")
        proc_v_layout.addWidget(self.processed_image_label)
        self.image_layout.addWidget(processed_container)
        
        layout.addLayout(self.image_layout)

        # 6. Save Back to DB Button
        self.btn_save_processed = QPushButton("Save Processed Image to Patient Record")
        self.btn_save_processed.setMinimumHeight(50)
        self.btn_save_processed.setObjectName("SaveProcessedBtn")
        # Ensure this color matches your theme, or use your CSS file
        self.btn_save_processed.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        self.btn_save_processed.clicked.connect(self.save_processed_image_to_db)
        layout.addWidget(self.btn_save_processed)

        # 7. Reset Button
        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.clicked.connect(self.reset_image_view)
        layout.addWidget(self.btn_reset)

        return panel
    
    def load_image_for_analysis(self, report_id):
        """Retrieves BLOB from DB and displays in the 'Original Image' container."""
        # 1. Store the ID so the "Save" button knows which record to update later
        self.current_analysis_report_id = report_id 
        
        # 2. Fetch the binary data from the database manager
        image_data = self.db_manager.retrieve_image_from_db(report_id) 
        
        if image_data:
            pixmap = QPixmap()
            # 3. Load the bytes directly into a QPixmap
            if pixmap.loadFromData(image_data):
                self.original_image_label.setPixmap(
                    pixmap.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                
                # 4. Also convert to OpenCV format for processing
                nparr = np.frombuffer(image_data, np.uint8)
                self.cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.processed_cv_image = self.cv_image.copy()
            else:
                self.original_image_label.setText("Error loading image data")
        else:
            self.original_image_label.setText("No image found for this record")
    
    def load_image_by_patient_id(self):
        """Fetches the latest image for a specific Patient ID from the database."""
        patient_id_text = self.id_fetch_input.text().strip()
        
        if not patient_id_text.isdigit():
            QMessageBox.warning(self, "Input Error", "Please enter a valid numeric Patient ID.")
            return

        patient_id = int(patient_id_text)

        try:
            # Query the database for the most recent image for this patient
            sql = """
                SELECT Image_Data, report_id 
                FROM patient_health_metrics 
                WHERE patient_id = ? AND Image_Data IS NOT NULL
                ORDER BY report_id DESC LIMIT 1
            """
            self.db_manager.cursor.execute(sql, (patient_id,))
            result = self.db_manager.cursor.fetchone()

            if result and result[0]:
                # Use your existing logic to display and prepare for OpenCV
                self.load_image_for_analysis(result[1]) 
                QMessageBox.information(self, "Success", f"Loaded image for Patient ID: {patient_id}")
            else:
                QMessageBox.warning(self, "Not Found", f"No image record found for Patient ID: {patient_id}")

        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to fetch image: {str(e)}")

    def save_processed_image_to_db(self):
        """Encodes the processed OpenCV image and saves it back to the database."""
        if not hasattr(self, 'processed_cv_image') or self.processed_cv_image is None:
            QMessageBox.warning(self, "Save Error", "No processed image found to save.")
            return
            
        if not hasattr(self, 'current_analysis_report_id'):
            QMessageBox.warning(self, "Save Error", "No active record linked to this image.")
            return

        try:
            # Convert processed OpenCV image to bytes
            success, buffer = cv2.imencode(".png", self.processed_cv_image)
            if not success:
                raise ValueError("Image encoding failed")
            
            image_bytes = buffer.tobytes()

            # Call DB manager to update the BLOB
            # Note: Ensure your DatabaseManager has save_image_to_db(report_id, image_bytes)
            success = self.db_manager.save_image_to_db(self.current_analysis_report_id, image_bytes)
            
            if success:
                QMessageBox.information(self, "Success", "Processed image saved to database successfully.")
            else:
                QMessageBox.critical(self, "Error", "Failed to update database record.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def reset_image_view(self):
        """Clears the image display labels and resets internal image variables."""
        # 1. Clear the UI Labels
        self.original_image_label.clear()
        self.original_image_label.setText("No image loaded")
        
        self.processed_image_label.clear()
        self.processed_image_label.setText("Result will appear here")
        
        # 2. Reset the internal OpenCV variables
        self.cv_image = None
        self.processed_cv_image = None
        self.current_analysis_report_id = None
        
        # 3. Clear the Patient ID input
        if hasattr(self, 'id_fetch_input'):
            self.id_fetch_input.clear()
            
        QMessageBox.information(self, "Reset", "Image view and analysis data have been cleared.")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Medical Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.dcm);;All Files (*)")
        if file_path:
            self.cv_image = cv2.imread(file_path)
            if self.cv_image is None:
                QMessageBox.critical(self, "Image Error", "Could not load image file.")
                return
                
            # Copy original to processed for initial display
            self.processed_cv_image = self.cv_image.copy()
            
            # Display original image on the LEFT side
            self.display_image(self.cv_image, self.original_image_label)
            
            # Display processed image on the RIGHT side (initially same as original)
            self.display_image(self.processed_cv_image, self.processed_image_label)
            
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Image Loaded")
            msg_box.setText(f"Image loaded successfully from {os.path.basename(file_path)}")
            msg_box.setStandardButtons(QMessageBox.Ok)
            ok_button = msg_box.button(QMessageBox.Ok)
            if ok_button:
                ok_button.setObjectName("SuccessMessageButton")
            msg_box.exec_()

    def display_image(self, cv_img, label):
        if cv_img is None:
            label.clear()
            return
            
        if len(cv_img.shape) == 2:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            h, w = cv_img.shape
            ch = 3
        else:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qt_image = qt_image.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(QPixmap.fromImage(qt_image))

    def convert_to_grayscale(self):
        if self.cv_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        try:
            if len(self.cv_image.shape) == 3:
                gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.cv_image
            self.processed_cv_image = gray
            self.display_image(self.processed_cv_image, self.processed_image_label)
            self.update_viz_image()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to convert to grayscale: {str(e)}")

    def apply_blur(self):
        if self.cv_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        try:
            blur_type = self.blur_dropdown.currentText()
            if len(self.cv_image.shape) == 3:
                img = self.cv_image
            else:
                img = cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2BGR)

            if blur_type.startswith("Gaussian"):
                self.processed_cv_image = cv2.GaussianBlur(img, (15, 15), 0)
            elif blur_type.startswith("Median"):
                self.processed_cv_image = cv2.medianBlur(img, 5)

            self.display_image(self.processed_cv_image, self.processed_image_label)
            self.update_viz_image()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply blur: {str(e)}")

    def apply_edge_detection(self):
        if self.cv_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        try:
            if len(self.cv_image.shape) == 3:
                gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.cv_image
            edges = cv2.Canny(gray, 50, 150)
            self.processed_cv_image = edges
            self.display_image(self.processed_cv_image, self.processed_image_label)
            self.update_viz_image()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply edge detection: {str(e)}")

    def apply_threshold(self):
        if self.cv_image is not None:
            thresh_val = self.threshold_slider.value()
            
            if len(self.cv_image.shape) == 3:
                gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.cv_image
                
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            self.processed_cv_image = thresh
            self.display_image(self.processed_cv_image, self.processed_image_label)
            self.update_viz_image()



    def update_viz_image(self):
        """Updates the image placeholder in the Visualization tab with the latest processed image."""
        if self.processed_cv_image is not None:
            # Re-use your existing display logic to push the image to the Viz tab
            self.display_image(self.processed_cv_image, self.viz_image_label)
        else:
            self.viz_image_label.setText("No processed image available. Please process an image in the Image Tab.")

    def run_ma(self):
        col = self.ts_analysis_column.currentText()
        win = self.ma_window.value() 
        from data_analyzer import apply_moving_average
        self.filtered_df = apply_moving_average(self.df, col, win)
        
        self.analysis_ax.clear()
        self.analysis_ax.plot(self.df[col].values, label="Raw Signal", alpha=0.4, color='gray')
        self.analysis_ax.plot(self.filtered_df[f'{col}_smoothed'].values, label=f"Smoothed ({win}pt)", color='blue')
        self.analysis_ax.legend()
        self.analysis_ax.set_title(f"Moving Average Smoothing: {col}")
        self.analysis_canvas.draw()

    def run_threshold_filter(self):
        col = self.ts_analysis_column.currentText()
        val = self.thresh_val.value() 
        from data_analyzer import apply_threshold_filter
        self.filtered_df = apply_threshold_filter(self.df, col, min_val=val)
        self.populate_table(self.filtered_df)
        self.analysis_status_label.setText(f"Filtered {col}: Removed values below {val}")

    def plot_fft_viz(self):
        col = self.fft_column_combo.currentText()
        if not col: return
        
        # Clean data for FFT
        y = self.df[col].dropna().values
        n = len(y)
        if n == 0: return

        yf = np.fft.rfft(y)
        xf = np.fft.rfftfreq(n, d=1.0)

        self.figure.clear() 
        ax = self.figure.add_subplot(111)
        ax.plot(xf, np.abs(yf), color='#E74C3C')
        ax.set_title(f"Power Spectrum (FFT): {col}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        self.canvas.draw()


    def create_data_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Identify numerical columns for initial population
        cols = []
        if self.df is not None and not self.df.empty:
            cols = self.df.select_dtypes(include=['number']).columns.tolist()

        title = QLabel("Data Visualization")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        controls_group = QWidget()
        controls_group_layout = QGridLayout(controls_group)
        
        # --- Scatter Plot Controls ---
        self.scatter_x_combo = QComboBox()
        self.scatter_x_combo.setObjectName("VizScatterX")
        self.scatter_x_combo.addItems(cols)
        
        self.scatter_y_combo = QComboBox()
        self.scatter_y_combo.setObjectName("VizScatterY")
        self.scatter_y_combo.addItems(cols)
        
        self.scatter_plot_btn = QPushButton("Plot Scatter")
        self.scatter_plot_btn.setObjectName("VizScatterBtn")
        self.scatter_plot_btn.setToolTip("Plots relationship between two selected variables.")
        self.scatter_plot_btn.clicked.connect(self.plot_scatter)
        
        controls_group_layout.addWidget(QLabel("Scatter X:"), 0, 0)
        controls_group_layout.addWidget(self.scatter_x_combo, 0, 1)
        controls_group_layout.addWidget(QLabel("Scatter Y:"), 0, 2)
        controls_group_layout.addWidget(self.scatter_y_combo, 0, 3)
        controls_group_layout.addWidget(self.scatter_plot_btn, 0, 4)

        # --- Time-Series Plot Controls ---
        self.ts_column_combo = QComboBox()
        self.ts_column_combo.setObjectName("VizTSCombo")
        self.ts_column_combo.addItems(cols)
        
        self.ts_plot_btn = QPushButton("Plot Time-Series")
        self.ts_plot_btn.setObjectName("VizTimeSeriesBtn")
        self.ts_plot_btn.setToolTip("Plots the value of a variable over index/time.")
        self.ts_plot_btn.clicked.connect(self.plot_time_series)
        
        controls_group_layout.addWidget(QLabel("Time-Series:"), 1, 0)
        controls_group_layout.addWidget(self.ts_column_combo, 1, 1, 1, 3)
        controls_group_layout.addWidget(self.ts_plot_btn, 1, 4)

        # --- FFT Spectrum Plot Controls ---
        self.fft_column_combo = QComboBox()
        self.fft_column_combo.setObjectName("VizFFTCombo")
        self.fft_column_combo.addItems(cols)
        
        self.fft_plot_btn = QPushButton("Plot FFT Spectrum")
        self.fft_plot_btn.setObjectName("VizFFTBtn")
        self.fft_plot_btn.setToolTip("Displays the frequency components of a signal.")
        self.fft_plot_btn.clicked.connect(self.plot_fft_viz) 
        
        controls_group_layout.addWidget(QLabel("FFT Spectrum:"), 2, 0)
        controls_group_layout.addWidget(self.fft_column_combo, 2, 1, 1, 3)
        controls_group_layout.addWidget(self.fft_plot_btn, 2, 4)
        
        # --- Heatmap Button ---
        self.heatmap_btn = QPushButton("Show Correlation Heatmap")
        self.heatmap_btn.setObjectName("VizHeatmapBtn")
        self.heatmap_btn.setToolTip("Visualize correlations between all numerical variables.")
        self.heatmap_btn.clicked.connect(self.plot_heatmap)
        controls_group_layout.addWidget(self.heatmap_btn, 3, 0, 1, 5)

        layout.addWidget(controls_group)
        
        # --- Matplotlib Canvas Section ---
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.figure.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)
        self.canvas.setMinimumHeight(620)

        viz_scroll = QScrollArea()
        viz_scroll.setWidgetResizable(True)
        viz_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        viz_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        viz_scroll.setWidget(self.canvas)
        layout.addWidget(viz_scroll)
        
        # Optional Placeholder for static images
        self.viz_image_label = QLabel("Image Visualization Placeholder")
        self.viz_image_label.setAlignment(Qt.AlignCenter)
        self.viz_image_label.setFixedSize(900, 180) 
        layout.addWidget(self.viz_image_label) 

        return panel

    def plot_scatter(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data before plotting scatter.")
            return

        x_col = self.scatter_x_combo.currentText()
        y_col = self.scatter_y_combo.currentText()
        
        if not x_col or not y_col:
            QMessageBox.warning(self, "Error", "Select both X and Y columns.")
            return

        if x_col not in self.df.columns or y_col not in self.df.columns:
            QMessageBox.warning(self, "Invalid Columns", f"One or both columns ('{x_col}', '{y_col}') do not exist in the dataset.")
            return

        try:
            if not pd.api.types.is_numeric_dtype(self.df[x_col]) or not pd.api.types.is_numeric_dtype(self.df[y_col]):
                QMessageBox.warning(self, "Invalid Columns", "Both columns must be numeric for scatter plot.")
                return

            valid_data = self.df[[x_col, y_col]].dropna()
            if len(valid_data) == 0:
                QMessageBox.warning(self, "No Data", "No valid data points available for scatter plot.")
                return

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_box_aspect(0.6)
            ax.scatter(valid_data[x_col], valid_data[y_col], alpha=0.6, color='#3498DB')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot scatter: {str(e)}")
        
    def plot_heatmap(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data before showing heatmap.")
            return
        
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            numerical_df = self.df.select_dtypes(include=np.number)

            if numerical_df.empty:
                QMessageBox.warning(self, "Data Error", "No numerical data available for heatmap.")
                return

            if len(numerical_df.columns) < 2:
                QMessageBox.warning(self, "Insufficient Data", "Need at least 2 numerical columns for correlation heatmap.")
                return

            corr = numerical_df.corr()

            if corr.empty:
                QMessageBox.warning(self, "No Correlation", "Unable to compute correlation matrix.")
                return

            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                        square=True,
                        cbar_kws={"shrink": 0.7})

            ax.set_title("Correlation Heatmap")
            self.figure.tight_layout()
            self.canvas.draw()
            ax.set_box_aspect(0.8)
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display heatmap: {str(e)}")

    def plot_time_series(self):
        """Generates a time-series plot comparing Raw and Cleaned signals."""
        if self.df.empty:
            return

        try:
            # Get selected column from the dropdown in your spectrum panel
            col = self.signal_dropdown.currentText()
            
            # Convert signal data to numeric for processing
            raw_data = pd.to_numeric(self.df[col], errors='coerce').dropna().values
            
            if len(raw_data) == 0:
                QMessageBox.warning(self, "Warning", "No valid numerical data in this column.")
                return

            # Import and apply the new FFT denoising from data_analyzer.py
            from data_analyzer import fft_denoise_signal
            cleaned_data = fft_denoise_signal(raw_data, threshold_percent=0.1)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot Raw in background and Cleaned in foreground
            ax.plot(raw_data, color='lightgray', label='Raw Signal', alpha=0.5)
            ax.plot(cleaned_data, color='#2ECC71', label='FFT Cleaned', linewidth=1.5)
            
            ax.set_title(f"Signal Comparison: {col}")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot signal: {str(e)}")

    def plot_fft_viz(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data before plotting FFT spectrum.")
            return

        col = self.fft_column_combo.currentText()
        if not col:
            QMessageBox.warning(self, "Error", "Select a column for FFT.")
            return

        if col not in self.df.columns:
            QMessageBox.warning(self, "Invalid Column", f"Column '{col}' does not exist in the dataset.")
            return

        try:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                QMessageBox.warning(self, "Invalid Column", f"Column '{col}' is not numeric.")
                return

            y = self.df[col].dropna().values
            n = len(y)

            if n < 2:
                QMessageBox.warning(self, "Error", "Not enough data points for FFT analysis.")
                return

            yf = np.fft.rfft(y)
            xf = np.fft.rfftfreq(n, d=1.0)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_box_aspect(0.6)
            ax.plot(xf, 2.0/n * np.abs(yf), color='#E74C3C', linewidth=1.5)
            ax.set_title(f"Power Spectrum (FFT): {col}")
            ax.set_xlabel("Frequency (Cycles/Sample)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, linestyle='--', alpha=0.6)
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot FFT spectrum: {str(e)}")

    def load_next_page(self):
        self.current_page += 1
        self.db_retrieve_data()

    def load_previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.db_retrieve_data()

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    data = {
        'Age': np.random.randint(20, 80, 50),
        'heart_rate': np.random.randint(60, 100, 50),
        'Blood_Pressure': np.random.randint(100, 150, 50),
        'Cholesterol_Level': np.random.randint(150, 300, 50),
        'signal_data': np.sin(np.linspace(0, 100, 50)) + np.random.randn(50) * 0.5,
        'Heart_Disease_Status': np.random.choice(['Yes', 'No'], 50)
    }
    dummy_df = pd.DataFrame(data)

    app = QApplication(sys.argv)
    window = HealthcareApp(dummy_df, db_manager=None) 
    window.show()
    sys.exit(app.exec_())