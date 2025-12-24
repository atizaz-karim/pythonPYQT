

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QTabWidget, QFileDialog, QTableWidget, 
    QTableWidgetItem, QScrollArea, QSlider, QLineEdit, QCheckBox, 
    QSpinBox, QMessageBox, QGridLayout, QInputDialog, QDoubleSpinBox,
    QFrame, QGroupBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QSize

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
        # 1. Setup Data Variables
        self.df = df
        self.db_manager = db_manager
        self.current_page = 0
        self.rows_per_page = 50
        self.filtered_df = df.copy()
        self.cv_image = None
        self.processed_cv_image = None  

        self.filtered_df = df.copy()
        self.cv_image = None
        self.processed_cv_image = None
        
        self.setWindowTitle("Healthcare Data and Medical Image Processing Tool")
        self.setGeometry(50, 50, 1400, 900)

        # Global white theme for pop-ups and transparent inputs with black text
        self.setStyleSheet("""
            /* Pop-up dialogs */
            QMessageBox, QInputDialog {
                background-color: white;
                color: black;
            }
            QMessageBox QLabel, QInputDialog QLabel {
                color: black;
            }
            QMessageBox QPushButton, QInputDialog QPushButton {
                background-color: white;
                color: black;
                border: 1px solid #bfbfbf;
                padding: 4px 10px;
            }

            /* Inputs (including those inside popups) */
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
                background-color: transparent;
                color: black;
                border: 1px solid #bfbfbf;
                padding: 2px 4px;
            }
            QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
            QTextEdit:disabled, QPlainTextEdit:disabled {
                color: #555555;
            }

            /* Dropdown list view for combo boxes */
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                border: 1px solid #bfbfbf;
            }
            QComboBox QAbstractItemView::item {
                background-color: white;
                color: black;
                padding: 4px 8px;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #e6f2ff; /* light blue highlight */
                color: black;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.sidebar = QWidget()
        self.sidebar.setObjectName("SidebarWidget") 
        self.sidebar.setFixedWidth(250)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        

        self.stacked_widget = QTabWidget() 
        self.stacked_widget.tabBar().setVisible(False) 

        # Add Tabs (Panels)
        self.stacked_widget.addTab(self.create_data_management_panel(), "Data Loading and Management")
        self.stacked_widget.addTab(self.create_analysis_panel(), "Health Data Analysis")
        self.stacked_widget.addTab(self.create_spectrum_panel(), "Spectrum Analysis")
        self.stacked_widget.addTab(self.create_image_processing_panel(), "Medical Image Processing")
        self.stacked_widget.addTab(self.create_data_visualization_panel(), "Data Visualization")
        
        tab_names = ["Patient Data Management", "Health Data Analysis", "Spectrum Analysis", 
                     "Image Processing", "Data Visualization"]
                     
        for i, name in enumerate(tab_names):
            btn = QPushButton(name)
            btn.setObjectName(f"SidebarBtn_{i}") # SET OBJECT NAME FOR QSS TARGETING
            btn.setToolTip(f"Switch to the {name} section.")
            btn.clicked.connect(lambda checked, index=i: self.stacked_widget.setCurrentIndex(index))
            self.sidebar_layout.addWidget(btn)
        
        self.sidebar_layout.addStretch(1) 

        # Add components to the main layout
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stacked_widget)
        
        self._setup_menu_bar()

        if self.db_manager:
            self.db_retrieve_data() 
        else:
            self.populate_table(self.df.head(self.rows_per_page))

    # --- Utility Methods ---
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

        # 2. CLEAR EVERYTHING: Reset rows and content
        self.table_widget.setRowCount(0) 
        self.table_widget.clearContents()

        if df is None or df.empty:
            self.table_widget.setColumnCount(0)
            return

        self.table_widget.setRowCount(len(df))
        self.table_widget.setColumnCount(len(df.columns))
        self.table_widget.setHorizontalHeaderLabels(df.columns)
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
        self.table_widget.resizeColumnsToContents()
        self.table_widget.setToolTip("Scrollable view of the loaded or retrieved dataset.")

# --- Panel 1: Data Management (CRUD) ---
    def create_data_management_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Data Loading and Management")
        title.setObjectName("PanelTitle") 
        layout.addWidget(title)

        # Data Source Section (Loaders)
        source_group = QWidget()
        source_layout = QHBoxLayout(source_group)
        
        self.load_csv_btn = QPushButton("Load CSV File (Local)")
        self.load_csv_btn.setToolTip("Browse and load data from a local CSV file.")
        self.load_csv_btn.clicked.connect(self.load_csv)

        self.db_connect_btn = QPushButton("Refresh Table")
        self.db_connect_btn.setToolTip("Retrieves all patient records from the database and loads them into the table.")
        self.db_connect_btn.clicked.connect(self.db_retrieve_data) 
        
        source_layout.addWidget(self.load_csv_btn)
        source_layout.addWidget(self.db_connect_btn)
        layout.addWidget(source_group)
        
        # Database Operations (CRUD)
        db_ops_group = QWidget()
        db_ops_group.setObjectName("DbOpsGroup") # SET OBJECT NAME FOR QSS TARGETING
        # REMOVED: db_ops_group.setStyleSheet("...") 
        db_ops_layout = QGridLayout(db_ops_group)
        
        db_ops_layout.addWidget(QLabel("### Database CRUD Operations"), 0, 0, 1, 4)
        
        db_ops_layout.addWidget(QLabel("Patient ID:"), 1, 0)
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setObjectName("PatientIDInput")
        self.patient_id_input.setPlaceholderText("Enter Patient ID for Update/Delete")
        self.patient_id_input.setFixedWidth(200)
        self.patient_id_input.setToolTip("Specify the unique ID of the record to modify or delete.")
        db_ops_layout.addWidget(self.patient_id_input, 1, 1)

        self.insert_btn = QPushButton("Insert Current Data")
        self.insert_btn.setObjectName("InsertButton") # SET OBJECT NAME FOR QSS TARGETING
        self.insert_btn.setToolTip("Inserts the currently loaded DataFrame into the DB (bulk insert).")
        # REMOVED: self.insert_btn.setStyleSheet("...") 
        self.insert_btn.clicked.connect(self.db_insert_data)

        self.update_btn = QPushButton("Update Record")
        self.update_btn.setToolTip("Updates the record specified by Patient ID (requires prompt).")
        self.update_btn.clicked.connect(self.db_update_prompt) 
        
        self.delete_btn = QPushButton("Delete Record")
        self.delete_btn.setObjectName("DeleteButton") # SET OBJECT NAME FOR QSS TARGETING
        self.delete_btn.setToolTip("Removes the record specified by Patient ID permanently.")
        # REMOVED: self.delete_btn.setStyleSheet("...") 
        self.delete_btn.clicked.connect(self.db_delete_record)

        db_ops_layout.addWidget(self.insert_btn, 2, 0)
        db_ops_layout.addWidget(self.update_btn, 2, 1)
        db_ops_layout.addWidget(self.delete_btn, 2, 2)
        db_ops_layout.setColumnStretch(3, 1) 

        layout.addWidget(db_ops_group)

        # Table view
        layout.addWidget(QLabel("Loaded Dataset / Database View:"))
        self.table_widget = QTableWidget()
        # Make the database view taller for easier browsing
        self.table_widget.setMinimumHeight(450)
        # REMOVED: self.table_widget.setStyleSheet("...") (Moved to QSS)
        layout.addWidget(self.table_widget)
        self.populate_table(self.df) 
        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)

        # Pagination controls anchored at the bottom of the panel
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous Page")
        self.next_btn = QPushButton("Next Page")
        self.page_label = QLabel("Page 1")
        self.prev_btn.setEnabled(False) # Start disabled
        
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

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.filtered_df = self.df.copy() 
                self.populate_table(self.df)
                self._update_viz_dropdowns()
                self._update_analysis_dropdowns()
                QMessageBox.information(self, "Success", f"Data loaded successfully from {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Data", f"Failed to load CSV: {e}")

    def _update_analysis_dropdowns(self):
        if self.df is None or self.df.empty:
            return
            
        try:
            numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            
            if hasattr(self, 'ts_analysis_column'):
                current_ts = self.ts_analysis_column.currentText()
                self.ts_analysis_column.clear()
                self.ts_analysis_column.addItems(numerical_cols if numerical_cols else [])
                if current_ts in numerical_cols:
                    self.ts_analysis_column.setCurrentText(current_ts)
                elif len(numerical_cols) > 0:
                    self.ts_analysis_column.setCurrentIndex(0)
            
            if hasattr(self, 'metric1_dropdown'):
                current_m1 = self.metric1_dropdown.currentText()
                self.metric1_dropdown.clear()
                self.metric1_dropdown.addItems(numerical_cols if numerical_cols else [])
                if current_m1 in numerical_cols:
                    self.metric1_dropdown.setCurrentText(current_m1)
                elif len(numerical_cols) > 0:
                    self.metric1_dropdown.setCurrentIndex(0)
            
            if hasattr(self, 'metric2_dropdown'):
                current_m2 = self.metric2_dropdown.currentText()
                self.metric2_dropdown.clear()
                self.metric2_dropdown.addItems(numerical_cols if numerical_cols else [])
                if current_m2 in numerical_cols:
                    self.metric2_dropdown.setCurrentText(current_m2)
                elif len(numerical_cols) > 1:
                    self.metric2_dropdown.setCurrentIndex(1)
                elif len(numerical_cols) > 0:
                    self.metric2_dropdown.setCurrentIndex(0)
        except Exception as e:
            print(f"Error updating analysis dropdowns: {e}")

    def _update_viz_dropdowns(self):
        if self.df is None or self.df.empty:
            return
            
        try:
            numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            
            if hasattr(self, 'scatter_x_combo'):
                current_x = self.scatter_x_combo.currentText()
                self.scatter_x_combo.clear()
                self.scatter_x_combo.addItems(numerical_cols if numerical_cols else [])
                if current_x in numerical_cols:
                    self.scatter_x_combo.setCurrentText(current_x)
                elif len(numerical_cols) > 0:
                    self.scatter_x_combo.setCurrentIndex(0)
            
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
            
            if hasattr(self, 'ts_column_combo'):
                current_ts = self.ts_column_combo.currentText()
                self.ts_column_combo.clear()
                self.ts_column_combo.addItems(numerical_cols if numerical_cols else [])
                if current_ts in numerical_cols:
                    self.ts_column_combo.setCurrentText(current_ts)
                elif len(numerical_cols) > 0:
                    self.ts_column_combo.setCurrentIndex(0)
            
            if hasattr(self, 'fft_column_combo'):
                current_fft = self.fft_column_combo.currentText()
                self.fft_column_combo.clear()
                self.fft_column_combo.addItems(numerical_cols if numerical_cols else [])
                if current_fft in numerical_cols:
                    self.fft_column_combo.setCurrentText(current_fft)
                elif len(numerical_cols) > 0:
                    self.fft_column_combo.setCurrentIndex(0)
            
            if hasattr(self, 'signal_dropdown'):
                current_signal = self.signal_dropdown.currentText()
                self.signal_dropdown.clear()
                self.signal_dropdown.addItems(numerical_cols if numerical_cols else [])
                if current_signal in numerical_cols:
                    self.signal_dropdown.setCurrentText(current_signal)
                elif len(numerical_cols) > 0:
                    self.signal_dropdown.setCurrentIndex(0)
        except Exception:
            pass

    # --- DB HANDLERS (Unchanged logic) ---
    def _check_db_manager(self):
        if not self.db_manager:
            QMessageBox.critical(self, "DB Error", "Database Manager is not initialized. Check your main.py setup.")
            return False
        return True
    def db_retrieve_data(self):
        if not self._check_db_manager(): return
        try:
            # 1. Calculate the starting point (Offset)
            offset = self.current_page * self.rows_per_page
            
            # 2. Get the specific 50 rows from the database
            # This ensures we are only pulling a small slice of data into RAM
            retrieved_df = self.db_manager.get_patient_data(limit=self.rows_per_page, offset=offset) 
            
            # 3. Update the app's data reference
            self.df = retrieved_df
            self.filtered_df = self.df.copy()
            
            # 4. Clear and Fill the table with ONLY these 50 rows
            self.populate_table(self.df)
            self._update_viz_dropdowns()
            self._update_analysis_dropdowns()
            
            # 5. Get the TOTAL record count to calculate total pages
            # Without this, the GUI doesn't know how many pages exist in total
            total_records = self.db_manager.get_total_count()
            
            # 6. Calculate total pages (handles the math for the last partial page)
            if total_records == 0:
                total_pages = 1
            else:
                total_pages = (total_records // self.rows_per_page) + (1 if total_records % self.rows_per_page > 0 else 0)
            
            # 7. Update the UI Labels
            self.page_label.setText(f"Page {self.current_page + 1} of {total_pages}")
            
            # 8. Set Button States (Enable/Disable)
            # Disable 'Previous' if on page 1
            self.prev_btn.setEnabled(self.current_page > 0)
            # Disable 'Next' if we have reached the end of the database
            self.next_btn.setEnabled(offset + len(retrieved_df) < total_records)
            
            self.status_label.setText(f"Showing rows {offset + 1} to {offset + len(retrieved_df)} of {total_records}")
            
        except Exception as e:
            QMessageBox.critical(self, "DB Retrieval Error", f"Failed to retrieve data: {e}")

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

    def db_update_prompt(self):
        if not self._check_db_manager(): return

        patient_id_str = self.patient_id_input.text().strip()
        if not patient_id_str.isdigit():
            QMessageBox.warning(self, "Input Error", "Please enter a valid Patient ID (number) to update.")
            return
        patient_id = int(patient_id_str)
        
        fields = ['Age', 'Gender', 'Blood_Pressure', 'Cholesterol_Level', 'BMI'] 

        field, ok = QInputDialog.getItem(self, "Update Record", "Select Field to Update:", fields, 0, False)
        if ok and field:
            new_value, ok = QInputDialog.getText(self, "Update Record", f"Enter new value for {field}:")
            if ok:
                try:
                    update_kwargs = {field: new_value}
                    self.db_manager.update_patient_data(patient_id, **update_kwargs)
                    QMessageBox.information(self, "Success", f"Patient {patient_id} updated successfully. Reload data to view changes.")
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
                                     f"Are you sure you want to DELETE record for Patient ID {patient_id}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                self.db_manager.delete_patient_data(patient_id)
                QMessageBox.information(self, "Success", f"Patient {patient_id} deleted successfully.")
                self.db_retrieve_data() 
            except Exception as e:
                QMessageBox.critical(self, "DB Deletion Error", f"Failed to delete record: {e}")

# --- Panel 2: Analysis ---
    def create_analysis_panel(self):
        panel = QWidget()
        main_layout = QVBoxLayout(panel)

        # 1. Safety Check: Ensure we have columns to show even if df is empty
        if self.df is not None and not self.df.empty:
            numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        else:
            # Placeholder columns if no data is loaded yet
            numerical_cols = ['Age', 'Blood_Pressure', 'Cholesterol_Level', 'BMI']

        title = QLabel("Health Data Analysis")
        title.setObjectName("PanelTitle")
        main_layout.addWidget(title)

        controls_group = QHBoxLayout()

        # --- Left Side: Filtering and Time-series ---
        filtering_widget = QWidget()
        filtering_widget.setObjectName("FilterGroup") 
        filtering_group = QVBoxLayout(filtering_widget)
        
        # Using a Header label for styling
        filter_header = QLabel("Data Filtering & Smoothing")
        filter_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        filtering_group.addWidget(filter_header)

        # Moving Average Control
        ma_layout = QHBoxLayout()
        ma_layout.addWidget(QLabel("MA Window:"))
        self.ma_window = QSpinBox()
        self.ma_window.setRange(2, 100)
        self.ma_window.setValue(5)
        ma_layout.addWidget(self.ma_window)
        filtering_group.addLayout(ma_layout)

        # Threshold Control
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
        self.ts_raw_checkbox = QCheckBox("Show Raw")
        self.ts_raw_checkbox.setChecked(True)
        ts_controls.addWidget(self.ts_analysis_column)
        ts_controls.addWidget(self.ts_raw_checkbox)
        filtering_group.addLayout(ts_controls)
        
        # BUTTONS: Fixed the connection name to apply_filters_and_plot
        self.apply_filter_btn = QPushButton("Apply & Plot")
        self.apply_filter_btn.clicked.connect(self.apply_filters_and_plot)
        self.reset_filter_btn = QPushButton("Reset")
        self.reset_filter_btn.clicked.connect(self.reset_filters)
        
        filter_btn_layout = QHBoxLayout()
        filter_btn_layout.addWidget(self.apply_filter_btn)
        filter_btn_layout.addWidget(self.reset_filter_btn)
        filtering_group.addLayout(filter_btn_layout)
        
        controls_group.addWidget(filtering_widget)
        
        # --- Right Side: Correlation Analysis ---
        corr_widget = QWidget()
        corr_widget.setObjectName("CorrGroup") 
        corr_group = QVBoxLayout(corr_widget)
        corr_header = QLabel("Correlation Analysis")
        corr_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        corr_group.addWidget(corr_header)
        
        self.metric1_dropdown = QComboBox()
        self.metric1_dropdown.addItems(numerical_cols if numerical_cols else [])
        self.metric2_dropdown = QComboBox()
        self.metric2_dropdown.addItems(numerical_cols if numerical_cols else [])
        
        corr_group.addWidget(QLabel("Metric 1:"))
        corr_group.addWidget(self.metric1_dropdown)
        corr_group.addWidget(QLabel("Metric 2:"))
        corr_group.addWidget(self.metric2_dropdown)

        self.compute_corr_btn = QPushButton("Scatter Plot")
        self.compute_corr_btn.clicked.connect(self.compute_correlation_plot)
        self.show_heatmap_btn = QPushButton("Heatmap")
        self.show_heatmap_btn.clicked.connect(self.show_heatmap)
        
        corr_btn_layout = QHBoxLayout()
        corr_btn_layout.addWidget(self.compute_corr_btn)
        corr_btn_layout.addWidget(self.show_heatmap_btn)
        corr_group.addLayout(corr_btn_layout)
        
        controls_group.addWidget(corr_widget)
        main_layout.addLayout(controls_group)

        # --- Plot Area (Scrollable) ---
        self.analysis_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        # Increased height as requested
        self.analysis_canvas.setMinimumHeight(700) 

        analysis_scroll = QScrollArea()
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setWidget(self.analysis_canvas)
        main_layout.addWidget(analysis_scroll)
        
        # Create the initial axis
        self.analysis_ax = self.analysis_canvas.figure.add_subplot(111)

        self.analysis_status_label = QLabel("Ready for analysis.")
        main_layout.addWidget(self.analysis_status_label)

        return panel

    def apply_filters_and_plot(self):
        # 1. Safety Check
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data before applying filters.")
            return

        # 2. Get values from UI controls
        factor = self.outlier_slider.value() / 10.0
        ma_window = self.ma_window.value()
        col = self.ts_analysis_column.currentText()
        
        # 3. Create working copy for filtering
        self.filtered_df = self.df.copy()

        # 4. OUTLIER REMOVAL (IQR Method)
        # This actually uses the 'factor' from your slider
        Q1 = self.filtered_df[col].quantile(0.25)
        Q3 = self.filtered_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Filter the dataframe
        self.filtered_df = self.filtered_df[
            (self.filtered_df[col] >= lower_bound) & 
            (self.filtered_df[col] <= upper_bound)
        ]

        # 5. MOVING AVERAGE (Smoothing)
        # Applies the window size from your SpinBox
        self.filtered_df['MA'] = self.filtered_df[col].rolling(window=ma_window, min_periods=1).mean()

        # 6. PLOTTING
        self.analysis_ax.clear()
        
        if self.ts_raw_checkbox.isChecked():
            # Plot raw data in a light gray so it's visible but not distracting
            self.analysis_ax.plot(self.df[col].values, color='#BDC3C7', alpha=0.4, label='Raw Data')
        
        # Plot the processed (Filtered + Moving Average) data
        self.analysis_ax.plot(self.filtered_df['MA'].values, color='#3498DB', linewidth=2, label='Filtered (MA)')
        
        # 7. UI Formatting
        self.analysis_ax.set_title(f"Analysis: {col}")
        self.analysis_ax.set_xlabel("Sample Index")
        self.analysis_ax.set_ylabel("Value")
        self.analysis_ax.legend()
        self.analysis_ax.grid(True, linestyle='--', alpha=0.5)
        
        # Maintain professional aspect ratio in large container
        self.analysis_ax.set_box_aspect(0.6)
        
        self.analysis_canvas.figure.tight_layout()
        self.analysis_canvas.draw()
        
        self.analysis_status_label.setText(
            f"Applied IQR Factor: {factor} | MA Window: {ma_window} | Rows Remaining: {len(self.filtered_df)}"
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

        working_df = self.filtered_df if (self.filtered_df is not None and not self.filtered_df.empty) else self.df
        
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
            if len(valid_data) == 0:
                QMessageBox.warning(self, "No Data", "No valid data points available for correlation analysis.")
                return

            self.analysis_ax.clear()
            valid_data.plot.scatter(x=col1, y=col2, ax=self.analysis_ax)
            corr_value = valid_data[col1].corr(valid_data[col2])
            self.analysis_ax.set_title(f'Scatter: {col1} vs {col2} (corr={corr_value:.2f})')
            self.analysis_canvas.draw()
            self.analysis_status_label.setText(f"Correlation between {col1} and {col2}: {corr_value:.2f}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compute correlation: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_heatmap(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data before showing heatmap.")
            return

        try:
            self.analysis_canvas.figure.clear()
            self.analysis_ax = self.analysis_canvas.figure.add_subplot(111)
            
            working_df = self.filtered_df if (self.filtered_df is not None and not self.filtered_df.empty) else self.df
            numerical_df = working_df.select_dtypes(include='number')
            
            if numerical_df.empty:
                QMessageBox.warning(self, "No Numerical Data", "No numerical columns available for heatmap.")
                return

            if len(numerical_df.columns) < 2:
                QMessageBox.warning(self, "Insufficient Data", "Need at least 2 numerical columns for correlation heatmap.")
                return
                
            corr = numerical_df.corr()

            if corr.empty:
                QMessageBox.warning(self, "No Correlation", "Unable to compute correlation matrix.")
                return

            sns.heatmap(
                corr, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                ax=self.analysis_ax,
                square=True, 
                cbar_kws={"shrink": 0.7}
            )
            
            self.analysis_ax.set_title("Correlation Heatmap")
            self.analysis_ax.set_box_aspect(0.8) 
            self.analysis_canvas.figure.tight_layout()
            self.analysis_canvas.draw()
            self.analysis_status_label.setText("Correlation heatmap displayed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display heatmap: {str(e)}")
            import traceback
            traceback.print_exc()


    def create_spectrum_panel(self):
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        main_layout.setSpacing(10)

        title = QLabel("Spectrum Analysis")
        title.setObjectName("PanelTitle")
        main_layout.addWidget(title)
        
        signal_loading_group = QGroupBox("Signal Loading & Selection")
        signal_loading_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        signal_loading_layout = QHBoxLayout(signal_loading_group)
        signal_loading_layout.setSpacing(15)
        
        if self.df is not None and not self.df.empty:
            numeric_cols = self.df.select_dtypes(include='number').columns.tolist()
        else:
            numeric_cols = []
        
        signal_loading_layout.addWidget(QLabel("Biomedical Signal:"))
        self.signal_dropdown = QComboBox()
        self.signal_dropdown.addItems(numeric_cols if numeric_cols else [])
        self.signal_dropdown.setMinimumWidth(200)
        self.signal_dropdown.setToolTip("Select a signal column from the dataset (e.g., ECG, EEG data)")
        signal_loading_layout.addWidget(self.signal_dropdown)
        
        self.plot_signal_btn = QPushButton("Load & Display Signal")
        self.plot_signal_btn.setToolTip("Loads and displays the selected signal in a scrollable plot.")
        self.plot_signal_btn.clicked.connect(self.plot_raw_signal)
        signal_loading_layout.addWidget(self.plot_signal_btn)
        signal_loading_layout.addStretch()
        
        main_layout.addWidget(signal_loading_group)

        raw_signal_group = QGroupBox("Raw Signal Display (Scrollable)")
        raw_signal_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
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
        fft_controls_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
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
        self.fft_btn.setToolTip("Calculates and displays the Fast Fourier Transform (Power Spectrum) for the selected segment.")
        self.fft_btn.clicked.connect(self.plot_fft)
        fft_button_layout.addWidget(self.fft_btn)
        fft_button_layout.addStretch()
        fft_controls_layout.addLayout(fft_button_layout)
        
        main_layout.addWidget(fft_controls_group)

        visualization_controls_group = QGroupBox("Visualization Controls")
        visualization_controls_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
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
        self.apply_zoom_btn.setToolTip("Apply the specified axis limits to zoom into the frequency spectrum.")
        self.apply_zoom_btn.clicked.connect(self.apply_fft_zoom)
        viz_controls_layout.addWidget(self.apply_zoom_btn)
        
        self.reset_zoom_btn = QPushButton("Reset View")
        self.reset_zoom_btn.setToolTip("Reset the view to show the full spectrum.")
        self.reset_zoom_btn.clicked.connect(self.reset_fft_zoom)
        viz_controls_layout.addWidget(self.reset_zoom_btn)
        
        main_layout.addWidget(visualization_controls_group)

        fft_display_group = QGroupBox("FFT Power Spectrum")
        fft_display_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; }")
        fft_display_layout = QVBoxLayout(fft_display_group)
        
        self.spectrum_canvas = FigureCanvas(Figure(figsize=(12, 5)))
        self.spectrum_canvas.setMinimumHeight(350)
        fft_display_layout.addWidget(self.spectrum_canvas)
        self.spectrum_ax = self.spectrum_canvas.figure.add_subplot(111)
        
        main_layout.addWidget(fft_display_group)

        return panel

    def plot_raw_signal(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data before plotting signal.")
            return

        col = self.signal_dropdown.currentText()
        if not col:
            QMessageBox.warning(self, "No Selection", "Please select a signal column.")
            return

        if col not in self.df.columns:
            QMessageBox.warning(self, "Invalid Column", f"Column '{col}' does not exist in the dataset.")
            return

        try:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                QMessageBox.warning(self, "Invalid Column", f"Column '{col}' is not numeric.")
                return

            signal = self.df[col].dropna().values
            if len(signal) == 0:
                QMessageBox.warning(self, "No Data", f"Column '{col}' contains no valid data.")
                return

            self.raw_signal_ax.clear()
            self.raw_signal_ax.plot(signal, color='#3498DB', linewidth=1)
            self.raw_signal_ax.set_title(f"Raw Signal: {col} ({len(signal)} samples)", fontsize=12, fontweight='bold')
            self.raw_signal_ax.set_xlabel("Sample Index", fontsize=10)
            self.raw_signal_ax.set_ylabel(col, fontsize=10)
            self.raw_signal_ax.grid(True, linestyle='--', alpha=0.5)
            self.raw_signal_canvas.figure.tight_layout()
            self.raw_signal_canvas.draw()
            
            max_samples = len(signal)
            self.segment_start_slider.setRange(0, max_samples - 1)
            self.segment_end_slider.setRange(10, max_samples)
            self.segment_end_slider.setValue(min(100, max_samples))
            self.segment_end_label.setText(str(min(100, max_samples)))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot signal: {str(e)}")

    def plot_fft(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data before computing FFT.")
            return

        col = self.signal_dropdown.currentText()
        if not col:
            QMessageBox.warning(self, "No Selection", "Please select a signal column.")
            return

        if col not in self.df.columns:
            QMessageBox.warning(self, "Invalid Column", f"Column '{col}' does not exist in the dataset.")
            return

        try:
            signal = self.df[col].dropna().values
            if len(signal) == 0:
                QMessageBox.warning(self, "No Data", f"Column '{col}' contains no valid data.")
                return

            start_idx = self.segment_start_slider.value()
            end_idx = self.segment_end_slider.value()
            
            if start_idx >= end_idx:
                QMessageBox.warning(self, "Invalid Range", "Start index must be less than end index.")
                return

            signal_segment = signal[start_idx:end_idx]
            n = len(signal_segment)
            
            if n < 2:
                QMessageBox.warning(self, "Insufficient Data", "Selected segment must contain at least 2 samples.")
                return
            
            freq = np.fft.rfftfreq(n, d=1.0)
            fft_vals = np.abs(np.fft.rfft(signal_segment))
            power_spectrum = 2.0/n * fft_vals

            self.spectrum_ax.clear()
            self.spectrum_ax.plot(freq, power_spectrum, color='#E74C3C', linewidth=1.5)
            self.spectrum_ax.set_title(f"FFT Power Spectrum: {col} (Samples {start_idx}-{end_idx}, n={n})", 
                                      fontsize=12, fontweight='bold')
            self.spectrum_ax.set_xlabel("Frequency (cycles/sample)", fontsize=10)
            self.spectrum_ax.set_ylabel("Amplitude", fontsize=10)
            self.spectrum_ax.grid(True, linestyle='--', alpha=0.5)
            self.spectrum_ax.set_xlim(self.freq_min_input.value(), self.freq_max_input.value())
            self.spectrum_ax.set_ylim(self.amp_min_input.value(), self.amp_max_input.value())
            self.spectrum_canvas.figure.tight_layout()
            self.spectrum_canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compute FFT: {str(e)}")

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

        title = QLabel("Medical Image Processing")
        title.setObjectName("PanelTitle")
        # REMOVED: title.setStyleSheet("...")
        layout.addWidget(title)

        controls_layout = QHBoxLayout()
        
        self.load_image_btn = QPushButton("Upload Image")
        self.load_image_btn.setToolTip("Load X-ray, MRI, or CT scan image.")
        self.load_image_btn.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_image_btn)

        self.grayscale_btn = QPushButton("Grayscale Conversion")
        self.grayscale_btn.setToolTip("Convert the image to grayscale.")
        self.grayscale_btn.clicked.connect(self.convert_to_grayscale)
        controls_layout.addWidget(self.grayscale_btn)

        blur_layout = QVBoxLayout()
        blur_layout.addWidget(QLabel("Smoothing Filter:"))
        self.blur_dropdown = QComboBox()
        self.blur_dropdown.addItems(["Gaussian Blur (15x15)", "Median Filter (5x5)"])
        self.blur_dropdown.setToolTip("Select the type of noise reduction filter to apply.")
        blur_layout.addWidget(self.blur_dropdown)
        self.apply_blur_btn = QPushButton("Apply Blur")
        self.apply_blur_btn.clicked.connect(self.apply_blur)
        blur_layout.addWidget(self.apply_blur_btn)
        controls_layout.addLayout(blur_layout)
        
        self.edge_btn = QPushButton("Canny Edge Detection")
        self.edge_btn.setToolTip("Apply Canny algorithm to detect edges.")
        self.edge_btn.clicked.connect(self.apply_edge_detection)
        controls_layout.addWidget(self.edge_btn)

        thresh_layout = QVBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold Level (0-255):"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(127)
        self.threshold_slider.setToolTip("Adjusts the pixel value for image binarization (segmentation).")
        thresh_layout.addWidget(self.threshold_slider)
        self.apply_threshold_btn = QPushButton("Apply Threshold")
        self.apply_threshold_btn.clicked.connect(self.apply_threshold)
        thresh_layout.addWidget(self.apply_threshold_btn)
        controls_layout.addLayout(thresh_layout)
        
        layout.addLayout(controls_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #ccc; margin: 10px 0;")
        layout.addWidget(separator)

        image_comparison_label = QLabel("Image Comparison")
        image_comparison_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        layout.addWidget(image_comparison_label)

        # Display images side by side - Original on LEFT, Processed on RIGHT
        self.image_layout = QHBoxLayout()
        self.image_layout.setSpacing(20)
        
        # Left side - Original Image
        original_container = QWidget()
        original_container_layout = QVBoxLayout(original_container)
        original_container_layout.setSpacing(5)
        
        original_title = QLabel("Original Image (Uploaded)")
        original_title.setAlignment(Qt.AlignCenter)
        original_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #333;")
        original_container_layout.addWidget(original_title)
        
        self.original_image_label = QLabel("No image loaded")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setObjectName("ImageDisplayLabelOriginal")
        self.original_image_label.setFixedSize(500, 500)
        self.original_image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #3498DB;
                border-radius: 5px;
                background-color: #f0f0f0;
                color: #888;
            }
        """)
        original_container_layout.addWidget(self.original_image_label)
        self.image_layout.addWidget(original_container)
        
        # Right side - Processed Image
        processed_container = QWidget()
        processed_container_layout = QVBoxLayout(processed_container)
        processed_container_layout.setSpacing(5)
        
        processed_title = QLabel("Processed Image (Result)")
        processed_title.setAlignment(Qt.AlignCenter)
        processed_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #333;")
        processed_container_layout.addWidget(processed_title)
        
        self.processed_image_label = QLabel("No processed image")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setObjectName("ImageDisplayLabelProcessed")
        self.processed_image_label.setFixedSize(500, 500)
        self.processed_image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #2ECC71;
                border-radius: 5px;
                background-color: #f0f0f0;
                color: #888;
            }
        """)
        processed_container_layout.addWidget(self.processed_image_label)
        self.image_layout.addWidget(processed_container)
        
        layout.addLayout(self.image_layout)
        
        info_label = QLabel("Note: Images are scaled to fit the display area. Original image is shown on the left, processed result on the right for easy comparison.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        return panel

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
            
            QMessageBox.information(self, "Image Loaded", f"Image loaded successfully from {os.path.basename(file_path)}")

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
        if self.cv_image is not None:
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            self.processed_cv_image = gray 
            self.display_image(self.processed_cv_image, self.processed_image_label)
            self.update_viz_image()
            
    def apply_blur(self):
        if self.cv_image is not None:
            blur_type = self.blur_dropdown.currentText()
            img = self.cv_image if len(self.cv_image.shape) == 3 else cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2BGR)

            if blur_type.startswith("Gaussian"):
                self.processed_cv_image = cv2.GaussianBlur(img, (15, 15), 0)
            elif blur_type.startswith("Median"):
                self.processed_cv_image = cv2.medianBlur(img, 5) 
                
            self.display_image(self.processed_cv_image, self.processed_image_label)
            self.update_viz_image()

    def apply_edge_detection(self):
        if self.cv_image is not None:
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            self.processed_cv_image = edges 
            self.display_image(self.processed_cv_image, self.processed_image_label)
            self.update_viz_image()

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


    # --- Panel 5: Visualization ---
    def create_data_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("Data Visualization")
        title.setObjectName("PanelTitle")
        # REMOVED: title.setStyleSheet("...")
        layout.addWidget(title)
        
        controls_group = QWidget()
        controls_group_layout = QGridLayout(controls_group)
        
        if self.df is not None and not self.df.empty:
            numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        else:
            numerical_cols = []

        # Scatter Plot Controls
        self.scatter_x_combo = QComboBox(); self.scatter_x_combo.addItems(numerical_cols if numerical_cols else [])
        self.scatter_y_combo = QComboBox(); self.scatter_y_combo.addItems(numerical_cols if numerical_cols else [])
        self.scatter_plot_btn = QPushButton("Plot Scatter")
        self.scatter_plot_btn.setToolTip("Plots relationship between two selected variables.")
        self.scatter_plot_btn.clicked.connect(self.plot_scatter)
        
        controls_group_layout.addWidget(QLabel("Scatter X:"), 0, 0); controls_group_layout.addWidget(self.scatter_x_combo, 0, 1)
        controls_group_layout.addWidget(QLabel("Scatter Y:"), 0, 2); controls_group_layout.addWidget(self.scatter_y_combo, 0, 3)
        controls_group_layout.addWidget(self.scatter_plot_btn, 0, 4)

        # Time-Series Plot Controls
        self.ts_column_combo = QComboBox(); self.ts_column_combo.addItems(numerical_cols if numerical_cols else [])
        self.ts_plot_btn = QPushButton("Plot Time-Series")
        self.ts_plot_btn.setToolTip("Plots the value of a variable over index/time.")
        self.ts_plot_btn.clicked.connect(self.plot_time_series)
        
        controls_group_layout.addWidget(QLabel("Time-Series:"), 1, 0); controls_group_layout.addWidget(self.ts_column_combo, 1, 1, 1, 3)
        controls_group_layout.addWidget(self.ts_plot_btn, 1, 4)

        # FFT Spectrum Plot Controls
        self.fft_column_combo = QComboBox(); self.fft_column_combo.addItems(numerical_cols if numerical_cols else [])
        self.fft_plot_btn = QPushButton("Plot FFT Spectrum")
        self.fft_plot_btn.setToolTip("Displays the frequency components of a signal.")
        self.fft_plot_btn.clicked.connect(self.plot_fft_viz) 
        
        controls_group_layout.addWidget(QLabel("FFT Spectrum:"), 2, 0); controls_group_layout.addWidget(self.fft_column_combo, 2, 1, 1, 3)
        controls_group_layout.addWidget(self.fft_plot_btn, 2, 4)
        
        # Heatmap Button
        self.heatmap_btn = QPushButton("Show Correlation Heatmap")
        self.heatmap_btn.setToolTip("Visualize correlations between all numerical variables.")
        self.heatmap_btn.clicked.connect(self.plot_heatmap)
        controls_group_layout.addWidget(self.heatmap_btn, 3, 0, 1, 5)

        layout.addWidget(controls_group)
        
        # Plot area wrapped in scroll area so visualization heatmaps/plots are fully visible
        # Reduced default size so the map looks more compact and balanced
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.figure.subplots_adjust(left=0.15, right=0.85, top=0.8, bottom=0.2)
        # More compact minimum height; scroll still available for larger plots
        self.canvas.setMinimumHeight(620)

        viz_scroll = QScrollArea()
        viz_scroll.setWidgetResizable(True)
        viz_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        viz_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        viz_scroll.setWidget(self.canvas)
        layout.addWidget(viz_scroll)
        
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
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data before plotting time-series.")
            return

        col = self.ts_column_combo.currentText()
        if not col:
            QMessageBox.warning(self, "Error", "Select a column for time-series.")
            return

        if col not in self.df.columns:
            QMessageBox.warning(self, "Invalid Column", f"Column '{col}' does not exist in the dataset.")
            return

        try:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                QMessageBox.warning(self, "Invalid Column", f"Column '{col}' is not numeric.")
                return

            data = self.df[col].dropna()
            if len(data) == 0:
                QMessageBox.warning(self, "No Data", f"Column '{col}' contains no valid numeric data.")
                return

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_box_aspect(0.6)
            ax.plot(data, color='#2ECC71')
            ax.set_xlabel("Index")
            ax.set_ylabel(col)
            ax.set_title(f"Time-Series Plot: {col}")
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot time-series: {str(e)}")

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