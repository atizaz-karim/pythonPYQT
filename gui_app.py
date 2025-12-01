# gui_app.py (CLEANED VERSION - NO STYLES)

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QTabWidget, QFileDialog, QTableWidget, 
    QTableWidgetItem, QScrollArea, QSlider, QLineEdit, QCheckBox, 
    QSpinBox, QMessageBox, QGridLayout, QInputDialog
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

# --- Placeholder for DB Manager ---
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
        
        self.filtered_df = df.copy()
        self.cv_image = None
        self.processed_cv_image = None
        
        self.setWindowTitle("Healthcare Data and Medical Image Processing Tool")
        self.setGeometry(50, 50, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 1. Sidebar (Left Navigation) ---
        self.sidebar = QWidget()
        # SET OBJECT NAME FOR QSS TARGETING
        self.sidebar.setObjectName("SidebarWidget") 
        self.sidebar.setFixedWidth(250)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        
        # REMOVED: self.sidebar.setStyleSheet("...") 

        # --- 2. Main Content Area ---
        self.stacked_widget = QTabWidget() 
        self.stacked_widget.tabBar().setVisible(False) 

        # Add Tabs (Panels)
        self.stacked_widget.addTab(self.create_data_management_panel(), "Data Loading and Management")
        self.stacked_widget.addTab(self.create_analysis_panel(), "Health Data Analysis")
        self.stacked_widget.addTab(self.create_spectrum_panel(), "Spectrum Analysis")
        self.stacked_widget.addTab(self.create_image_processing_panel(), "Medical Image Processing")
        self.stacked_widget.addTab(self.create_data_visualization_panel(), "Data Visualization")
        
        # --- 3. Sidebar Buttons and Connection ---
        tab_names = ["Patient Data Management", "Health Data Analysis", "Spectrum Analysis", 
                     "Image Processing", "Data Visualization"]
                     
        for i, name in enumerate(tab_names):
            btn = QPushButton(name)
            btn.setObjectName(f"SidebarBtn_{i}") # SET OBJECT NAME FOR QSS TARGETING
            btn.setToolTip(f"Switch to the {name} section.")
            # REMOVED: btn.setStyleSheet("...") 
            btn.clicked.connect(lambda checked, index=i: self.stacked_widget.setCurrentIndex(index))
            self.sidebar_layout.addWidget(btn)
        
        self.sidebar_layout.addStretch(1) 

        # Add components to the main layout
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stacked_widget)
        
        self._setup_menu_bar()

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
        self.table_widget.clear()
        if df.empty:
            self.table_widget.setRowCount(0)
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

        self.db_connect_btn = QPushButton("Retrieve Data from DB")
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
        # REMOVED: self.table_widget.setStyleSheet("...") (Moved to QSS)
        layout.addWidget(self.table_widget)
        self.populate_table(self.df) 

        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)

        return panel

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.filtered_df = self.df.copy() 
                self.populate_table(self.df)
                QMessageBox.information(self, "Success", f"Data loaded successfully from {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Data", f"Failed to load CSV: {e}")

    # --- DB HANDLERS (Unchanged logic) ---
    def _check_db_manager(self):
        if not self.db_manager:
            QMessageBox.critical(self, "DB Error", "Database Manager is not initialized. Check your main.py setup.")
            return False
        return True

    def db_retrieve_data(self):
        if not self._check_db_manager(): return
        try:
            retrieved_df = self.db_manager.get_patient_data() 
            self.df = retrieved_df
            self.populate_table(self.df)
            QMessageBox.information(self, "Success", f"Retrieved {len(retrieved_df)} records from the database.")
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

        title = QLabel("Health Data Analysis")
        title.setObjectName("PanelTitle")
        # REMOVED: title.setStyleSheet("...") 
        main_layout.addWidget(title)

        controls_group = QHBoxLayout()

        # --- Left Side: Filtering and Time-series ---
        filtering_widget = QWidget()
        filtering_widget.setObjectName("FilterGroup") # SET OBJECT NAME FOR QSS TARGETING
        # REMOVED: filtering_widget.setStyleSheet("...") 
        filtering_group = QVBoxLayout(filtering_widget)
        filtering_group.addWidget(QLabel("### Data Filtering"))
        
        filtering_group.addWidget(QLabel("Outlier Removal Threshold (IQR Factor):"))
        self.outlier_slider = QSlider(Qt.Horizontal)
        self.outlier_slider.setRange(10, 50) 
        self.outlier_slider.setValue(15) 
        self.outlier_slider.setToolTip("Adjusts the factor applied to the Interquartile Range (IQR) for outlier detection (1.0 to 5.0).")
        filtering_group.addWidget(self.outlier_slider)

        filtering_group.addWidget(QLabel("Time-series Plotting:"))
        ts_controls = QHBoxLayout()
        self.ts_analysis_column = QComboBox()
        self.ts_analysis_column.addItems(self.df.select_dtypes(include='number').columns)
        self.ts_raw_checkbox = QCheckBox("Show Raw Data")
        self.ts_raw_checkbox.setChecked(True)
        self.ts_raw_checkbox.setToolTip("Check to plot raw data, uncheck to plot filtered data.")
        ts_controls.addWidget(self.ts_analysis_column)
        ts_controls.addWidget(self.ts_raw_checkbox)
        filtering_group.addLayout(ts_controls)
        
        self.apply_filter_btn = QPushButton("Apply Filters & Plot Time-series")
        self.apply_filter_btn.setToolTip("Applies the selected filters to the dataset.")
        self.apply_filter_btn.clicked.connect(self.apply_filters_and_plot)
        self.reset_filter_btn = QPushButton("Reset All Filters")
        self.reset_filter_btn.setToolTip("Resets the working data back to the original loaded DataFrame.")
        self.reset_filter_btn.clicked.connect(self.reset_filters)
        
        filter_btn_layout = QHBoxLayout()
        filter_btn_layout.addWidget(self.apply_filter_btn)
        filter_btn_layout.addWidget(self.reset_filter_btn)
        filtering_group.addLayout(filter_btn_layout)
        
        controls_group.addWidget(filtering_widget)
        
        # --- Right Side: Correlation Analysis ---
        corr_widget = QWidget()
        corr_widget.setObjectName("CorrGroup") # SET OBJECT NAME FOR QSS TARGETING
        # REMOVED: corr_widget.setStyleSheet("...") 
        corr_group = QVBoxLayout(corr_widget)
        corr_group.addWidget(QLabel("### Correlation Analysis"))
        
        self.metric1_dropdown = QComboBox()
        self.metric1_dropdown.addItems(self.df.select_dtypes(include='number').columns)
        self.metric2_dropdown = QComboBox()
        self.metric2_dropdown.addItems(self.df.select_dtypes(include='number').columns)
        
        corr_group.addWidget(QLabel("Metric 1:"))
        corr_group.addWidget(self.metric1_dropdown)
        corr_group.addWidget(QLabel("Metric 2:"))
        corr_group.addWidget(self.metric2_dropdown)

        self.compute_corr_btn = QPushButton("Compute Scatter Plot")
        self.compute_corr_btn.setToolTip("Calculates correlation and displays a scatter plot.")
        self.compute_corr_btn.clicked.connect(self.compute_correlation_plot)
        self.show_heatmap_btn = QPushButton("Show Correlation Heatmap")
        self.show_heatmap_btn.setToolTip("Displays a heatmap of all numerical variable correlations.")
        self.show_heatmap_btn.clicked.connect(self.show_heatmap)
        
        corr_btn_layout = QHBoxLayout()
        corr_btn_layout.addWidget(self.compute_corr_btn)
        corr_btn_layout.addWidget(self.show_heatmap_btn)
        corr_group.addLayout(corr_btn_layout)
        
        controls_group.addWidget(corr_widget)
        main_layout.addLayout(controls_group)

        # Plot area
        self.analysis_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        main_layout.addWidget(self.analysis_canvas)
        self.analysis_ax = self.analysis_canvas.figure.add_subplot(111)

        self.analysis_status_label = QLabel("Ready for analysis.")
        main_layout.addWidget(self.analysis_status_label)

        return panel

    def apply_filters_and_plot(self):
        factor = self.outlier_slider.value() / 10.0
        self.filtered_df = self.df.copy() 

        col = self.ts_analysis_column.currentText()
        data_to_plot = self.df[col].values if self.ts_raw_checkbox.isChecked() else self.filtered_df[col].values
        
        self.analysis_ax.clear()
        # NOTE: Matplotlib colors are NOT controlled by QSS
        self.analysis_ax.plot(data_to_plot, color='green' if self.ts_raw_checkbox.isChecked() else '#3498DB')
        self.analysis_ax.set_title(f"Time-series: {col} ({'Raw' if self.ts_raw_checkbox.isChecked() else 'Filtered'})")
        self.analysis_canvas.draw()
        self.analysis_status_label.setText(f"Filters applied (factor: {factor}). Time-series plotted.")
        
    def reset_filters(self):
        self.filtered_df = self.df.copy()
        self.outlier_slider.setValue(15)
        self.analysis_status_label.setText("All filters reset. Data returned to raw state.")

    def compute_correlation_plot(self):
        col1 = self.metric1_dropdown.currentText()
        col2 = self.metric2_dropdown.currentText()
        if col1 and col2:
            self.analysis_ax.clear()
            self.filtered_df.plot.scatter(x=col1, y=col2, ax=self.analysis_ax)
            corr_value = self.filtered_df[col1].corr(self.filtered_df[col2])
            self.analysis_ax.set_title(f'Scatter: {col1} vs {col2} (corr={corr_value:.2f})')
            self.analysis_canvas.draw()
            self.analysis_status_label.setText(f"Correlation between {col1} and {col2}: {corr_value:.2f}")

    def show_heatmap(self):
        self.analysis_ax.clear()
        corr = self.filtered_df.select_dtypes(include='number').corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=self.analysis_ax)
        self.analysis_ax.set_title("Correlation Heatmap")
        self.analysis_canvas.draw()
        self.analysis_status_label.setText("Correlation heatmap displayed.")


    # --- Panel 3: Spectrum Analysis ---
    def create_spectrum_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("Spectrum Analysis")
        title.setObjectName("PanelTitle")
        # REMOVED: title.setStyleSheet("...")
        layout.addWidget(title)
        
        control_layout = QHBoxLayout()

        signal_controls = QVBoxLayout()
        self.signal_dropdown = QComboBox()
        numeric_cols = self.df.select_dtypes(include='number').columns
        self.signal_dropdown.addItems(numeric_cols)
        signal_controls.addWidget(QLabel("Select Signal Column:"))
        signal_controls.addWidget(self.signal_dropdown)
        control_layout.addLayout(signal_controls)

        time_window_controls = QVBoxLayout()
        time_window_controls.addWidget(QLabel("FFT Time Window (Samples):"))
        self.time_window_slider = QSlider(Qt.Horizontal)
        self.time_window_slider.setRange(10, len(self.df))
        self.time_window_slider.setValue(min(256, len(self.df)))
        self.time_window_slider.setSingleStep(10)
        self.time_window_slider.setToolTip(f"Adjusts the number of samples (Max: {len(self.df)}) used for FFT computation.")
        time_window_controls.addWidget(self.time_window_slider)
        control_layout.addLayout(time_window_controls)
        
        button_layout = QVBoxLayout()
        self.plot_signal_btn = QPushButton("Plot Raw Signal")
        self.plot_signal_btn.setToolTip("Displays the selected signal over time/index.")
        self.plot_signal_btn.clicked.connect(self.plot_raw_signal)
        self.fft_btn = QPushButton("Compute FFT Spectrum")
        self.fft_btn.setToolTip("Calculates and displays the Fast Fourier Transform (Power Spectrum).")
        self.fft_btn.clicked.connect(self.plot_fft)
        button_layout.addWidget(self.plot_signal_btn)
        button_layout.addWidget(self.fft_btn)
        control_layout.addLayout(button_layout)
        
        layout.addLayout(control_layout)

        # Plot area
        self.spectrum_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        layout.addWidget(self.spectrum_canvas)
        self.spectrum_ax = self.spectrum_canvas.figure.add_subplot(111)

        return panel

    def plot_raw_signal(self):
        col = self.signal_dropdown.currentText()
        if col:
            signal = self.df[col].values
            self.spectrum_ax.clear()
            self.spectrum_ax.plot(signal, color='blue')
            self.spectrum_ax.set_title(f"Raw Signal: {col}")
            self.spectrum_ax.set_xlabel("Sample Index")
            self.spectrum_ax.set_ylabel(col)
            self.spectrum_canvas.draw()

    def plot_fft(self):
        col = self.signal_dropdown.currentText()
        window_size = self.time_window_slider.value()
        
        if col:
            signal = self.df[col].values
            signal_segment = signal[:window_size] 
            
            n = len(signal_segment)
            T = 1.0
            freq = np.fft.rfftfreq(n, d=T)
            fft_vals = np.abs(np.fft.rfft(signal_segment))

            self.spectrum_ax.clear()
            self.spectrum_ax.plot(freq, 2.0/n * fft_vals, color='red')
            self.spectrum_ax.set_title(f"FFT Spectrum: {col} (Window: {n} samples)")
            self.spectrum_ax.set_xlabel("Frequency")
            self.spectrum_ax.set_ylabel("Amplitude")
            self.spectrum_canvas.draw()

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

        # Display images side by side
        self.image_layout = QHBoxLayout()
        
        self.original_image_label = QLabel("Original Image")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setObjectName("ImageDisplayLabelOriginal") # SET OBJECT NAME FOR QSS TARGETING
        self.original_image_label.setFixedSize(450, 450)
        # REMOVED: self.original_image_label.setStyleSheet("...") 
        
        self.processed_image_label = QLabel("Processed Image")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setObjectName("ImageDisplayLabelProcessed") # SET OBJECT NAME FOR QSS TARGETING
        self.processed_image_label.setFixedSize(450, 450)
        # REMOVED: self.processed_image_label.setStyleSheet("...") 
        
        self.image_layout.addWidget(self.original_image_label)
        self.image_layout.addWidget(self.processed_image_label)
        layout.addLayout(self.image_layout)
        
        layout.addWidget(QLabel("Note: Processed images are scaled for display. Use visualization tab for interactive analysis."))

        return panel

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Medical Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.dcm);;All Files (*)")
        if file_path:
            self.cv_image = cv2.imread(file_path)
            if self.cv_image is None:
                QMessageBox.critical(self, "Image Error", "Could not load image file.")
                return
                
            self.processed_cv_image = self.cv_image.copy()
            self.display_image(self.cv_image, self.original_image_label)
            self.display_image(self.processed_cv_image, self.processed_image_label)

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
            
    def apply_blur(self):
        if self.cv_image is not None:
            blur_type = self.blur_dropdown.currentText()
            img = self.cv_image if len(self.cv_image.shape) == 3 else cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2BGR)

            if blur_type.startswith("Gaussian"):
                self.processed_cv_image = cv2.GaussianBlur(img, (15, 15), 0)
            elif blur_type.startswith("Median"):
                self.processed_cv_image = cv2.medianBlur(img, 5) 
                
            self.display_image(self.processed_cv_image, self.processed_image_label)

    def apply_edge_detection(self):
        if self.cv_image is not None:
            gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            self.processed_cv_image = edges 
            self.display_image(self.processed_cv_image, self.processed_image_label)

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
        
        numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()

        # Scatter Plot Controls
        self.scatter_x_combo = QComboBox(); self.scatter_x_combo.addItems(numerical_cols)
        self.scatter_y_combo = QComboBox(); self.scatter_y_combo.addItems(numerical_cols)
        self.scatter_plot_btn = QPushButton("Plot Scatter")
        self.scatter_plot_btn.setToolTip("Plots relationship between two selected variables.")
        self.scatter_plot_btn.clicked.connect(self.plot_scatter)
        
        controls_group_layout.addWidget(QLabel("Scatter X:"), 0, 0); controls_group_layout.addWidget(self.scatter_x_combo, 0, 1)
        controls_group_layout.addWidget(QLabel("Scatter Y:"), 0, 2); controls_group_layout.addWidget(self.scatter_y_combo, 0, 3)
        controls_group_layout.addWidget(self.scatter_plot_btn, 0, 4)

        # Time-Series Plot Controls
        self.ts_column_combo = QComboBox(); self.ts_column_combo.addItems(numerical_cols)
        self.ts_plot_btn = QPushButton("Plot Time-Series")
        self.ts_plot_btn.setToolTip("Plots the value of a variable over index/time.")
        self.ts_plot_btn.clicked.connect(self.plot_time_series)
        
        controls_group_layout.addWidget(QLabel("Time-Series:"), 1, 0); controls_group_layout.addWidget(self.ts_column_combo, 1, 1, 1, 3)
        controls_group_layout.addWidget(self.ts_plot_btn, 1, 4)

        # FFT Spectrum Plot Controls
        self.fft_column_combo = QComboBox(); self.fft_column_combo.addItems(numerical_cols)
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
        
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.viz_image_label = QLabel("Image Visualization Placeholder")
        self.viz_image_label.setAlignment(Qt.AlignCenter)
        self.viz_image_label.setFixedSize(900, 300) 
        layout.addWidget(self.viz_image_label) 

        return panel

    def plot_scatter(self):
        x_col = self.scatter_x_combo.currentText(); y_col = self.scatter_y_combo.currentText()
        if not x_col or not y_col: QMessageBox.warning(self, "Error", "Select both X and Y columns."); return
        self.figure.clear(); ax = self.figure.add_subplot(111)
        # NOTE: Matplotlib colors are NOT controlled by QSS
        ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6, color='#3498DB')
        ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        self.canvas.draw()
        
    def plot_heatmap(self):
        self.figure.clear(); ax = self.figure.add_subplot(111)
        numerical_df = self.df.select_dtypes(include=np.number); corr = numerical_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap"); self.canvas.draw()

    def plot_time_series(self):
        col = self.ts_column_combo.currentText()
        if not col: QMessageBox.warning(self, "Error", "Select a column for time-series."); return
        self.figure.clear(); ax = self.figure.add_subplot(111)
        ax.plot(self.df[col], color='#2ECC71'); ax.set_xlabel("Index"); ax.set_ylabel(col)
        ax.set_title(f"Time-Series Plot: {col}"); self.canvas.draw()

    def plot_fft_viz(self):
        col = self.fft_column_combo.currentText()
        if not col: QMessageBox.warning(self, "Error", "Select a column for FFT."); return
        self.figure.clear(); ax = self.figure.add_subplot(111)
        y = self.df[col].values; N = len(y); T = 1.0; yf = np.fft.fft(y)
        xf = np.fft.fftfreq(N, T)[:N//2]
        ax.plot(xf, 2.0/N * np.abs(yf[:N//2]), color='#E74C3C'); ax.set_xlabel("Frequency"); ax.set_ylabel("Amplitude")
        ax.set_title(f"FFT Spectrum: {col}"); self.canvas.draw()

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