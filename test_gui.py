from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout
import sys

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test GUI")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        label = QLabel("Hello! GUI is working.")
        label.setStyleSheet("font-size: 32px; font-weight: bold;")
        layout.addWidget(label)

app = QApplication(sys.argv)
window = TestWindow()
window.show()
sys.exit(app.exec_())