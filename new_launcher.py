import subprocess
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QProcess, Qt
from GUI_LAUNCHER import Ui_LAUNCHER_MainWindow


class LoadingWindow(QWidget):
    def __init__(self):
        super(LoadingWindow, self).__init__()
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setWindowTitle('Loading')
        self.setFixedSize(200, 100)
        layout = QVBoxLayout(self)
        self.label = QLabel('Loading, please wait...', self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)


class MainWindow(QMainWindow, Ui_LAUNCHER_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.exit_button.clicked.connect(self.confirm_exit)
        self.emo_button.clicked.connect(self.open_emo)
        self.doctor_button.clicked.connect(self.open_doctor)
        self.process = QProcess(self)
        self.loading_window = LoadingWindow()

        self.process.readyRead.connect(self.on_ready_read)
        self.process.finished.connect(self.on_new_main_finished)

    def open_emo(self):
        self.hide()
        self.loading_window.show()
        self.process.start(sys.executable, ['new_main.py'])

    def open_doctor(self):
        self.hide()
        self.loading_window.show()
        self.process.start(sys.executable, ['new_doctor.py'])

    def confirm_exit(self):
        reply = QMessageBox.question(self, 'Xác nhận thoát', 'Bạn có muốn tắt ứng dụng không?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            sys.exit()

    def on_ready_read(self):
        self.loading_window.hide()

    def on_new_main_finished(self):
        self.loading_window.hide()
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
