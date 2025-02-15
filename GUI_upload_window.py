# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'upload_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_UploadWindow(object):
    def setupUi(self, UploadWindow):
        UploadWindow.setObjectName("UploadWindow")
        UploadWindow.resize(400, 200)
        self.centralwidget = QtWidgets.QWidget(UploadWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.image_pushButton.setGeometry(QtCore.QRect(50, 50, 120, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.image_pushButton.setFont(font)
        self.image_pushButton.setObjectName("image_pushButton")
        self.file_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.file_pushButton.setGeometry(QtCore.QRect(220, 50, 120, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.file_pushButton.setFont(font)
        self.file_pushButton.setObjectName("file_pushButton")
        self.cancel_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.cancel_pushButton.setGeometry(QtCore.QRect(280, 120, 80, 40))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cancel_pushButton.setFont(font)
        self.cancel_pushButton.setObjectName("cancel_pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 30, 330, 80))
        self.label.setStyleSheet("background-color: rgb(85, 255, 127);")
        self.label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label.setText("")
        self.label.setObjectName("label")
        self.label.raise_()
        self.image_pushButton.raise_()
        self.file_pushButton.raise_()
        self.cancel_pushButton.raise_()
        UploadWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(UploadWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 400, 26))
        self.menubar.setObjectName("menubar")
        UploadWindow.setMenuBar(self.menubar)

        self.retranslateUi(UploadWindow)
        QtCore.QMetaObject.connectSlotsByName(UploadWindow)

    def retranslateUi(self, UploadWindow):
        _translate = QtCore.QCoreApplication.translate
        UploadWindow.setWindowTitle(_translate("UploadWindow", "MainWindow"))
        self.image_pushButton.setText(_translate("UploadWindow", "Image"))
        self.file_pushButton.setText(_translate("UploadWindow", "File"))
        self.cancel_pushButton.setText(_translate("UploadWindow", "Cancel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    UploadWindow = QtWidgets.QMainWindow()
    ui = Ui_UploadWindow()
    ui.setupUi(UploadWindow)
    UploadWindow.show()
    sys.exit(app.exec_())
