# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1031, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.load_image = QtWidgets.QPushButton(self.centralwidget)
        self.load_image.setGeometry(QtCore.QRect(10, 100, 121, 31))
        self.load_image.setObjectName("load_image")
        self.load_video = QtWidgets.QPushButton(self.centralwidget)
        self.load_video.setGeometry(QtCore.QRect(10, 150, 121, 31))
        self.load_video.setObjectName("load_video")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(140, 30, 221, 80))
        self.groupBox.setObjectName("groupBox")
        self.button1 = QtWidgets.QPushButton(self.groupBox)
        self.button1.setGeometry(QtCore.QRect(10, 30, 201, 31))
        self.button1.setObjectName("button1")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(140, 140, 221, 111))
        self.groupBox_2.setObjectName("groupBox_2")
        self.button2_1 = QtWidgets.QPushButton(self.groupBox_2)
        self.button2_1.setGeometry(QtCore.QRect(10, 30, 201, 31))
        self.button2_1.setObjectName("button2_1")
        self.button2_2 = QtWidgets.QPushButton(self.groupBox_2)
        self.button2_2.setGeometry(QtCore.QRect(10, 70, 201, 31))
        self.button2_2.setObjectName("button2_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(140, 290, 221, 80))
        self.groupBox_3.setObjectName("groupBox_3")
        self.button3 = QtWidgets.QPushButton(self.groupBox_3)
        self.button3.setGeometry(QtCore.QRect(10, 30, 201, 31))
        self.button3.setObjectName("button3")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(370, 30, 631, 251))
        self.groupBox_4.setObjectName("groupBox_4")
        self.button4_1 = QtWidgets.QPushButton(self.groupBox_4)
        self.button4_1.setGeometry(QtCore.QRect(20, 30, 251, 31))
        self.button4_1.setObjectName("button4_1")
        self.button4_2 = QtWidgets.QPushButton(self.groupBox_4)
        self.button4_2.setGeometry(QtCore.QRect(20, 70, 251, 31))
        self.button4_2.setObjectName("button4_2")
        self.button4_3 = QtWidgets.QPushButton(self.groupBox_4)
        self.button4_3.setGeometry(QtCore.QRect(20, 110, 251, 31))
        self.button4_3.setObjectName("button4_3")
        self.button4_4 = QtWidgets.QPushButton(self.groupBox_4)
        self.button4_4.setGeometry(QtCore.QRect(20, 150, 251, 31))
        self.button4_4.setObjectName("button4_4")
        self.label4 = QtWidgets.QLabel(self.groupBox_4)
        self.label4.setGeometry(QtCore.QRect(26, 190, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label4.setFont(font)
        self.label4.setText("")
        self.label4.setAlignment(QtCore.Qt.AlignCenter)
        self.label4.setObjectName("label4")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(370, 300, 631, 251))
        self.groupBox_5.setObjectName("groupBox_5")
        self.button5_0 = QtWidgets.QPushButton(self.groupBox_5)
        self.button5_0.setGeometry(QtCore.QRect(20, 30, 251, 31))
        self.button5_0.setObjectName("button5_0")
        self.button5_1 = QtWidgets.QPushButton(self.groupBox_5)
        self.button5_1.setGeometry(QtCore.QRect(20, 70, 251, 31))
        self.button5_1.setObjectName("button5_1")
        self.button5_2 = QtWidgets.QPushButton(self.groupBox_5)
        self.button5_2.setGeometry(QtCore.QRect(20, 110, 251, 31))
        self.button5_2.setObjectName("button5_2")
        self.button5_3 = QtWidgets.QPushButton(self.groupBox_5)
        self.button5_3.setGeometry(QtCore.QRect(20, 150, 251, 31))
        self.button5_3.setObjectName("button5_3")
        self.button5_4 = QtWidgets.QPushButton(self.groupBox_5)
        self.button5_4.setGeometry(QtCore.QRect(20, 190, 251, 31))
        self.button5_4.setObjectName("button5_4")
        self.label5_1 = QtWidgets.QLabel(self.groupBox_5)
        self.label5_1.setGeometry(QtCore.QRect(400, 20, 161, 161))
        self.label5_1.setText("")
        self.label5_1.setObjectName("label5_1")
        self.label5_2 = QtWidgets.QLabel(self.groupBox_5)
        self.label5_2.setGeometry(QtCore.QRect(430, 200, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label5_2.setFont(font)
        self.label5_2.setObjectName("label5_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1031, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        from application.xml.DrawingPad import DrawingPad
        self.drawingPad = DrawingPad(self.centralwidget)
        DrawingPad.set_global_drawing_pad(self.drawingPad)
        self.drawingPad.setGeometry(QtCore.QRect(self.drawingPad.offset[0], self.drawingPad.offset[1], self.drawingPad.shape[0], self.drawingPad.shape[1]))
        self.drawingPad.setObjectName("drawingPad")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.load_image.setText(_translate("MainWindow", "Load Image"))
        self.load_video.setText(_translate("MainWindow", "Load Video"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Background Subtraction"))
        self.button1.setText(_translate("MainWindow", "1. Background Subtraction"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Optical Flow"))
        self.button2_1.setText(_translate("MainWindow", "2.1 Preprocessing"))
        self.button2_2.setText(_translate("MainWindow", "2.2 Video Tracking"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3. PCA"))
        self.button3.setText(_translate("MainWindow", "3. Dimension Reduction"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. MNIST Classifier Using VGG19"))
        self.button4_1.setText(_translate("MainWindow", "1. Show Model Structure"))
        self.button4_2.setText(_translate("MainWindow", "2. Show Accuracy an Loss"))
        self.button4_3.setText(_translate("MainWindow", "3. Predict"))
        self.button4_4.setText(_translate("MainWindow", "4. Reset"))
        self.groupBox_5.setTitle(_translate("MainWindow", "5. ResNet50"))
        self.button5_0.setText(_translate("MainWindow", "Load Image"))
        self.button5_1.setText(_translate("MainWindow", "5.1 Show Images"))
        self.button5_2.setText(_translate("MainWindow", "5.2 Show Model Structure"))
        self.button5_3.setText(_translate("MainWindow", "5.3 Show Comprasion"))
        self.button5_4.setText(_translate("MainWindow", "5.4 Inference"))
        self.label5_2.setText(_translate("MainWindow", "Prediction:"))
