# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'chatbot_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5

class Ui_MainWindow(object):
    def __init__(self, **kwargs):

        if "window" in kwargs:
            self.window = kwargs.get("window")
        else:
            raise Exception("window is not found")

        if "model" in kwargs:
            self.model = kwargs.get("model")
        else:
            raise Exception("model is not found")

        if "device" in kwargs:
            self.device = kwargs.get("device")
        else:
            raise Exception("device is not found")

    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(837, 843)
        MainWindow.setMinimumSize(QtCore.QSize(837, 843))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.modelAutoLoadCheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.modelAutoLoadCheckBox.setGeometry(QtCore.QRect(530, 45, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.modelAutoLoadCheckBox.setFont(font)
        self.modelAutoLoadCheckBox.setChecked(False)
        self.modelAutoLoadCheckBox.setObjectName("modelAutoLoadCheckBox")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 100, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.modelPathTextBox = QtWidgets.QLineEdit(self.centralwidget)
        self.modelPathTextBox.setGeometry(QtCore.QRect(20, 50, 411, 20))
        self.modelPathTextBox.setReadOnly(True)
        self.modelPathTextBox.setObjectName("modelPathTextBox")
        self.modelSearchButton = QtWidgets.QPushButton(self.centralwidget)
        self.modelSearchButton.setGeometry(QtCore.QRect(440, 48, 75, 23))
        self.modelSearchButton.setObjectName("modelSearchButton")
        self.messageListBox = QtWidgets.QTextEdit(self.centralwidget)
        self.messageListBox.setGeometry(QtCore.QRect(20, 130, 471, 621))
        self.messageListBox.setReadOnly(True)
        self.messageListBox.setObjectName("messageListBox")
        self.sendMessageBox = QtWidgets.QTextEdit(self.centralwidget)
        self.sendMessageBox.setGeometry(QtCore.QRect(510, 130, 231, 81))
        self.sendMessageBox.setObjectName("sendMessageBox")
        self.sendMessageButton = QtWidgets.QPushButton(self.centralwidget)
        self.sendMessageButton.setGeometry(QtCore.QRect(750, 145, 75, 51))
        self.sendMessageButton.setObjectName("sendMessageButton")
        self.messageSaveDirectoryTextBox = QtWidgets.QLineEdit(self.centralwidget)
        self.messageSaveDirectoryTextBox.setGeometry(QtCore.QRect(20, 780, 411, 20))
        self.messageSaveDirectoryTextBox.setReadOnly(True)
        self.messageSaveDirectoryTextBox.setObjectName("messageSaveDirectoryTextBox")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 750, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.messageAutoSaveCheckBox = QtWidgets.QCheckBox(self.centralwidget)
        self.messageAutoSaveCheckBox.setGeometry(QtCore.QRect(200, 750, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.messageAutoSaveCheckBox.setFont(font)
        self.messageAutoSaveCheckBox.setChecked(False)
        self.messageAutoSaveCheckBox.setObjectName("messageAutoSaveCheckBox")
        self.messageListSaveButton = QtWidgets.QPushButton(self.centralwidget)
        self.messageListSaveButton.setGeometry(QtCore.QRect(130, 105, 91, 23))
        self.messageListSaveButton.setObjectName("messageListSaveButton")
        self.messageSaveDirectorySearchButton = QtWidgets.QPushButton(self.centralwidget)
        self.messageSaveDirectorySearchButton.setGeometry(QtCore.QRect(440, 778, 75, 23))
        self.messageSaveDirectorySearchButton.setObjectName("messageSaveDirectorySearchButton")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(600, 0, 41, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.dateLabel = QtWidgets.QLabel(self.centralwidget)
        self.dateLabel.setGeometry(QtCore.QRect(650, 6, 171, 20))
        self.dateLabel.setObjectName("dateLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 837, 21))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSave_Settings = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.actionSave_Settings.setFont(font)
        self.actionSave_Settings.setObjectName("actionSave_Settings")
        self.actionExit = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.actionExit.setFont(font)
        self.actionExit.setObjectName("actionExit")
        self.menuMenu.addAction(self.actionSave_Settings)
        self.menuMenu.addAction(self.actionExit)
        self.menubar.addAction(self.menuMenu.menuAction())

        # Event and text ---------------------------------------------------------------------------------------------------
        self.add_event()
        self.timerStart(MainWindow)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Movie Information Chatbot"))
        self.label.setText(_translate("MainWindow", "Model Path"))
        self.modelAutoLoadCheckBox.setText(_translate("MainWindow", "Model Auto Load"))
        self.label_2.setText(_translate("MainWindow", "Message List"))
        self.modelSearchButton.setText(_translate("MainWindow", "Search"))
        self.sendMessageButton.setText(_translate("MainWindow", "Send"))
        self.label_3.setText(_translate("MainWindow", "Message Save Directory"))
        self.messageAutoSaveCheckBox.setText(_translate("MainWindow", "Message Auto Save"))
        self.messageListSaveButton.setText(_translate("MainWindow", "Save Message"))
        self.messageSaveDirectorySearchButton.setText(_translate("MainWindow", "Search"))
        self.label_4.setText(_translate("MainWindow", "Date"))
        self.dateLabel.setText(_translate("MainWindow", "None"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.actionSave_Settings.setText(_translate("MainWindow", "Save Settings"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        MainWindow.setWindowIcon(QtGui.QIcon('img/chatbot.png'))

    # Function -------------------------------------------------------------------------------------------------------------
    def add_event(self):
        self.actionExit.triggered.connect(QtWidgets.qApp.quit)
        self.sendMessageButton.clicked.connect(self.send_message_button_clicked)

    
    def send_message_button_clicked(self):
        text = self.sendMessageBox.toPlainText()

        self.messageListBox.setTextColor(QtGui.QColor(255, 0, 0))
        self.messageListBox.setAlignment(QtCore.Qt.AlignRight)
        self.messageListBox.insertPlainText(text)
        self.messageListBox.append("")

        self.messageListBox.setTextColor(QtGui.QColor(0, 0, 255))
        self.messageListBox.setAlignment(QtCore.Qt.AlignLeft)
        self.messageListBox.insertPlainText("Reply")
        self.messageListBox.append("")

    
    def timerStart(self, MainWindow):
        self.timer = QtCore.QTimer(MainWindow)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.show_date)
        self.timer.start()

    
    def show_date(self):
        currentTime = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss ap ddd")
        self.dateLabel.setText(currentTime)

    
    def show_messagebox(title, text):
        QtWidgets.QMessageBox.information(None, title, text, 
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.NoButton)