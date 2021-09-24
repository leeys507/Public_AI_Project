# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'chatbot_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from genericpath import exists
from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5

import torch

from model import Combination

from general import load_pretrained_weights, load_checkpoint, sentence_prediction

import os
from reply import SCRIPT_LIST

class Ui_MainWindow(object):
    def __init__(self, **kwargs):

        if "window" in kwargs:
            self.window = kwargs.get("window")
        else:
            raise Exception("window is not found")

        self.model = None
        self.device = None

    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(837, 843)
        MainWindow.setMinimumSize(QtCore.QSize(837, 843))
        MainWindow.setMaximumSize(QtCore.QSize(837, 843))
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
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
        self.saveSettingsButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveSettingsButton.setGeometry(QtCore.QRect(724, 60, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.saveSettingsButton.setFont(font)
        self.saveSettingsButton.setStyleSheet("color:red")
        self.saveSettingsButton.setObjectName("saveSettingsButton")
        self.modelLoadButton = QtWidgets.QPushButton(self.centralwidget)
        self.modelLoadButton.setGeometry(QtCore.QRect(440, 80, 75, 23))
        self.modelLoadButton.setObjectName("modelLoadButton")
        self.manualButton = QtWidgets.QPushButton(self.centralwidget)
        self.manualButton.setGeometry(QtCore.QRect(510, 240, 161, 61))
        self.manualButton.setObjectName("manualButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 837, 23))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.actionExit.setFont(font)
        self.actionExit.setObjectName("actionExit")
        self.menuMenu.addAction(self.actionExit)
        self.menubar.addAction(self.menuMenu.menuAction())

        # Event and Settings ---------------------------------------------------------------------------------------------------
        self.add_event()
        self.load_settings()
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
        self.saveSettingsButton.setText(_translate("MainWindow", "Save Settings"))
        self.modelLoadButton.setText(_translate("MainWindow", "Load"))
        self.manualButton.setText(_translate("MainWindow", "Manual"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))

        MainWindow.setWindowIcon(QtGui.QIcon('img/chatbot.png'))

        self.sendMessageButton.setDisabled(True)

    # Function -------------------------------------------------------------------------------------------------------------
    def add_event(self):
        self.actionExit.triggered.connect(QtWidgets.qApp.quit)
        self.sendMessageButton.clicked.connect(self.send_message_button_clicked)
        self.sendMessageBox.keyPressEvent = self.send_message_keypress_event
        self.messageSaveDirectorySearchButton.clicked.connect(self.search_msg_save_folder_clicked)
        self.modelSearchButton.clicked.connect(self.search_model_folder_clicked)
        self.saveSettingsButton.clicked.connect(self.save_settings_clicked)
        self.modelLoadButton.clicked.connect(self.load_model)

    
    def send_message_button_clicked(self):
        text = self.sendMessageBox.toPlainText()
        self.sendMessageBox.clear()

        if self.model is None:
            show_messagebox("오류", "model load를 해주세요")
            self.sendMessageButton.setDisabled(True)
            return

        self.messageListBox.setTextColor(QtGui.QColor(255, 0, 0))
        self.messageListBox.setAlignment(QtCore.Qt.AlignRight)
        self.messageListBox.insertPlainText(text)
        self.messageListBox.append("")

        self.messageListBox.setTextColor(QtGui.QColor(0, 0, 255))
        self.messageListBox.setAlignment(QtCore.Qt.AlignLeft)
        self.messageListBox.insertPlainText("Reply")
        self.messageListBox.append("")
        self.sendMessageButton.setDisabled(True)
        self.sendMessageBox.setFocus()

    
    def timerStart(self, MainWindow):
        self.timer = QtCore.QTimer(MainWindow)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.show_date)
        self.timer.start()

    
    def show_date(self):
        currentTime = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss ap ddd")
        self.dateLabel.setText(currentTime)


    def send_message_keypress_event(self, e):
        if e.key() == QtCore.Qt.Key_Return and not (e.modifiers() & QtCore.Qt.ShiftModifier):
            if self.sendMessageButton.isEnabled() == True:
                self.sendMessageButton.click()
                self.sendMessageBox.clear()
                self.sendMessageButton.setDisabled(True)
        else:
            QtWidgets.QTextEdit.keyPressEvent(self.sendMessageBox, e)
            temp_text = self.sendMessageBox.toPlainText()
            
            temp_text = temp_text.replace(" ", "").replace("\n", "")

            if temp_text == "":
                self.sendMessageButton.setDisabled(True)
            else:
                self.sendMessageButton.setEnabled(True)


    def search_msg_save_folder_clicked(self):
        default_path = os.path.join(os.path.expanduser('~'), 'Desktop/')
        fname = QtWidgets.QFileDialog.getExistingDirectory(self.window, 'Search Folder', default_path)

        if fname:
            self.messageSaveDirectoryTextBox.setText(fname)

    
    def search_model_folder_clicked(self):
        default_path = os.path.join(os.path.expanduser('~'), 'Desktop/')
        filter = "model file(*.pt *.pth .*pwf)"

        fname = QtWidgets.QFileDialog.getOpenFileName(self.window, 'Search File', default_path, filter=filter)
        
        if fname[0]:
            self.modelPathTextBox.setText(fname[0])


    def save_settings_clicked(self):
        with open("Settings.dat", "wb") as f:
            f.write((self.modelPathTextBox.text() + "\n").encode())
            f.write("1\n".encode() if self.modelAutoLoadCheckBox.isChecked() else "0\n".encode())
            f.write((self.messageSaveDirectoryTextBox.text() + "\n").encode())
            f.write("1\n".encode() if self.messageAutoSaveCheckBox.isChecked() else "0\n".encode())
        
        show_messagebox("확인", "설정 저장 완료")


    def load_settings(self):
        if os.path.exists("Settings.dat"):
            with open("Settings.dat", "rb") as f:
                self.modelPathTextBox.setText(f.readline().decode().replace("\n", ""))
                self.modelAutoLoadCheckBox.setChecked(True if f.readline().decode() == "1\n" else False)
                self.messageSaveDirectoryTextBox.setText(f.readline().decode().replace("\n", ""))
                self.messageAutoSaveCheckBox.setChecked(True if f.readline().decode() == "1\n" else False)
            
            if self.modelAutoLoadCheckBox.isChecked():
                show_messagebox("확인", "model auto load")
                self.load_model()

    
    def load_model(self):
        path = self.modelPathTextBox.text()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        classes = ["unknown", "hello", "manual", "title", "actor", "director", "rank", "year", "country"]

        try:
            if self.model is not None:
                show_messagebox("이미 model load 되었습니다")
            elif path is not None and path != "":
                # load vocab
                vocab, weight = load_pretrained_weights(path, self.device)

                self.model = Combination(vocab, len(vocab), class_num=len(classes), embed_dim=512, n_filters=128).to(self.device)
                load_checkpoint(path, self.model, self.device, optimizer=None, strict=False)
                show_messagebox("확인", "model 로드 완료")
            else:
                show_messagebox("오류", "model 경로를 지정하세요")

        except Exception as e:
            show_messagebox("오류", e)


def show_messagebox(title, text):
    QtWidgets.QMessageBox.information(None, title, text, 
        QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.NoButton)