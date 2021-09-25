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

from general import load_pretrained_weights, load_checkpoint, sentence_prediction, show_messagebox
from chatbot_get_reply_information import get_title, get_actor, get_director, get_rank, get_year, get_country

import os
import random
import datetime
from unicodedata import normalize
from eunjeon import Mecab
from reply import SCRIPT_LIST


class Ui_MainWindow(object):
    def __init__(self, **kwargs):
        if "window" in kwargs:
            self.window = kwargs.get("window")
        else:
            raise Exception("window is not found")

        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.classes = ["unknown", "hello", "manual", "title", "actor", "director", "rank", "year", "country"]
        self.tokenize = Mecab()
        self.cpu_device = "cpu"
        self.softmax = torch.nn.Softmax(dim=1)
        self.threshold = 0.7

        self.model_path = None

        self.check_get_info = False
        self.continue_search = False
        self.label_name = ""
        self.current_search_text = ""

        self.nation = None
        self.current_page = 1
        self.list_count = 3

    
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
        self.messageClearButton = QtWidgets.QPushButton(self.centralwidget)
        self.messageClearButton.setGeometry(QtCore.QRect(230, 105, 75, 23))
        self.messageClearButton.setObjectName("messageClearButton")
        self.nationSettingButton = QtWidgets.QPushButton(self.centralwidget)
        self.nationSettingButton.setGeometry(QtCore.QRect(670, 400, 101, 31))
        self.nationSettingButton.setObjectName("nationSettingButton")
        self.nationTextBox = QtWidgets.QLineEdit(self.centralwidget)
        self.nationTextBox.setGeometry(QtCore.QRect(510, 400, 151, 31))
        self.nationTextBox.setText("")
        self.nationTextBox.setMaxLength(100)
        self.nationTextBox.setObjectName("nationTextBox")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(510, 370, 171, 21))
        self.label_5.setObjectName("label_5")
        self.nextButton = QtWidgets.QPushButton(self.centralwidget)
        self.nextButton.setGeometry(QtCore.QRect(650, 480, 131, 51))
        self.nextButton.setObjectName("nextButton")
        self.cancelButton = QtWidgets.QPushButton(self.centralwidget)
        self.cancelButton.setGeometry(QtCore.QRect(510, 480, 131, 51))
        self.cancelButton.setObjectName("cancelButton")
        self.pageLabel = QtWidgets.QLabel(self.centralwidget)
        self.pageLabel.setGeometry(QtCore.QRect(650, 590, 91, 31))
        self.pageLabel.setObjectName("pageLabel")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(510, 590, 91, 31))
        self.label_6.setObjectName("label_6")
        self.listCountLabel = QtWidgets.QLabel(self.centralwidget)
        self.listCountLabel.setGeometry(QtCore.QRect(650, 670, 91, 31))
        self.listCountLabel.setObjectName("listCountLabel")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(510, 670, 121, 31))
        self.label_7.setObjectName("label_7")
        self.currentListCountTextBox = QtWidgets.QLineEdit(self.centralwidget)
        self.currentListCountTextBox.setGeometry(QtCore.QRect(510, 700, 121, 31))
        self.currentListCountTextBox.setMaxLength(10)
        self.currentListCountTextBox.setObjectName("currentListCountTextBox")
        self.currentPageTextBox = QtWidgets.QLineEdit(self.centralwidget)
        self.currentPageTextBox.setGeometry(QtCore.QRect(510, 620, 121, 31))
        self.currentPageTextBox.setMaxLength(10)
        self.currentPageTextBox.setObjectName("currentPageTextBox")
        self.currentPageSettingButton = QtWidgets.QPushButton(self.centralwidget)
        self.currentPageSettingButton.setGeometry(QtCore.QRect(650, 620, 101, 31))
        self.currentPageSettingButton.setObjectName("currentPageSettingButton")
        self.currentListCountSettingButton = QtWidgets.QPushButton(self.centralwidget)
        self.currentListCountSettingButton.setGeometry(QtCore.QRect(650, 700, 101, 31))
        self.currentListCountSettingButton.setObjectName("currentListCountSettingButton")
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
        self.messageListSaveButton.setText(_translate("MainWindow", "Save Message"))
        self.messageSaveDirectorySearchButton.setText(_translate("MainWindow", "Search"))
        self.label_4.setText(_translate("MainWindow", "Date"))
        self.dateLabel.setText(_translate("MainWindow", "None"))
        self.saveSettingsButton.setText(_translate("MainWindow", "Save Settings"))
        self.modelLoadButton.setText(_translate("MainWindow", "Load"))
        self.manualButton.setText(_translate("MainWindow", "Manual"))
        self.messageClearButton.setText(_translate("MainWindow", "Clear"))
        self.nationSettingButton.setText(_translate("MainWindow", "Apply"))
        self.label_5.setText(_translate("MainWindow", "Nation Setting (Empty is All)"))
        self.nextButton.setText(_translate("MainWindow", "Next Information"))
        self.cancelButton.setText(_translate("MainWindow", "Cancel Information"))
        self.pageLabel.setText(_translate("MainWindow", "1"))
        self.label_6.setText(_translate("MainWindow", "Current Page: "))
        self.listCountLabel.setText(_translate("MainWindow", "3"))
        self.label_7.setText(_translate("MainWindow", "Current List Count:"))
        self.currentListCountTextBox.setText(_translate("MainWindow", "3"))
        self.currentPageTextBox.setText(_translate("MainWindow", "1"))
        self.currentPageSettingButton.setText(_translate("MainWindow", "Apply"))
        self.currentListCountSettingButton.setText(_translate("MainWindow", "Apply"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))

        MainWindow.setWindowIcon(QtGui.QIcon('img/chatbot.png'))

        self.sendMessageButton.setDisabled(True)
        self.next_and_cancel_button_enable(False)

    # Function -------------------------------------------------------------------------------------------------------------
    def add_event(self):
        self.actionExit.triggered.connect(QtWidgets.qApp.quit)
        self.sendMessageButton.clicked.connect(self.send_message_button_clicked)
        self.sendMessageBox.keyPressEvent = self.send_message_keypress_event
        self.messageSaveDirectorySearchButton.clicked.connect(self.search_msg_save_folder_clicked)
        self.modelSearchButton.clicked.connect(self.search_model_folder_clicked)
        self.saveSettingsButton.clicked.connect(self.save_settings_clicked)
        self.modelLoadButton.clicked.connect(self.load_model)
        self.messageClearButton.clicked.connect(self.message_clear)
        self.manualButton.clicked.connect(self.manual_button_clicked)
        self.messageListSaveButton.clicked.connect(self.message_save_clicked)
        self.nationSettingButton.clicked.connect(self.apply_nation_button_clicked)
        self.currentPageSettingButton.clicked.connect(self.apply_current_page_button_clicked)
        self.currentListCountSettingButton.clicked.connect(self.apply_list_count_button_clicked)
        self.nextButton.clicked.connect(self.next_button_clicked)
        self.cancelButton.clicked.connect(self.cancel_button_clicked)

    # 사용자 메시지
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
        self.messageListBox.append("\n")
        self.messageListBox.moveCursor(QtGui.QTextCursor.End)

        self.sendMessageButton.setDisabled(True)
        self.sendMessageBox.setFocus()

        if self.check_get_info == False:
            self.prediction_chatbot_message(text)
        else:
            if self.label_name == "rank" and (text == "ㄴ" or text == "n"): # 순위일 때 취소
                self.reply_chatbot_message(SCRIPT_LIST["cancel"][random.randrange(0, len(SCRIPT_LIST["cancel"]))])
                self.next_and_cancel_button_enable(False)
                self.reset_current_param()
                return

            if self.continue_search == True and (text != "ㅇ" and text != "y"):
                self.reply_chatbot_message(SCRIPT_LIST["cancel"][random.randrange(0, len(SCRIPT_LIST["cancel"]))])
                self.next_and_cancel_button_enable(False)
                self.reset_current_param()
                return
            
            if self.continue_search == False:
                self.current_search_text = text
            else:
                text = self.current_search_text

            info_list = self.get_information(text)

            if info_list is not None:
                for info in info_list:
                    self.reply_chatbot_message(info)

                if self.label_name == "rank":
                    self.reset_current_param()
                    self.reply_chatbot_message(SCRIPT_LIST["other"][random.randrange(0, len(SCRIPT_LIST["other"]))])
                    return
                else:
                    self.continue_search = True
                    self.next_and_cancel_button_enable(True)
                    self.pageLabel.setText(str(self.current_page))
                
                self.current_page += 1
                self.reply_chatbot_message(SCRIPT_LIST["next"][random.randrange(0, len(SCRIPT_LIST["next"]))])

            else:
                if self.continue_search == True:
                    self.reply_chatbot_message(SCRIPT_LIST["end"][random.randrange(0, len(SCRIPT_LIST["end"]))])
                else:
                    self.reply_chatbot_message(SCRIPT_LIST["none"][random.randrange(0, len(SCRIPT_LIST["none"]))])
                self.reset_current_param()


    # 챗봇 메시지
    def reply_chatbot_message(self, text: str):
        self.messageListBox.setTextColor(QtGui.QColor(0, 0, 255))
        self.messageListBox.setAlignment(QtCore.Qt.AlignLeft)
        self.messageListBox.insertPlainText(text)
        self.messageListBox.append("\n")
        self.messageListBox.moveCursor(QtGui.QTextCursor.End)


    # 사용자 메시지 예측
    def prediction_chatbot_message(self, text):
        output = sentence_prediction(self.model, self.model.vocab, text, self.tokenize.morphs, self.device, self.cpu_device)
        output= self.softmax(output)
        output = (output > self.threshold).int().to(self.cpu_device)
        top_idx = torch.topk(output, 1)
        top_idx = top_idx.indices.numpy().reshape(-1)
        pred_str = self.classes[top_idx[0]]

        if top_idx != 0 and top_idx != 1 and top_idx != 2:
            self.check_get_info = True
            self.label_name = pred_str
        else:
            self.check_get_info = False
        
        self.reply_chatbot_message(SCRIPT_LIST[pred_str][random.randrange(0, len(SCRIPT_LIST[pred_str]))])

        if(pred_str == "rank"):
            nation_type = None
            add_nation_str = ""

            if "한국" in text:
                nation_type = "K"
                add_nation_str = "(한국 영화)"
            elif "외국" in text:
                nation_type = "F"
                add_nation_str = "(외국 영화)"
            
            current = datetime.datetime.now()

            date = (current - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            self.reply_chatbot_message(date + " 기준 " + add_nation_str +"\n")

            info_list = get_rank((current - datetime.timedelta(days=7)).strftime("%Y%m%d"), nation_type)

            if info_list is not None:
                for info in info_list:
                    self.reply_chatbot_message(info)

                self.reply_chatbot_message(SCRIPT_LIST["rank_next"][random.randrange(0, len(SCRIPT_LIST["rank_next"]))])
            else:
                self.reply_chatbot_message(SCRIPT_LIST["none"][random.randrange(0, len(SCRIPT_LIST["none"]))])

    # ----------------------------------------------------------------------------------------------------------------------
    

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
        
        show_messagebox("확인", "설정 저장 완료")


    def load_settings(self):
        if os.path.exists("Settings.dat"):
            with open("Settings.dat", "rb") as f:
                self.modelPathTextBox.setText(f.readline().decode().replace("\n", ""))
                self.modelAutoLoadCheckBox.setChecked(True if f.readline().decode() == "1\n" else False)
                self.messageSaveDirectoryTextBox.setText(f.readline().decode().replace("\n", ""))
            
            if self.modelAutoLoadCheckBox.isChecked():
                show_messagebox("확인", "model auto load")
                self.model_path = self.modelPathTextBox.text()
                self.load_model()

    
    def load_model(self):
        path = self.modelPathTextBox.text()

        try:
            if self.model is not None and self.model_path == path:
                show_messagebox("이미 model load 되었습니다")
            elif path is not None and path != "":
                # load vocab
                if self.model is not None and self.model_path != path:
                    reply = self.message_clear()
                    if reply == QtWidgets.QMessageBox.No:
                        return

                vocab, weight = load_pretrained_weights(path, self.device)

                self.model = Combination(vocab, len(vocab), class_num=len(self.classes), embed_dim=512, n_filters=128).to(self.device)
                load_checkpoint(path, self.model, self.device, optimizer=None, strict=False)
                show_messagebox("확인", "model load 완료")
                self.model_path = path
                self.reply_chatbot_message(SCRIPT_LIST["welcome"][random.randrange(0, len(SCRIPT_LIST["welcome"]))])
            else:
                show_messagebox("오류", "model 경로를 지정하세요")

        except Exception as e:
            show_messagebox("오류", e)
            QtWidgets.qApp.quit()


    def manual_button_clicked(self):
        self.reply_chatbot_message(SCRIPT_LIST["manual"][random.randrange(0, len(SCRIPT_LIST["manual"]))])


    def message_clear(self):
        reply = QtWidgets.QMessageBox.warning(None, "확인", "대화 내역이 모두 지워집니다. 계속하시겠습니까?", 
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            self.messageListBox.clear()

        return reply


    def message_save_clicked(self):
        path = self.messageSaveDirectoryTextBox.text()
        date = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd")
        filename = "/chatbot_message_list_" + date + ".txt"

        if path == "" or path is None:
            path = "."

        if os.path.exists(path + filename):
            reply = QtWidgets.QMessageBox.warning(None, "확인", "메시지 파일이 존재합니다.\n현재 메시지 내용으로 덮어씌워집니다.\n계속하시겠습니까?", 
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.Yes:
                with open(path + filename, "w", encoding="utf-8") as f:
                    f.write(self.messageListBox.toPlainText())
                show_messagebox("확인", "저장 완료")
        else:
            with open(path + filename, "w", encoding="utf-8") as f:
                f.write(self.messageListBox.toPlainText())
            show_messagebox("확인", "저장 완료")


    def get_information(self, text):
        info_list = None

        if self.label_name == "title":
            info_list = get_title(text, self.nation, self.current_page, self.list_count)
        elif self.label_name == "actor":
            info_list = get_actor(text, self.current_page, self.list_count)
        elif self.label_name == "director":
            info_list = get_director(text, self.nation, self.current_page, self.list_count)
        elif self.label_name == "rank":
            info_list = get_rank(text, None)
        elif self.label_name == "year":
            info_list = get_year(text, self.nation, self.current_page, self.list_count)
        elif self.label_name == "country":
            info_list = get_country(text, self.current_page, self.list_count)

        return info_list


    def apply_nation_button_clicked(self):
        nation = self.nationTextBox.text()

        if nation == "" or nation is None:
            self.nation = None
        else:
            self.nation = nation

    def apply_list_count_button_clicked(self):
        try:
            list_count = int(self.currentListCountTextBox.text())
            
            if list_count < 1 or list_count > 10:
                show_messagebox("범위 제한", "list 범위는 1 ~ 10 입니다")
                return
            self.list_count = list_count
            self.listCountLabel.setText(self.currentListCountTextBox.text())
        except Exception as e:
            show_messagebox("오류", "1 ~ 10 사이의 숫자만 입력하세요")
            return

    
    def apply_current_page_button_clicked(self):
        try:
            current_page = int(self.currentPageTextBox.text())
            self.current_page = current_page
            self.pageLabel.setText(self.currentPageTextBox.text())
        except Exception as e:
            show_messagebox("오류", "숫자만 입력하세요")
            return


    def reset_current_param(self):
        self.check_get_info = False
        self.continue_search = False
        self.label_name = ""
        self.current_search_text = ""
        
        self.current_page = 1
        self.pageLabel.setText(str(self.current_page))


    def next_and_cancel_button_enable(self, enable: bool):
        self.nextButton.setEnabled(enable)
        self.cancelButton.setEnabled(enable)

    
    def next_button_clicked(self):
        self.sendMessageBox.setText("ㅇ")
        self.send_message_button_clicked()

    
    def cancel_button_clicked(self):
        self.sendMessageBox.setText("N")
        self.send_message_button_clicked()