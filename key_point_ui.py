# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'key_point.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from prediction import prediction
import os

class Ui_MainWindow(object):
    def __init__(self, **kwargs):
        self.img_format = ("png", "jpg", "jpeg")
        self.img_path_list = None
        self.img_index = 0

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

        if "trf" in kwargs:
            self.trf = kwargs.get("trf")
        else:
            raise Exception("transform is not found")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1344, 866)
        MainWindow.setMinimumSize(QtCore.QSize(1344, 866))
        MainWindow.setMaximumSize(QtCore.QSize(1344, 866))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.prevButton = QtWidgets.QPushButton(self.centralwidget)
        self.prevButton.setGeometry(QtCore.QRect(440, 670, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.prevButton.setFont(font)
        self.prevButton.setObjectName("prevButton")
        self.nextButton = QtWidgets.QPushButton(self.centralwidget)
        self.nextButton.setGeometry(QtCore.QRect(680, 670, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.nextButton.setFont(font)
        self.nextButton.setObjectName("nextButton")
        self.nameLabel = QtWidgets.QLabel(self.centralwidget)
        self.nameLabel.setGeometry(QtCore.QRect(20, 20, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.nameLabel.setFont(font)
        self.nameLabel.setObjectName("nameLabel")
        self.countLabel = QtWidgets.QLabel(self.centralwidget)
        self.countLabel.setGeometry(QtCore.QRect(660, 750, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.countLabel.setFont(font)
        self.countLabel.setObjectName("countLabel")
        self.annoView = QtWidgets.QLabel(self.centralwidget)
        self.annoView.setEnabled(True)
        self.annoView.setGeometry(QtCore.QRect(20, 50, 641, 581))
        self.annoView.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid black;")
        self.annoView.setFrameShadow(QtWidgets.QFrame.Plain)
        self.annoView.setLineWidth(1)
        self.annoView.setText("")
        self.annoView.setObjectName("annoView")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(260, 630, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(960, 630, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.predView = QtWidgets.QLabel(self.centralwidget)
        self.predView.setEnabled(True)
        self.predView.setGeometry(QtCore.QRect(680, 50, 641, 581))
        self.predView.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid black;")
        self.predView.setFrameShadow(QtWidgets.QFrame.Plain)
        self.predView.setLineWidth(1)
        self.predView.setText("")
        self.predView.setObjectName("predView")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1344, 21))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_Image_Folder = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.actionOpen_Image_Folder.setFont(font)
        self.actionOpen_Image_Folder.setObjectName("actionOpen_Image_Folder")
        self.actionOpen_Video_Folder = QtWidgets.QAction(MainWindow)
        self.actionOpen_Video_Folder.setObjectName("actionOpen_Video_Folder")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuMenu.addAction(self.actionOpen_Image_Folder)
        self.menuMenu.addAction(self.actionOpen_Video_Folder)
        self.menuMenu.addAction(self.actionExit)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.retranslateUi(MainWindow)
        self.center(MainWindow)
        self.add_event()

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Key Point Prediction"))
        self.prevButton.setText(_translate("MainWindow", "Previous"))
        self.nextButton.setText(_translate("MainWindow", "Next"))
        self.nameLabel.setText(_translate("MainWindow", "None"))
        self.countLabel.setText(_translate("MainWindow", "N / N"))
        self.label_2.setText(_translate("MainWindow", "Annotation"))
        self.label_3.setText(_translate("MainWindow", "Prediction"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.actionOpen_Image_Folder.setText(_translate("MainWindow", "Open Image Folder"))
        self.actionOpen_Video_Folder.setText(_translate("MainWindow", "Open Video Folder"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        MainWindow.setWindowIcon(QtGui.QIcon('img/key.png'))

    def center(self, MainWindow):
        pos = MainWindow.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        pos.moveCenter(cp)
        MainWindow.move(pos.topLeft())


    def add_event(self):
        self.actionOpen_Image_Folder.triggered.connect(self.open_img_folder_clicked)
        self.actionOpen_Video_Folder.triggered.connect(self.open_video_folder_clicked)
        self.actionExit.triggered.connect(QtWidgets.qApp.quit)
        self.nextButton.clicked.connect(self.next_button_clicked)
        self.prevButton.clicked.connect(self.prev_button_clicked)


    def next_button_clicked(self):
        if self.img_path_list is not None and self.img_index < len(self.img_path_list) - 1:
            self.img_index += 1
            self.show_image()


    def prev_button_clicked(self):
        if self.img_index > 0 and self.img_path_list is not None:
            self.img_index -= 1
            self.show_image()


    def open_img_folder_clicked(self):
        default_path = os.path.join(os.path.expanduser('~'), 'Desktop/')
        fname = QtWidgets.QFileDialog.getExistingDirectory(self.window, 'Open Image Folder', default_path)

        if fname:
            path_list = []
            for root, dirs, files in os.walk(fname):
                for file in files:
                    if file.endswith(self.img_format):
                        path_list.append(os.path.join(root, file))

            if len(path_list) != 0:
                self.img_path_list = path_list
                self.show_image()

            else:
                QtWidgets.QMessageBox.information(None, "이미지 파일 없음", "이미지 파일이 존재하지 않습니다", 
                    QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.NoButton)


    def open_video_folder_clicked(self):
        pass


    def changePixmap(self, img):
        self.pixmap = QtGui.QPixmap(img)
        self.window.repaint() # repaint() will trigger the paintEvent(self, event), this way the new pixmap will be drawn on the label


    def show_image(self):
        pixmap = QtGui.QPixmap(self.img_path_list[self.img_index])
        smaller_pixmap = pixmap.scaled(self.annoView.width(), self.annoView.height())
        # 비율 유지
        # smaller_pixmap = pixmap.scaled(self.annoView.width(), self.annoView.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.annoView.setPixmap(smaller_pixmap)

        img = prediction(self.model, self.device, self.trf, self.img_path_list[self.img_index])
        
        height, width, channel = img.shape
        bytesPerLine = channel * width

        pred_pixmap = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        smaller_pred_pixmap = pred_pixmap.scaled(self.annoView.width(), self.annoView.height())
        self.predView.setPixmap(QtGui.QPixmap(smaller_pred_pixmap))
        self.nameLabel.setText(self.img_path_list[self.img_index])
        self.countLabel.setText(f"{self.img_index + 1} / {len(self.img_path_list)}")

        self.nameLabel.adjustSize()
        self.countLabel.adjustSize()