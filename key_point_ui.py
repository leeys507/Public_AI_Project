# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'key_point.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import copy

from PyQt5 import QtCore, QtGui, QtWidgets
from prediction import img_prediction, video_prediction
import os
from annotation import *
from video_control import VideoFile

class Ui_MainWindow(object):
    def __init__(self, **kwargs):
        self.img_format = ("png", "jpg", "jpeg")
        self.video_format = ("mp4", "wmv", "avi")
        self.mode = "img"
        self.current_video_file = None

        self.img_path_list = None
        self.img_anno_list = []
        self.img_pred_list = []
        self.img_index = 0
        self.img_batch_size = 10

        self.video_path_list = None

        self.video_prev_anno_list = [] # buffer
        self.video_prev_pred_list = []

        self.video_anno_list = []
        self.video_pred_list = []
        self.video_file_index = 0
        self.video_index = 0
        self.video_start_frame_index = 0
        self.video_buffer_size = 10
        self.current_total_frame = 1

        self.new_prediction = True

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
        self.countLabel.setGeometry(QtCore.QRect(650, 750, 91, 21))
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
        self.label_2.setGeometry(QtCore.QRect(290, 630, 81, 21))
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
        self.videoBufferSizeTextEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.videoBufferSizeTextEdit.setGeometry(QtCore.QRect(150, 670, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.videoBufferSizeTextEdit.setFont(font)
        self.videoBufferSizeTextEdit.setObjectName("videoBufferSizeTextEdit")
        self.nextVideoButton = QtWidgets.QPushButton(self.centralwidget)
        self.nextVideoButton.setGeometry(QtCore.QRect(260, 770, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.nextVideoButton.setFont(font)
        self.nextVideoButton.setObjectName("nextVideoButton")
        self.videoCountLabel = QtWidgets.QLabel(self.centralwidget)
        self.videoCountLabel.setGeometry(QtCore.QRect(650, 790, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.videoCountLabel.setFont(font)
        self.videoCountLabel.setObjectName("videoCountLabel")
        self.videoBufferSizeLabel = QtWidgets.QLabel(self.centralwidget)
        self.videoBufferSizeLabel.setGeometry(QtCore.QRect(20, 680, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.videoBufferSizeLabel.setFont(font)
        self.videoBufferSizeLabel.setObjectName("videoBufferSizeLabel")
        self.nextVideoFrameButton = QtWidgets.QPushButton(self.centralwidget)
        self.nextVideoFrameButton.setGeometry(QtCore.QRect(1060, 670, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.nextVideoFrameButton.setFont(font)
        self.nextVideoFrameButton.setObjectName("nextVideoFrameButton")
        self.prevVideoFrameButton = QtWidgets.QPushButton(self.centralwidget)
        self.prevVideoFrameButton.setGeometry(QtCore.QRect(1060, 720, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.prevVideoFrameButton.setFont(font)
        self.prevVideoFrameButton.setObjectName("prevVideoFrameButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 700, 211, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.previousVideoButton = QtWidgets.QPushButton(self.centralwidget)
        self.previousVideoButton.setGeometry(QtCore.QRect(20, 770, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.previousVideoButton.setFont(font)
        self.previousVideoButton.setObjectName("previousVideoButton")
        self.videoFileCountLabel = QtWidgets.QLabel(self.centralwidget)
        self.videoFileCountLabel.setGeometry(QtCore.QRect(230, 730, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.videoFileCountLabel.setFont(font)
        self.videoFileCountLabel.setObjectName("videoFileCountLabel")
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
        font.setPointSize(10)
        self.actionOpen_Image_Folder.setFont(font)
        self.actionOpen_Image_Folder.setObjectName("actionOpen_Image_Folder")
        self.actionOpen_Video_Folder = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.actionOpen_Video_Folder.setFont(font)
        self.actionOpen_Video_Folder.setObjectName("actionOpen_Video_Folder")
        self.actionExit = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.actionExit.setFont(font)
        self.actionExit.setObjectName("actionExit")
        self.menuMenu.addAction(self.actionOpen_Image_Folder)
        self.menuMenu.addAction(self.actionOpen_Video_Folder)
        self.menuMenu.addAction(self.actionExit)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.retranslateUi(MainWindow)
        self.center(MainWindow)
        self.add_event()
        self.hide_contents()

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
        self.videoBufferSizeTextEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">10</p></body></html>"))
        self.nextVideoButton.setText(_translate("MainWindow", "Next Video"))
        self.videoCountLabel.setText(_translate("MainWindow", "N / N"))
        self.videoBufferSizeLabel.setText(_translate("MainWindow", "Video Buffer Size"))
        self.nextVideoFrameButton.setText(_translate("MainWindow", "Next Video Frame"))
        self.prevVideoFrameButton.setText(_translate("MainWindow", "Previous Video Frame"))
        self.label.setText(_translate("MainWindow", "Applies only when opening a video"))
        self.previousVideoButton.setText(_translate("MainWindow", "Previous Video"))
        self.videoFileCountLabel.setText(_translate("MainWindow", "N / N"))
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
        self.previousVideoButton.clicked.connect(self.previous_video_button_clicked)
        self.nextVideoButton.clicked.connect(self.next_video_button_clicked)
        self.prevVideoFrameButton.clicked.connect(self.previous_frame_button_clicked)
        self.nextVideoFrameButton.clicked.connect(self.next_frame_button_clicked)


    def next_button_clicked(self):
        if self.mode == "img":
            if self.img_index != 0 and self.img_index % self.img_batch_size == 0:
                anno_pixmap_list, anno_img_info = self.create_anno_pixmap(self.img_path_list[self.img_index:self.img_index + self.img_batch_size])
                pred_pixmap_list = self.create_pred_pixmap(self.img_path_list[self.img_index:self.img_index + self.img_batch_size], anno_img_info)
                self.img_anno_list.extend(anno_pixmap_list)
                self.img_pred_list.extend(pred_pixmap_list)
                

            if self.img_anno_list is not None and self.img_index < len(self.img_anno_list) - 1:
                self.img_index += 1
                self.show_image(self.img_anno_list[self.img_index], self.img_pred_list[self.img_index])
        else: # video
            if self.video_anno_list is not None and self.video_index < len(self.video_anno_list) - 1:
                self.video_index += 1
                self.show_image(self.video_anno_list[self.video_index], self.video_pred_list[self.video_index], self.current_video_file.frames)


    def prev_button_clicked(self):
        if self.mode == "img":
            if self.img_index > 0 and self.img_anno_list is not None:
                self.img_index -= 1
                self.show_image(self.img_anno_list[self.img_index], self.img_pred_list[self.img_index])
        else:
            if self.video_index > 0 and self.video_anno_list is not None:
                self.video_index -= 1
                self.show_image(self.video_anno_list[self.video_index], self.video_pred_list[self.video_index], self.current_total_frame)


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
                self.annoView.clear()
                self.predView.clear()

                if self.img_path_list is not None:
                    self.img_resource_clear()

                self.img_path_list = path_list

                if (len(path_list) > self.img_batch_size):
                    anno_pixmap_list, anno_img_info = self.create_anno_pixmap(self.img_path_list[:self.img_batch_size])
                    pred_pixmap_list = self.create_pred_pixmap(self.img_path_list[:self.img_batch_size], anno_img_info)
                else:
                    anno_pixmap_list, anno_img_info = self.create_anno_pixmap(self.img_path_list)
                    pred_pixmap_list = self.create_pred_pixmap(self.img_path_list, anno_img_info)

                self.img_anno_list.extend(anno_pixmap_list)
                self.img_pred_list.extend(pred_pixmap_list)

                self.show_image(self.img_anno_list[self.img_index], self.img_pred_list[self.img_index])
                self.hide_contents()

            else:
                show_messagebox("이미지 파일 없음", "이미지 파일이 존재하지 않습니다")

            self.mode = "img"


    def open_video_folder_clicked(self):
        try:
            buffer_size = int(self.videoBufferSizeTextEdit.toPlainText())
            if buffer_size < 1 or buffer_size > 50:
                show_messagebox("범위 제한", "buffer 범위는 1 ~ 50 입니다")
                return
            self.video_buffer_size = buffer_size
        except Exception as e:
            show_messagebox("오류", "1 ~ 50 사이의 숫자만 입력하세요")
            return

        default_path = os.path.join(os.path.expanduser('~'), 'Desktop/')
        fname = QtWidgets.QFileDialog.getExistingDirectory(self.window, 'Open Video Folder', default_path)

        if fname:
            path_list = []

            for root, dirs, files in os.walk(fname):
                for file in files:
                    if file.endswith(self.video_format):
                        path_list.append(os.path.join(root, file))

            if len(path_list) != 0:
                self.annoView.clear()
                self.predView.clear()

                if self.video_path_list is not None:
                    self.video_resource_clear()

                self.mode = "video"
                self.video_path_list = path_list
                self.current_video_file = VideoFile(self.video_path_list[self.video_file_index], self.video_buffer_size)
                anno_path = os.path.join(os.path.dirname(self.video_path_list[self.video_file_index]), "annotations.json")
                with open(anno_path) as f:
                    anno_json = json.load(f)

                self.current_total_frame = anno_json['total_frame']

                anno_pixmap_list, anno_img_info = self.create_anno_video_pixmap(anno_path)
                pred_pixmap_list = self.create_pred_video_pixmap(anno_img_info)
                
                self.video_anno_list = anno_pixmap_list
                self.video_pred_list = pred_pixmap_list

                self.show_image(self.video_anno_list[self.video_index], self.video_pred_list[self.video_index],
                                self.current_total_frame)
                self.show_contents()

                self.countLabel.setText(f"1 / {len(self.video_anno_list)}")
                self.videoCountLabel.setText(f"1 / {self.current_total_frame}")
                self.videoFileCountLabel.setText(f"1 / {len(self.video_path_list)}")

                self.countLabel.adjustSize()
                self.videoCountLabel.adjustSize()
                self.videoFileCountLabel.adjustSize()
                self.prevVideoFrameButton.setDisabled(True)

            else:
                show_messagebox("영상 파일 없음", "영상 파일이 존재하지 않습니다")


    def changePixmap(self, img):
        self.pixmap = QtGui.QPixmap(img)
        self.window.repaint() # repaint() will trigger the paintEvent(self, event), this way the new pixmap will be drawn on the label


    def show_image(self, anno_pixmap_img, pred_pixmap_img, total_frame=1):
        self.annoView.clear()
        self.predView.clear()
        
        if self.mode == "img":
            self.annoView.setPixmap(anno_pixmap_img)
            self.predView.setPixmap(pred_pixmap_img)

            self.nameLabel.setText(self.img_path_list[self.img_index])
            self.countLabel.setText(f"{self.img_index + 1} / {len(self.img_path_list)}")

        else:
            self.annoView.setPixmap(anno_pixmap_img)
            self.predView.setPixmap(pred_pixmap_img)

            self.nameLabel.setText(self.video_path_list[self.video_file_index])
            self.countLabel.setText(f"{self.video_index + 1} / {len(self.video_anno_list)}")
            self.videoCountLabel.setText(f"{self.video_index + self.video_start_frame_index + 1} / {total_frame}")

        self.nameLabel.adjustSize()
        self.countLabel.adjustSize()
        self.videoCountLabel.adjustSize()


    def create_anno_pixmap(self, anno_img_path_list):
        anno_pixmap_list = []
        anno_img_info = []

        for anno_img_path in anno_img_path_list:
            annotation_image, frame, keypoints = anno_image(anno_img_path)
            height, width, channel = annotation_image.shape
            bytesPerLine = channel * width

            anno_qimg = QtGui.QImage(annotation_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap(anno_qimg)
            smaller_anno_pixmap = pixmap.scaled(self.annoView.width(), self.annoView.height())
            # 비율 유지
            # smaller_pixmap = pixmap.scaled(self.annoView.width(), self.annoView.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            anno_pixmap_list.append(smaller_anno_pixmap)

            info_dict = {
                "frame": frame,
                "keypoints": keypoints,
            }
            anno_img_info.append(info_dict)

        return anno_pixmap_list, anno_img_info

    
    def create_anno_video_pixmap(self, anno_video_path):
        anno_pixmap_list = []
        anno_img_info = []

        annotation_image_list, frame_list, keypoints_list = anno_video(anno_video_path, self.current_video_file, self.video_start_frame_index, self.video_buffer_size)

        for annotation_image, frame, keypoints in zip(annotation_image_list, frame_list, keypoints_list):
            height, width, channel = annotation_image.shape
            bytesPerLine = channel * width

            anno_qimg = QtGui.QImage(annotation_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap(anno_qimg)
            smaller_anno_pixmap = pixmap.scaled(self.annoView.width(), self.annoView.height())
            # 비율 유지
            # smaller_pixmap = pixmap.scaled(self.annoView.width(), self.annoView.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            anno_pixmap_list.append(smaller_anno_pixmap)

            info_dict = {
                "frame": frame,
                "keypoints": keypoints,
            }
            anno_img_info.append(info_dict)

        return anno_pixmap_list, anno_img_info


    def create_pred_pixmap(self, pred_img_path_list, anno_img_info):
        pred_pixmap_list = []
        pred_img_list = img_prediction(self.model, self.device, self.trf, pred_img_path_list, anno_img_info)

        for pred_img in pred_img_list:
            height, width, channel = pred_img.shape
            bytesPerLine = channel * width

            pred_qimg = QtGui.QImage(pred_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pred_pixmap = QtGui.QPixmap(pred_qimg)
            smaller_pred_pixmap = pred_pixmap.scaled(self.annoView.width(), self.annoView.height())
            pred_pixmap_list.append(smaller_pred_pixmap)

        return pred_pixmap_list


    def create_pred_video_pixmap(self, anno_img_info):
        pred_pixmap_list = []
        frame_list = [x["frame"] for x in anno_img_info]
        pred_img_list = video_prediction(self.model, self.device, self.trf, self.current_video_file.file_path,
                frame_list, self.current_video_file.current_frame[:self.current_video_file.current_frame_length], anno_img_info)

        for pred_img in pred_img_list:
            height, width, channel = pred_img.shape
            bytesPerLine = channel * width

            pred_qimg = QtGui.QImage(pred_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            pred_pixmap = QtGui.QPixmap(pred_qimg)
            smaller_pred_pixmap = pred_pixmap.scaled(self.annoView.width(), self.annoView.height())
            pred_pixmap_list.append(smaller_pred_pixmap)

        return pred_pixmap_list

    
    def hide_contents(self):
        self.videoBufferSizeLabel.hide()
        self.videoBufferSizeTextEdit.hide()
        self.label.hide()
        self.previousVideoButton.hide()
        self.nextVideoButton.hide()
        self.videoCountLabel.hide()
        self.nextVideoFrameButton.hide()
        self.prevVideoFrameButton.hide()
        self.videoFileCountLabel.hide()


    def show_contents(self):
        self.videoBufferSizeLabel.show()
        self.videoBufferSizeTextEdit.show()
        self.label.show()
        self.previousVideoButton.show()
        self.nextVideoButton.show()
        self.videoCountLabel.show()
        self.nextVideoFrameButton.show()
        self.prevVideoFrameButton.show()
        self.videoFileCountLabel.show()


    def img_resource_clear(self):
        self.img_path_list.clear()
        self.img_anno_list.clear()
        self.img_pred_list.clear()
        self.img_index = 0
        self.nameLabel.setText("None")
        self.nameLabel.adjustSize()
        self.new_prediction = True


    def video_resource_clear(self):
        self.video_path_list.clear()
        self.video_anno_list.clear()
        self.video_pred_list.clear()
        self.video_prev_anno_list.clear()
        self.video_prev_pred_list.clear()
        self.video_index = 0
        self.video_start_frame_index = 0
        self.nameLabel.setText("None")
        self.nameLabel.adjustSize()
        self.new_prediction = True


    def previous_video_button_clicked(self):
        if self.video_file_index == 0:
            return

        self.new_prediction = True

        self.video_index = 0
        self.video_start_frame_index = 0
        self.video_file_index -= 1
        self.current_video_file = VideoFile(self.video_path_list[self.video_file_index], self.video_buffer_size)
        anno_path = os.path.join(os.path.dirname(self.video_path_list[self.video_file_index]), "annotations.json")
        with open(anno_path) as f:
            anno_json = json.load(f)
        self.current_total_frame = anno_json['total_frame']
        anno_pixmap_list, anno_img_info = self.create_anno_video_pixmap(anno_path)
        pred_pixmap_list = self.create_pred_video_pixmap(anno_img_info)

        self.video_anno_list = anno_pixmap_list
        self.video_pred_list = pred_pixmap_list

        self.show_image(self.video_anno_list[self.video_index], self.video_pred_list[self.video_index],
                        self.current_total_frame)
        self.show_contents()

        self.countLabel.setText(f"1 / {len(self.video_anno_list)}")
        self.videoCountLabel.setText(f"1 / {self.current_total_frame}")
        self.videoFileCountLabel.setText(f"{self.video_file_index + 1} / {len(self.video_path_list)}")
        
        self.videoCountLabel.adjustSize()
        self.videoFileCountLabel.adjustSize()
        self.countLabel.adjustSize()
        return


    def next_video_button_clicked(self):
        if self.video_file_index == len(self.video_path_list)-1:
            return
        
        self.new_prediction = True

        self.video_index = 0
        self.video_start_frame_index = 0
        self.video_file_index += 1
        self.current_video_file = VideoFile(self.video_path_list[self.video_file_index], self.video_buffer_size)
        anno_path = os.path.join(os.path.dirname(self.video_path_list[self.video_file_index]), "annotations.json")
        with open(anno_path) as f:
            anno_json = json.load(f)
        self.current_total_frame = anno_json['total_frame']

        anno_pixmap_list, anno_img_info = self.create_anno_video_pixmap(anno_path)
        pred_pixmap_list = self.create_pred_video_pixmap(anno_img_info)

        self.video_anno_list = anno_pixmap_list
        self.video_pred_list = pred_pixmap_list

        self.show_image(self.video_anno_list[self.video_index], self.video_pred_list[self.video_index],
                        self.current_total_frame)
        # self.show_contents()

        self.countLabel.setText(f"1 / {len(self.video_anno_list)}")
        self.videoCountLabel.setText(f"1 / {self.current_total_frame}")
        self.videoFileCountLabel.setText(f"{self.video_file_index + 1} / {len(self.video_path_list)}")
        
        self.countLabel.adjustSize()
        self.videoCountLabel.adjustSize()
        self.videoFileCountLabel.adjustSize()
        return


    def previous_frame_button_clicked(self):
        if len(self.video_prev_anno_list) == 0 or \
                self.video_start_frame_index - self.video_buffer_size < 0:
            show_messagebox("확인", "No Frames In Buffer")
            return

        self.video_start_frame_index -= self.video_buffer_size
        self.video_index = 0

        if self.new_prediction == False:
            self.video_prev_anno_list = self.video_anno_list.copy()
            self.video_prev_pred_list = self.video_pred_list.copy()

            self.current_video_file.readPrevFrame(False)
            anno_path = os.path.join(os.path.dirname(self.video_path_list[self.video_file_index]), "annotations.json")

            anno_pixmap_list, anno_img_info = self.create_anno_video_pixmap(anno_path)
            pred_pixmap_list = self.create_pred_video_pixmap(anno_img_info)

            self.video_anno_list = anno_pixmap_list
            self.video_pred_list = pred_pixmap_list

        else:
            temp_anno_list = self.video_anno_list.copy()
            temp_pred_list = self.video_pred_list.copy()

            self.video_anno_list = self.video_prev_anno_list
            self.video_pred_list = self.video_prev_anno_list

            self.video_prev_anno_list = temp_anno_list
            self.video_prev_pred_list = temp_pred_list

            self.new_prediction = False
            self.current_video_file.readPrevFrame()

        self.show_image(self.video_anno_list[self.video_index],
                        self.video_pred_list[self.video_index],
                        self.current_total_frame)

        self.countLabel.setText(f"{self.video_index + 1} / {len(self.video_anno_list)}")
        self.videoCountLabel.setText(
            f"{self.video_start_frame_index + 1} / {self.current_total_frame}")
        self.countLabel.adjustSize()
        #self.prevVideoFrameButton.setDisabled(True)
        return


    def next_frame_button_clicked(self):
        if self.video_start_frame_index + self.video_buffer_size > self.current_video_file.frames-1:
            show_messagebox("확인", "No More Next Frames")
            return

        self.video_start_frame_index += self.video_buffer_size
        self.video_index = 0

        if self.new_prediction:
            self.video_prev_anno_list = self.video_anno_list.copy()
            self.video_prev_pred_list = self.video_pred_list.copy()

            self.current_video_file.readFrame(False)
            anno_path = os.path.join(os.path.dirname(self.video_path_list[self.video_file_index]), "annotations.json")

            anno_pixmap_list, anno_img_info = self.create_anno_video_pixmap(anno_path)
            pred_pixmap_list = self.create_pred_video_pixmap(anno_img_info)

            self.video_anno_list = anno_pixmap_list
            self.video_pred_list = pred_pixmap_list
        else:
            temp_anno_list = self.video_anno_list.copy()
            temp_pred_list = self.video_pred_list.copy()

            self.video_anno_list = self.video_prev_anno_list
            self.video_pred_list = self.video_prev_anno_list

            self.video_prev_anno_list = temp_anno_list
            self.video_prev_pred_list = temp_pred_list

            self.current_video_file.readFrame()

        self.show_image(self.video_anno_list[self.video_index],
                        self.video_pred_list[self.video_index],
                        self.current_total_frame)

        self.countLabel.setText(f"{self.video_index + 1} / {len(self.video_anno_list)}")
        self.videoCountLabel.setText(f"{self.video_index + self.video_start_frame_index + 1} / {self.current_video_file.frames}")
        self.countLabel.adjustSize()
        self.prevVideoFrameButton.setEnabled(True)
        self.new_prediction = True
        return


def show_messagebox(title, text):
    QtWidgets.QMessageBox.information(None, title, text, 
        QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.NoButton)