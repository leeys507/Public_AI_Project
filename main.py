import torch
import torchvision.transforms as transforms
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from chatbot_ui import Ui_MainWindow


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(window=window)
    ui.setupUi(window)

    window.show()
    sys.exit(app.exec_())