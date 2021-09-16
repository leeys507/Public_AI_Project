import torch
import torchvision.transforms as transforms
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from chatbot_ui import Ui_MainWindow
from torchvision import models


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None

    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(window=window, model=model, device=device)
    ui.setupUi(window)

    window.show()
    sys.exit(app.exec_())