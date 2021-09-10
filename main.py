import torch
import torchvision.transforms as transforms
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from key_point_ui import Ui_MainWindow
from torchvision import models


if __name__ == "__main__":
    # device 정보 : cuda / cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # keypoint 모델 호출
    # eval() : 테스트 모드 실행
    model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

    # 테스트 transforms
    trf = transforms.Compose([
        transforms.ToTensor()
    ])

    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(window=window, model=model, device=device, trf=trf)
    ui.setupUi(window)

    window.show()
    sys.exit(app.exec_())