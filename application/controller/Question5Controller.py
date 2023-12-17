import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from application.controller.LoadMediaController import LoadMediaController
from domain.question5.ResNet50 import ResNet50
from application.xml.DrawingPad import DrawingPad
from domain.utils.DatasetLoader import DatasetLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import random

import torch
from PIL import Image
from matplotlib import pyplot as plt
import io


class Question5Controller:
    image_path = ""

    def __init__(self, main_window):
        self._main_window = main_window

        self._main_window.button5_0.clicked.connect(self.on_button5_0_clicked)
        self._main_window.button5_1.clicked.connect(self.on_button5_1_clicked)
        self._main_window.button5_2.clicked.connect(self.on_button5_2_clicked)
        self._main_window.button5_3.clicked.connect(self.on_button5_3_clicked)
        self._main_window.button5_4.clicked.connect(self.on_button5_4_clicked)
        self._dataset_loader = DatasetLoader()
        self._image_path = ""
        self._resnet50 = ResNet50()

    def on_button5_0_clicked(self):
        Question5Controller.image_path, _ = QFileDialog.getOpenFileName(
            self._main_window,
            "Open file",
            ".",
            "Image files (*.jpg *.gif *.png *.jpeg)",
        )
        
        preprocess = transforms.Compose(
            [
                # resize to 224x224
                transforms.Resize(224),
                transforms.CenterCrop(224),
            ]
        )
        
        image = Image.open(Question5Controller.image_path)
        processed_image = preprocess(image)

        buffer = io.BytesIO()
        processed_image.save(buffer, format="JPEG")
        qimage = QImage()
        qimage.loadFromData(buffer.getvalue())

        pixmap = QPixmap.fromImage(qimage)
        label5_1 = self._main_window.label5_1
        label5_1.setPixmap(pixmap)
        label5_1.setScaledContents(True) 

    def on_button5_1_clicked(self):
        # Load the inference dataset (class 0 and 1)
        self.dataset = ImageFolder(
            root="./inference_dataset", transform=transforms.ToTensor()
        )
        preprocess = transforms.Compose(
            [
                # resize to 224x224
                transforms.Resize(224),
                transforms.CenterCrop(224),
            ]
        )

        class_to_idx = self.dataset.class_to_idx
        selected_images = []

        # 對於每個類別，選擇一張圖片
        for class_name, _ in class_to_idx.items():
            indices = [i for i, (_, label) in enumerate(self.dataset.samples) if label == class_to_idx[class_name]]

            selected_index = random.choice(indices)

            image, _ = self.dataset[selected_index]
            selected_images.append((preprocess(image), class_name))
        
        # plot one image for two classes
        fig = plt.figure(figsize=(7, 7))
        for i, (image, label) in enumerate(selected_images):
            ax = plt.subplot(1, 2, i + 1)
            plt.tight_layout()
            ax.set_title(label)
            ax.axis("off")
            plt.imshow(image.permute(1, 2, 0))
        plt.savefig("./tmp/inference_dataset.png")
        plt.show()
        

    def on_button5_2_clicked(self):
        self._resnet50.print_model_summary()

    def on_button5_3_clicked(self):
        # show "./source/question5-comparison.png"
        image = cv2.imread("./source/question5-comparison.png")
        cv2.imshow("Question5", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_button5_4_clicked(self):
        if Question5Controller.image_path == "":
            QMessageBox.warning(self._main_window, "Warning", "Please load an image first!")
            return
        
        self._resnet50.load_model("./weights/resnet50_with_erase.pth")
        class_name = self._resnet50.inference(Question5Controller.image_path)

        # set label5_2 text
        self._main_window.label5_2.setText(f"Prediction: {class_name}")
