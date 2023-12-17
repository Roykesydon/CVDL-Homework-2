import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from application.controller.LoadMediaController import LoadMediaController
from domain.question4.VGG19 import VGG19
from application.xml.DrawingPad import DrawingPad
from domain.utils.DatasetLoader import DatasetLoader
from torchvision import transforms

import torch
from PIL import Image

class Question4Controller:
    def __init__(self, main_window):
        self._main_window = main_window

        self._main_window.button4_1.clicked.connect(self.on_button4_1_clicked)
        self._main_window.button4_2.clicked.connect(self.on_button4_2_clicked)
        self._main_window.button4_3.clicked.connect(self.on_button4_3_clicked)
        self._main_window.button4_4.clicked.connect(self.on_button4_4_clicked)
        self._dataset_loader = DatasetLoader()
        
        self._vgg19 = VGG19()

    def on_button4_1_clicked(self):
        self._vgg19.print_model_summary()
        
    def on_button4_2_clicked(self):
        cv2.imshow("Result", cv2.imread("./source/question4-result.png"))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def on_button4_3_clicked(self):
        self._vgg19.load_model("./weights/vgg19_bn_993.pth")
        img = DrawingPad.get_global_drawing_pad_as_grayscale()
        # predict it and show text on label4
        model = self._vgg19.get_model()
        device = self._vgg19.get_device()
        model.eval()
        
        preprocessing = self._dataset_loader.get_preprocessing()
        classes = self._dataset_loader.get_classes()
        
        img = Image.fromarray(img)
        img = preprocessing(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        self._main_window.label4.setText("Prediction: " + str(predicted.item()))
        self._vgg19.show_inference_probability(outputs, classes)
        
    def on_button4_4_clicked(self):
        if DrawingPad.global_drawing_pad is None:
            return
        
        DrawingPad.global_drawing_pad.reset_canvas()
    
    

    