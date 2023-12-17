import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from application.controller.LoadMediaController import LoadMediaController
from domain.question2.OpticalFlow import OpticalFlow

class Question2Controller:
    def __init__(self, main_window):
        self._main_window = main_window

        self._main_window.button2_1.clicked.connect(self.on_button2_1_clicked)
        self._main_window.button2_2.clicked.connect(self.on_button2_2_clicked)
        
        self._optical_flow = OpticalFlow()

    def on_button2_1_clicked(self):
        if LoadMediaController.video_path == "":
            QMessageBox.warning(self._main_window, "Warning", "Please load an video first!")
            return
        
        self._optical_flow.preprocessing(LoadMediaController.video_path)
    
    def on_button2_2_clicked(self):
        if LoadMediaController.video_path == "":
            QMessageBox.warning(self._main_window, "Warning", "Please load an video first!")
            return
        
        self._optical_flow.video_tracking(LoadMediaController.video_path)

    