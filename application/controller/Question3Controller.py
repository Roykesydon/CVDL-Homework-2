import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from application.controller.LoadMediaController import LoadMediaController
from domain.question3.PCA import PCA

class Question3Controller:
    def __init__(self, main_window):
        self._main_window = main_window

        self._main_window.button3.clicked.connect(self.on_button3_clicked)
        
        self._pca = PCA()

    def on_button3_clicked(self):
        if LoadMediaController.image_path == "":
            QMessageBox.warning(self._main_window, "Warning", "Please load an image first!")
            return
        self._pca.process(LoadMediaController.image_path)
    

    