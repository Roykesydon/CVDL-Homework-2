import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from application.controller.LoadMediaController import LoadMediaController
from domain.question1.BackgroundSubtraction import BackgroundSubtraction

class Question1Controller:
    def __init__(self, main_window):
        self._main_window = main_window

        self._main_window.button1.clicked.connect(self.on_button1_clicked)

    def on_button1_clicked(self):
        if LoadMediaController.video_path == "":
            QMessageBox.warning(self._main_window, "Warning", "Please load an video first!")
            return
            
        background_subtraction = BackgroundSubtraction()
        background_subtraction.process(LoadMediaController.video_path)

    