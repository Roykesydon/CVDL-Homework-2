import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from domain.question1.BackgroundSubtraction import BackgroundSubtraction


class LoadMediaController:
    image_path = ""
    video_path = ""
    
    def __init__(self, main_window):
        self._main_window = main_window

        self._main_window.load_image.clicked.connect(self.on_load_image_clicked)
        self._main_window.load_video.clicked.connect(self.on_load_video_clicked)

    def on_load_image_clicked(self):
        # let user select an image file
        # The `LoadMediaController` class is responsible for handling the loading of media files
        # (images and videos) in a graphical user interface (GUI) application. It is designed to work
        # with a main window object.
        LoadMediaController.image_path, _ = QFileDialog.getOpenFileName(self._main_window, 'Open file', '.', "Image files (*.jpg *.gif *.png *.jpeg)")
        
    def on_load_video_clicked(self):
        LoadMediaController.video_path, _ = QFileDialog.getOpenFileName(self._main_window, 'Open file', '.', "Video files (*.mp4 *.avi *.mov)")