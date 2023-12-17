import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from application.controller.Question1Controller import Question1Controller
from application.controller.Question2Controller import Question2Controller
from application.controller.Question3Controller import Question3Controller
from application.controller.Question4Controller import Question4Controller
from application.controller.Question5Controller import Question5Controller
from application.controller.LoadMediaController import LoadMediaController
from application.xml import main_ui


class myMainWindow(QMainWindow, main_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self._question1_controller = Question1Controller(self)
        self._question2_controller = Question2Controller(self)
        self._question3_controller = Question3Controller(self)
        self._question4_controller = Question4Controller(self)
        self._question5_controller = Question5Controller(self)
        self._load_media_controller = LoadMediaController(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())