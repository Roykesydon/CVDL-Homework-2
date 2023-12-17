from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

class DrawingPad(QtWidgets.QWidget):
    global_drawing_pad = None
    
    def __init__(self, parent=None):
        super(DrawingPad, self).__init__(parent)
        self.drawing = False
        self.lastPoint = QtCore.QPoint()
        self.shape = 200, 200
        self.offset = 650, 55
        self.image = QtGui.QPixmap(*self.shape)
        self.image.fill(QtCore.Qt.black)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(self.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton and self.drawing:
            painter = QtGui.QPainter(self.image)
            painter.setPen(QtGui.QPen(QtCore.Qt.white, 3, QtCore.Qt.SolidLine))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = False
            
    def get_drawing_as_grayscale(self):
        qimg = self.image.toImage()
        qimg = qimg.convertToFormat(QtGui.QImage.Format_Grayscale8)
        byte_arr = qimg.bits().asstring(qimg.byteCount())
        gray_image = np.frombuffer(byte_arr, dtype=np.uint8).reshape((qimg.height(), qimg.width()))
        return gray_image

    def reset_canvas(self):
        self.image.fill(QtCore.Qt.black)
        self.update()
        
    
    def set_global_drawing_pad(drawing_pad):
        DrawingPad.global_drawing_pad = drawing_pad
    
    def get_global_drawing_pad_as_grayscale():
        return DrawingPad.global_drawing_pad.get_drawing_as_grayscale()
"""
self.drawingPad = DrawingPad(self.centralwidget)
DrawingPad.set_global_drawing_pad(self.drawingPad)
self.drawingPad.setGeometry(QtCore.QRect(self.drawingPad.offset[0], self.drawingPad.offset[1], self.drawingPad.shape[0], self.drawingPad.shape[1]))
self.drawingPad.setObjectName("drawingPad")
"""