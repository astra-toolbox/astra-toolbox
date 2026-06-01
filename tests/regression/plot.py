import pyqtgraph as pg
import numpy as np

from PyQt6.QtWidgets import QMainWindow, QWidget, QGridLayout, QLabel
from PyQt6.QtGui import QShortcut, QKeySequence

# TODO: Add key shortcut to close window
# TODO: Close window on Ctrl-C / KeyboardInterrupt

# https://stackoverflow.com/questions/67604903/add-multiple-imageview-items-to-an-qt-window-to-get-several-images-within-one-wi

class MultiSlicerWidget(QWidget):
    def __init__(self, title, images, names, linked):
        if len(images) <= 0:
            raise ValueError("No images specified")
        if len(images) != len(names):
            raise ValueError("Number of images and names does not match")
        if linked:
            for i in range(len(images)-1):
                if images[i+1].shape != images[0].shape:
                    raise ValueError("Shapes of images do not match")

        super().__init__()
        self.layout = QGridLayout(self)

        self.layout.addWidget(QLabel(title), 0, 0)

        self.image = []

        for i in range(len(images)):
            img = pg.ImageView(self, name=f"Image{i}")
            self.layout.addWidget(img, 1, i)
            self.image.append(img)

        for i, name in enumerate(names):
            self.layout.addWidget(QLabel(name), 2, i)

        if linked:
            if images[0].ndim == 3:
                for i in range(len(images)):
                    self.image[i-1].sigTimeChanged.connect(self.image[i].setCurrentIndex)

            for i in range(len(images)-1):
                self.image[i].getView().linkView(pg.ViewBox.XAxis, self.image[i+1].getView())
                self.image[i].getView().linkView(pg.ViewBox.YAxis, self.image[i+1].getView())

        self.setLayout(self.layout)


class MultiSlicerWindow(QMainWindow):
    def __init__(self, title, images, names, linked):
        super().__init__()
        self.multiSlicerWidget = MultiSlicerWidget(title, images, names, linked)
        self.setCentralWidget(self.multiSlicerWidget)
        for i, img in enumerate(images):
            self.multiSlicerWidget.image[i].setImage(img)
        if images[0].ndim == 3:
            self.multiSlicerWidget.image[0].setCurrentIndex(images[0].shape[0]//2)

app = None

def close_all():
    app.closeAllWindows()

def run(title, images, names, linked):
    global app
    app = pg.mkQApp()
    W = MultiSlicerWindow(title, images, names, linked)
    W.show()

    sh = QShortcut(QKeySequence("q"), W)
    sh.activated.connect(close_all)

    app.exec()

def compare(title, images, names):
    run(title, images, names, True)

def show(title, images, names):
    run(title, images, names, False)
