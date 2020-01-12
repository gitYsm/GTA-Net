import sys
import os
import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2

class MyApp(QWidget):

    def __init__(self):
        super().__init__()

        self.i = 0

        self.initUI()

    def initUI(self):
        self.statusbar = QStatusBar()

        self.labelImage = QLabel(self)
        self.labelImage.resize(600, 400)
        vbox = QVBoxLayout()
        grid = QGridLayout()


        self.btnBrowse = QPushButton("Browse")
        self.btnBrowse.clicked.connect(self.getDir)
        self.btnHRconfirm = QPushButton("HR Confirm")
        self.btnHRconfirm.clicked.connect(self.saveNext)

        self.btnLRconfirm = QPushButton('LR Confirm')
        self.btnLRconfirm.clicked.connect(self.LR)

        self.btnQuit = QPushButton("Quit")
        self.btnQuit.clicked.connect(QCoreApplication.instance().quit)

        self.btnHRleft = QPushButton("HR_Left")
        self.btnHRleft.clicked.connect(self.HR_leftMove)
        self.btnHRright = QPushButton("HR_Right")
        self.btnHRright.clicked.connect(self.HR_rightMove)

        self.btnLRleft = QPushButton("LR_Left")
        self.btnLRleft.clicked.connect(self.LR_leftMove)
        self.btnLRright = QPushButton("LR_Right")
        self.btnLRright.clicked.connect(self.LR_rightMove)

        self.btnLRup = QPushButton("LR_up")
        self.btnLRup.clicked.connect(self.LR_upMove)
        self.btnHRup = QPushButton("HR_up")
        self.btnHRup.clicked.connect(self.HR_upMove)

        self.btnDelete = QPushButton("Delete")
        self.btnDelete.clicked.connect(self.deleteImage)


        self.labelImage = QLabel(self)



        vbox.addWidget(self.btnBrowse)

        vbox.addWidget(self.labelImage)

        vbox.addWidget(self.btnHRconfirm)
        vbox.addWidget(self.btnLRconfirm)

        grid.addWidget(self.btnHRleft, 0, 0)
        grid.addWidget(self.btnHRright, 0, 1)

        grid.addWidget(self.btnLRleft, 1, 0)
        grid.addWidget(self.btnLRright, 1, 1)

        grid.addWidget(self.btnHRup, 2, 0)
        grid.addWidget(self.btnLRup, 2, 1)

        vbox.addLayout(grid)


        vbox.addWidget(self.btnDelete)

        vbox.addWidget(self.btnQuit)


        self.setLayout(vbox)
        self.setLayout(grid)




        self.show()

    def getDir(self):
        self.path = QFileDialog.getExistingDirectory(self, 'open folder', './')
        if not os.path.exists('./HR_confirmed'):
            os.mkdir('./HR_confirmed')

        if not os.path.exists('./originalTemp'):
            os.mkdir('./originalTemp')

        if not os.path.exists('./LR_confirmed'):
            os.mkdir('./LR_confirmed')

        if not os.path.exists("./LR_leftFolder"):
            os.mkdir("./LR_leftFolder")

        if not os.path.exists("./LR_rightFolder"):
            os.mkdir("./LR_rightFolder")

        if not os.path.exists("./HR_leftFolder"):
            os.mkdir("./HR_leftFolder")

        if not os.path.exists("./HR_rightFolder"):
            os.mkdir("./HR_rightFolder")

        if not os.path.exists("./HR_up"):
            os.mkdir("./HR_up")

        if not os.path.exists("./LR_up"):
            os.mkdir("./LR_up")

        self.file_list = os.listdir(self.path)
        self.file_list.sort()

        self.filename = self.file_list[self.i]
        print(self.filename)

        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        self.getImage()

    def getImage(self):

        self.pixmap = QPixmap("./originalTemp" + '/' + self.filename)
        self.pixmap = self.pixmap.scaled(16, 16)
        self.labelImage.setPixmap(self.pixmap)

    def deleteImage(self):

        if not os.path.exists("./delFolder"):
            os.mkdir("./delFolder")

        shutil.move("./originalTemp" + '/' + self.filename, "./delFolder/" + self.filename)

        print(self.filename , "delateImage")

        self.i = self.i + 1
        self.filename = self.file_list[self.i]

        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        self.getImage()


    def saveNext(self):

        shutil.copyfile("./originalTemp" + '/' + self.filename, './HR_confirmed/' + self.filename)
        print("saveNext first check", self.filename)



        self.i = self.i + 1
        self.filename = self.file_list[self.i]


        print("HR_confirmed = ", self.filename)


        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        print("HR_confirm")
        self.getImage()

    def LR(self):

        shutil.copyfile("./originalTemp" + '/' + self.filename,
                        './LR_confirmed/' + self.filename)
        print("saveNext first check", self.filename)


        self.i = self.i + 1
        self.filename = self.file_list[self.i]


        print("LR_confirmed = ", self.filename)

        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        print("LR_confirm")
        self.getImage()

    def LR_leftMove(self):

        shutil.copyfile("./originalTemp" + '/' + self.filename,
                        './LR_leftFolder/' + self.filename)

        self.i = self.i + 1
        self.filename = self.file_list[self.i]

        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        print("Left")
        self.getImage()

    def LR_rightMove(self):


        shutil.copyfile("./originalTemp" + '/' + self.filename,
                        './LR_rightFolder/' + self.filename)

        self.i = self.i + 1
        self.filename = self.file_list[self.i]


        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        print("rightFolder")
        self.getImage()

    def HR_leftMove(self):

        shutil.copyfile("./originalTemp" + '/' + self.filename,
                        './HR_leftFolder/' + self.filename)

        self.i = self.i + 1
        self.filename = self.file_list[self.i]

        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        print("Left")
        self.getImage()

    def HR_rightMove(self):

        shutil.copyfile("./originalTemp" + '/' + self.filename,
                        './HR_rightFolder/' + self.filename)

        self.i = self.i + 1
        self.filename = self.file_list[self.i]

        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        print("Left")
        self.getImage()

    def HR_upMove(self):

        shutil.copyfile("./originalTemp" + '/' + self.filename,
                        './HR_up/' + self.filename)
        print("saveNext first check", self.filename)


        self.i = self.i + 1
        self.filename = self.file_list[self.i]


        print("HR_UP = ", self.filename)

        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        print("HR_UP")
        self.getImage()

    def LR_upMove(self):

        shutil.copyfile("./originalTemp" + '/' + self.filename,
                        './LR_up/' + self.filename)
        print("saveNext first check", self.filename)


        self.i = self.i + 1
        self.filename = self.file_list[self.i]


        print("LR_UP = ", self.filename)

        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        print("LR_UP")
        self.getImage()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())


