import sys
import os
import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2


# from PIL import Image, ImageDraw


class MyApp(QWidget):

    def __init__(self):
        super().__init__()

        self.i = 0
        self.preI = 0
        #
        # self.title = "PyQt5 Open File"
        # self.top = 200
        # self.left = 500
        # self.width = 400
        # self.height = 300
        # self.statusbar = QStatusBar()
        # self.setMouseTracking(True)

        self.initUI()

    def initUI(self):
        self.statusbar = QStatusBar()
        #self.setMouseTracking(True)

        #self.resize(600, 600)


        self.labelImage = QLabel(self)
        self.labelImage.resize(600, 400)

        #self.setWindowTitle(self.title)
        #self.setGeometry(self.left, self.top, self.width, self.height)

        vbox = QVBoxLayout()
        #hbox = QHBoxLayout()
        grid = QGridLayout()


        self.ln_rot = QLineEdit(self)

        self.num_rot = QLineEdit(self)
        # self.inputNum = int(self.num_rot.text())
        # self.inputNum_toString = str(self.inputNum)

        #self.inputNumStr = self.inputNum_toString.replace(".", "")


        # self.pbar = QProgressBar(self)
        # self.timer = QBasicTimer()
        # self.step = 0



        self.btn1 = QPushButton("Browse")
        self.btn1.clicked.connect(self.getDir)
        self.btn2 = QPushButton("saveNext")
        self.btn2.clicked.connect(self.saveNext)
        # self.btn3 = QPushButton("justNext")
        # self.btn3.clicked.connect(self.justNext)
        self.btn4 = QPushButton("prev")
        self.btn4.clicked.connect(self.imgPre)
        self.btn5 = QPushButton('rotate image')
        self.btn5.clicked.connect(self.changeImage)

        self.btn6 = QPushButton("Quit")
        self.btn6.clicked.connect(QCoreApplication.instance().quit)
        self.btnReturn = QPushButton("Return")
        self.btnReturn.clicked.connect(self.getImage)
        self.btnSaveRoImg = QPushButton("save rotate image")
        self.btnSaveRoImg.clicked.connect(self.saveRoImg)
        self.btnDelete = QPushButton("Delete")
        self.btnDelete.clicked.connect(self.deleteImage)
        self.btnMovedIsolation = QPushButton("moveToIsolation")
        self.btnMovedIsolation.clicked.connect(self.Isolation)



        self.btnInputAngleM20 = QPushButton("-2.0")
        self.btnInputAngleM20.clicked.connect(self.M20)
        self.btnInputAngleM19 = QPushButton("-1.9")
        self.btnInputAngleM19.clicked.connect(self.M19)
        self.btnInputAngleM18 = QPushButton("-1.8")
        self.btnInputAngleM18.clicked.connect(self.M18)
        self.btnInputAngleM17 = QPushButton("-1.7")
        self.btnInputAngleM17.clicked.connect(self.M17)
        self.btnInputAngleM16 = QPushButton("-1.6")
        self.btnInputAngleM16.clicked.connect(self.M16)
        self.btnInputAngleM15 = QPushButton("-1.5")
        self.btnInputAngleM15.clicked.connect(self.M15)
        self.btnInputAngleM14 = QPushButton("-1.4")
        self.btnInputAngleM14.clicked.connect(self.M14)
        self.btnInputAngleM13 = QPushButton("-1.3")
        self.btnInputAngleM13.clicked.connect(self.M13)
        self.btnInputAngleM12 = QPushButton("-1.2")
        self.btnInputAngleM12.clicked.connect(self.M12)
        self.btnInputAngleM11 = QPushButton("-1.1")
        self.btnInputAngleM11.clicked.connect(self.M11)
        self.btnInputAngleM10 = QPushButton("-1.0")
        self.btnInputAngleM10.clicked.connect(self.M10)
        self.btnInputAngleM09 = QPushButton("-0.9")
        self.btnInputAngleM09.clicked.connect(self.M09)
        self.btnInputAngleM08 = QPushButton("-0.8")
        self.btnInputAngleM08.clicked.connect(self.M08)
        self.btnInputAngleM07 = QPushButton("-0.7")
        self.btnInputAngleM07.clicked.connect(self.M07)
        self.btnInputAngleM06 = QPushButton("-0.6")
        self.btnInputAngleM06.clicked.connect(self.M06)
        self.btnInputAngleM05 = QPushButton("-0.5")
        self.btnInputAngleM05.clicked.connect(self.M05)
        self.btnInputAngleM04 = QPushButton("-0.4")
        self.btnInputAngleM04.clicked.connect(self.M04)
        self.btnInputAngleM03 = QPushButton("-0.3")
        self.btnInputAngleM03.clicked.connect(self.M03)
        self.btnInputAngleM02 = QPushButton("-0.2")
        self.btnInputAngleM02.clicked.connect(self.M02)
        self.btnInputAngleM01 = QPushButton("-0.1")
        self.btnInputAngleM01.clicked.connect(self.M01)
        self.btnInputAngle0 = QPushButton("0")
        self.btnInputAngle0.clicked.connect(self.P0)
        self.btnInputAngleP01 = QPushButton("0.1")
        self.btnInputAngleP01.clicked.connect(self.P01)
        self.btnInputAngleP02 = QPushButton("0.2")
        self.btnInputAngleP02.clicked.connect(self.P02)
        self.btnInputAngleP03 = QPushButton("0.3")
        self.btnInputAngleP03.clicked.connect(self.P03)
        self.btnInputAngleP04 = QPushButton("0.4")
        self.btnInputAngleP04.clicked.connect(self.P04)
        self.btnInputAngleP05 = QPushButton("0.5")
        self.btnInputAngleP05.clicked.connect(self.P05)
        self.btnInputAngleP06 = QPushButton("0.6")
        self.btnInputAngleP06.clicked.connect(self.P06)
        self.btnInputAngleP07 = QPushButton("0.7")
        self.btnInputAngleP07.clicked.connect(self.P07)
        self.btnInputAngleP08 = QPushButton("0.8")
        self.btnInputAngleP08.clicked.connect(self.P08)
        self.btnInputAngleP09 = QPushButton("0.9")
        self.btnInputAngleP09.clicked.connect(self.P09)
        self.btnInputAngleP10 = QPushButton("1")
        self.btnInputAngleP10.clicked.connect(self.P10)
        self.btnInputAngleP11 = QPushButton("1.1")
        self.btnInputAngleP11.clicked.connect(self.P11)
        self.btnInputAngleP12 = QPushButton("1.2")
        self.btnInputAngleP12.clicked.connect(self.P12)
        self.btnInputAngleP13 = QPushButton("1.3")
        self.btnInputAngleP13.clicked.connect(self.P13)
        self.btnInputAngleP14 = QPushButton("1.4")
        self.btnInputAngleP14.clicked.connect(self.P14)
        self.btnInputAngleP15 = QPushButton("1.5")
        self.btnInputAngleP15.clicked.connect(self.P15)
        self.btnInputAngleP16 = QPushButton("1.6")
        self.btnInputAngleP16.clicked.connect(self.P16)
        self.btnInputAngleP17 = QPushButton("1.7")
        self.btnInputAngleP17.clicked.connect(self.P17)
        self.btnInputAngleP18 = QPushButton("1.8")
        self.btnInputAngleP18.clicked.connect(self.P18)
        self.btnInputAngleP19 = QPushButton("1.9")
        self.btnInputAngleP19.clicked.connect(self.P19)
        self.btnInputAngleP20 = QPushButton("2")
        self.btnInputAngleP20.clicked.connect(self.P20)

        #self.btn5.clicked.connect(self.finish)

        self.labelImage = QLabel(self)
        # self.labelCvImage = QLabel(self)
        # self.label.resize(300, 300)



        vbox.addWidget(self.btn1)
        vbox.addWidget(self.btn2)
        vbox.addWidget(self.btn4)
        vbox.addWidget(self.labelImage)
        vbox.addWidget(self.num_rot)
        #vbox.addWidget(self.statusbar)

        grid.addWidget(self.btnInputAngleM20, 0, 0)
        grid.addWidget(self.btnInputAngleM19, 0, 1)
        grid.addWidget(self.btnInputAngleM18, 0, 2)
        grid.addWidget(self.btnInputAngleM17, 0, 3)
        grid.addWidget(self.btnInputAngleM16, 0, 4)
        grid.addWidget(self.btnInputAngleM15, 1, 0)
        grid.addWidget(self.btnInputAngleM14, 1, 1)
        grid.addWidget(self.btnInputAngleM13, 1, 2)
        grid.addWidget(self.btnInputAngleM12, 1, 3)
        grid.addWidget(self.btnInputAngleM11, 1, 4)
        grid.addWidget(self.btnInputAngleM10, 2, 0)
        grid.addWidget(self.btnInputAngleM09, 2, 1)
        grid.addWidget(self.btnInputAngleM08, 2, 2)
        grid.addWidget(self.btnInputAngleM07, 2, 3)
        grid.addWidget(self.btnInputAngleM06, 2, 4)
        grid.addWidget(self.btnInputAngleM05, 3, 0)
        grid.addWidget(self.btnInputAngleM04, 3, 1)
        grid.addWidget(self.btnInputAngleM03, 3, 2)
        grid.addWidget(self.btnInputAngleM02, 3, 3)
        grid.addWidget(self.btnInputAngleM01, 3, 4)
        grid.addWidget(self.btnInputAngle0, 4, 0)
        grid.addWidget(self.btnInputAngleP01, 4, 1)
        grid.addWidget(self.btnInputAngleP02, 4, 2)
        grid.addWidget(self.btnInputAngleP03, 4, 3)
        grid.addWidget(self.btnInputAngleP04, 4, 4)
        grid.addWidget(self.btnInputAngleP05, 5, 0)
        grid.addWidget(self.btnInputAngleP06, 5, 1)
        grid.addWidget(self.btnInputAngleP07, 5, 2)
        grid.addWidget(self.btnInputAngleP08, 5, 3)
        grid.addWidget(self.btnInputAngleP09, 5, 4)
        grid.addWidget(self.btnInputAngleP10, 6, 0)
        grid.addWidget(self.btnInputAngleP11, 6, 1)
        grid.addWidget(self.btnInputAngleP12, 6, 2)
        grid.addWidget(self.btnInputAngleP13, 6, 3)
        grid.addWidget(self.btnInputAngleP14, 6, 4)
        grid.addWidget(self.btnInputAngleP15, 7, 0)
        grid.addWidget(self.btnInputAngleP16, 7, 1)
        grid.addWidget(self.btnInputAngleP17, 7, 2)
        grid.addWidget(self.btnInputAngleP18, 7, 3)
        grid.addWidget(self.btnInputAngleP19, 7, 4)
        grid.addWidget(self.btnInputAngleP20, 8, 0)

        vbox.addLayout(grid)
        vbox.addWidget(self.ln_rot)
        vbox.addWidget(self.btn5)
        vbox.addWidget(self.btnReturn)
        vbox.addWidget(self.btnSaveRoImg)
        vbox.addWidget(self.btnDelete)
        vbox.addWidget(self.btnMovedIsolation)
        #vbox.addWidget(self.pbar)
        #vbox.addWidget(self.btn3)
        vbox.addWidget(self.btn6)


        self.setLayout(vbox)
        self.setLayout(grid)




        self.show()

    # def mouseMoveEvent(self, event):
    #     txt = "x = {0}, y = {1}, global = ({2}, {3})".format(event.x(), event.y(), event.globalX(), event.globalY())
    #     self.statusbar.showMessage(txt)

    def getDir(self):
        self.path = QFileDialog.getExistingDirectory(self, 'open folder', '/home/ducati/GTAproject/work/guimg')
        if not os.path.exists('./workedImage'):
            os.mkdir('./workedImage')

        if not os.path.exists('./temp'):
            os.mkdir('./temp')

        if not os.path.exists('./originalTemp'):
            os.mkdir('./originalTemp')

        self.file_list = os.listdir(self.path)
        self.file_list.sort()

        self.filename = self.file_list[self.i]
        print(self.filename)

        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        self.getImage()

    def getImage(self):

        # file_list = os.listdir(self.path)
        # file_list.sort()
        #
        # self.filename = file_list[self.i]
        # print(self.filename)

        #shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        self.pixmap = QPixmap("./originalTemp" + '/' + self.filename)
        self.pixmap = self.pixmap.scaled(520, 110)
        self.labelImage.setPixmap(self.pixmap)

    def deleteImage(self):

        os.remove("./originalTemp" + '/' + self.filename)

        self.i = self.i + 1
        self.filename = self.file_list[self.i]

        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        print("deleteImage")

        self.getImage()

    def Isolation(self):

        if not os.path.exists("./isollat"):
            os.mkdir("./isolat")

        shutil.move("./originalTemp" + "/" + self.filename, "./isolat" + "/" + self.filename)

        self.i = self.i + 1
        self.filename = self.file_list[self.i]

        self.getImage()

    def preGetImage(self):
        self.prePixmap = QPixmap("./originalTemp/" + self.preFilename)
        self.prePixmap = self.prePixmap.scaled(520, 110)
        self.labelImage.setPixmap(self.prePixmap)

    def saveNext(self):

        self.inputNum = int(self.num_rot.text())
        self.inputNum_toString = str(self.inputNum)

        shutil.copyfile("./originalTemp" + '/' + self.filename, './workedImage/' + self.inputNum_toString + self.filename)
        print("saveNext first check", self.filename)



        #print(os.path.isfile(self.path + '/' + "*"))

        # if not os.path.isfile(self.path + '/' + "*"):
        #     self.messageBox()


        self.i = self.i + 1
        self.filename = self.file_list[self.i]

        #if not self.file_list[self.i]:

        print("saveNext = ", self.filename)


        shutil.move(self.path + "/" + self.filename, './originalTemp/' + self.filename)

        print("saveNext")
        self.getImage()

    # def justNext(self):
    #     self.i = self.i + 1
    #     print("justNext")
    #     self.getImage()


    def imgPre(self):


        self.prePath = "./originalTemp/"

        self.preFile_list = os.listdir(self.prePath)
        self.preFile_list.reverse()

        self.preFilename = self.preFile_list[self.preI]
        print("prev = ", self.preFilename)
        print("prev")

        self.preI = self.preI - 1
        self.preGetImage()

    def saveRoImg(self):

        self.inputNum = int(self.num_rot.text())
        self.inputNum_toString = str(self.inputNum)

        cv2.imwrite(os.path.join(self.roPath, self.inputNum_toString + self.filename[:-4] + "_r" + self.str_angle + ".png"), self.testimg,[cv2.IMWRITE_PNG_COMPRESSION, 0])
        self.i = self.i + 1
        self.filename = self.file_list[self.i]

        shutil.move(self.path + "/" + self.filename, "./originalTemp/" + self.filename)
        print(self.filename)
        self.getImage()

    def changeImage(self):

        if not os.path.exists('./rotatedImage'):
            os.mkdir('./rotatedImage')

        self.roPath = './rotatedImage/'
        self.tempPath = './temp/'

        self.testimg = cv2.imread("./originalTemp" + '/' + self.filename, cv2.IMREAD_COLOR)

        height, width, _ = self.testimg.shape
        self.r_angle = float(self.ln_rot.text())
        image_center = (width / 2, height / 2)
        matrix_temp = cv2.getRotationMatrix2D(image_center, self.r_angle, 1)
        self.testimg = cv2.warpAffine(self.testimg, matrix_temp, (width, height))

        r_angle_to_str = str(self.r_angle)
        self.str_angle = r_angle_to_str.replace(".", "")

        cv2.imwrite(os.path.join(self.tempPath, self.filename[:-4] + "_r" + self.str_angle + ".png"), self.testimg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(self.filename[-4:] + "_r" + self.str_angle + ".png")
        pixmap_new = QPixmap(self.tempPath + self.filename[:-4] + "_r" + self.str_angle + ".png")
        # print(self.filename)
        pixmap_new = pixmap_new.scaled(520, 110)
        # print(self.filename)
        self.labelImage.setPixmap(pixmap_new)

        QApplication.processEvents()

    # def messageBox(self):
    #     buttonReply = QMessageBox.information(self, "warning", "no", QMessageBox.Ok)
    #
    #     if buttonReply == QMessageBox.Ok:
    #         print("Ok")


    def M20(self):
        self.ln_rot.setText("-2.0")

    def M19(self):
        self.ln_rot.setText("-1.9")

    def M18(self):
        self.ln_rot.setText("-1.8")

    def M17(self):
        self.ln_rot.setText("-1.7")

    def M16(self):
        self.ln_rot.setText("-1.6")

    def M15(self):
        self.ln_rot.setText("-1.5")

    def M14(self):
        self.ln_rot.setText("-1.4")

    def M13(self):
        self.ln_rot.setText("-1.3")

    def M12(self):
        self.ln_rot.setText("-1.2")

    def M11(self):
        self.ln_rot.setText("-1.1")

    def M10(self):
        self.ln_rot.setText("-1.0")

    def M09(self):
        self.ln_rot.setText("-0.9")

    def M08(self):
        self.ln_rot.setText("-0.8")

    def M07(self):
        self.ln_rot.setText("-0.7")

    def M06(self):
        self.ln_rot.setText("-0.6")

    def M05(self):
        self.ln_rot.setText("-0.5")

    def M04(self):
        self.ln_rot.setText("-0.4")

    def M03(self):
        self.ln_rot.setText("-0.3")

    def M02(self):
        self.ln_rot.setText("-0.2")

    def M01(self):
        self.ln_rot.setText("-0.1")

    def P0(self):
        self.ln_rot.setText("0")

    def P01(self):
        self.ln_rot.setText("0.1")

    def P02(self):
        self.ln_rot.setText("0.2")

    def P03(self):
        self.ln_rot.setText("0.3")

    def P04(self):
        self.ln_rot.setText("0.4")

    def P05(self):
        self.ln_rot.setText("0.5")

    def P06(self):
        self.ln_rot.setText("0.6")

    def P07(self):
        self.ln_rot.setText("0.7")

    def P08(self):
        self.ln_rot.setText("0.8")

    def P09(self):
        self.ln_rot.setText("0.9")

    def P10(self):
        self.ln_rot.setText("1")

    def P11(self):
        self.ln_rot.setText("1.1")

    def P12(self):
        self.ln_rot.setText("1.2")

    def P13(self):
        self.ln_rot.setText("1.3")

    def P14(self):
        self.ln_rot.setText("1.4")

    def P15(self):
        self.ln_rot.setText("1.5")

    def P16(self):
        self.ln_rot.setText("1.6")

    def P17(self):
        self.ln_rot.setText("1.7")

    def P18(self):
        self.ln_rot.setText("1.8")

    def P19(self):
        self.ln_rot.setText("1.9")

    def P20(self):
        self.ln_rot.setText("2")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())


