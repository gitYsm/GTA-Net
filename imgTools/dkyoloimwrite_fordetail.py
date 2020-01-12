import numpy as np
import scipy.stats as stats
import os
import cv2
import matplotlib as plt
import imutils
import random
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import tarfile
import six.moves.urllib as urllib
import time
from os import listdir
import math


def Img_Slice(img, line):

    imgcpy = img.copy()
    classnum = []
    retimg = []

    height, width = img.shape[:2]

    for idx in range(len(line)):

        line[idx] = line[idx].split(" ")
        box_width = float(line[idx][3]) * width
        box_height = float(line[idx][4]) * height
        cent_x = float(line[idx][1]) * width
        cent_y = float(line[idx][2]) * height

        left_x = cent_x - (box_width / 2)
        left_y = cent_y - (box_height / 2)
        right_x = cent_x + (box_width / 2)
        right_y = cent_y + (box_height / 2)

        classnum.append(int(line[idx][0]))
        croped_img = imgcpy[int(left_y): int(right_y), int(left_x): int(right_x)]
        retimg.append(croped_img)

    return classnum, retimg


def Create_Folder():

    foldername = "/home/bit/added_final_cp_dataset_zerotonine"
    numberingfold = foldername + "/" + "dataset"

    if not os.path.isdir(foldername):
        os.mkdir(foldername)

    for i in range(10):

        fold = numberingfold + "_" + str(i)
        if not os.path.isdir(fold):
            os.mkdir(fold)

    return foldername


def Check_Savenum(foldername):

    ret = []
    foldernamelist = os.listdir(foldername)
    foldernamelist.sort()

    for idx in range(len(foldernamelist)):

        temp = foldername + "/" + foldernamelist[idx]
        templist = os.listdir(temp)
        ret.append(len(templist) + 1)

    return ret


if __name__ == '__main__':

    print("main exec")

    basepath = "/home/bit/Yolo_mark/x64/Release/data/img"            # <-- edit folder
    filelist = os.listdir(basepath)
    txtlist = []

    save_continue = sys.argv[1]
    foldername = Create_Folder()
    
    if save_continue == 'true':
        savenum = Check_Savenum(foldername)
    else:
        savenum = [1 for i in range(10)]

    for file in filelist:

        if file[-4:] == '.txt':
            txtlist.append(file)

    for idx,file in enumerate(txtlist):

        numbering = file[:4]
        temp = file[:-4]
        temp = basepath + "/" + temp + ".png"
        img = cv2.imread(temp, cv2.IMREAD_COLOR)

        txt = open(basepath + "/" + file, mode='rt', encoding='utf-8')
        line = txt.readlines()

        if len(line) == 0 or len(line) == 1: 
            continue

        cpnum, imgs = Img_Slice(img, line)
        for jdx in range(len(imgs)):

            savetitle = foldername + "/" + "dataset_" + str(cpnum[jdx]) + "/" + \
                        str(cpnum[jdx]) + "_" + str(savenum[cpnum[jdx]]) + ".png"

            savenum[cpnum[jdx]] += 1
            cv2.imwrite(savetitle, imgs[jdx], [cv2.IMWRITE_PNG_COMPRESSION, 0])

