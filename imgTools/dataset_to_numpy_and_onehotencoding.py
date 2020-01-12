import numpy as np
import cv2
import sys
import os
from random import randint


def Label_Encoding(label_num, one_hot_encoding):

    if one_hot_encoding == "1":
        ret = np.zeros(10)
        ret[label_num] = 1
        return ret 
    else:
        ret = label_num
        return ret



if __name__ == '__main__':

    print("main exec")
    one_hot_encoding = sys.argv[1]
    label = []

    base_path = "/home/bit/final_cp_dataset/ver5/HR"
    base_path_list = os.listdir(base_path)
    base_path_list.sort()
    np_img = [] 

    for idx in range(len(base_path_list)):

        cpfolder = base_path + "/"  + base_path_list[idx]
        cpfolder_list = os.listdir(cpfolder)
        label_num = int(cpfolder[-1])

        for jdx in range(len(cpfolder_list)):

            img_path = cpfolder + "/" + cpfolder_list[jdx]

            img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            img = cv2.resize(img, dsize=(256,256))
            np_img.append(img)
            label.append(Label_Encoding(label_num, one_hot_encoding))

            #save_np = np.array(np_img)
            #savetitle = "/home/bit/final_cp_dataset/ver5/" + str(idx) + "_" + "lr_ndarray.npy"
            #np.save(savetitle, save_np)

    save_np = np.array(np_img)
    save_label = np.array(label)
    np.save('/home/bit/final_cp_dataset/ver5/256_ndarray.npy', save_np)
    np.save('/home/bit/final_cp_dataset/ver5/256_label.npy', save_label)

    #aa = np.load('/home/bit/final_cp_dataset/ver3pzz(finaldktemp)/hr_ndarray.npy')
    #bb = np.load('/home/bit/final_cp_dataset/ver3pzz(finaldktemp)/hr_label.npy')

    print(save_np.shape)
    print(save_label.shape)

