import numpy as np
import cv2
#from keras.models import Model

def batch_generator(X, Y=None, label=None, OHlabel=False,  batch_size=32):
    indices = np.arange(len(X))
    if Y is not None:
        if len(X) != len(Y):
            raise ValueError
    batch = []
    if label is None:
        while True:
            np.random.shuffle(indices)
            for i in indices:
                batch.append(i)
                if len(batch) == batch_size:
                    if Y is not None:
                        yield (X[batch], Y[batch])
                    else:
                        yield X[batch]
                    batch = []
    else:
        if Y is None:
            raise ValueError
        elif len(X) != len(Y):
            raise ValueError
    
        while True:
            np.random.shuffle(indices)
            i = np.random.randint(len(X))
            if np.argmax(Y[i]) == label:
                batch.append(i)
            if len(batch) == batch_size:
                if OHlabel:
                    yield (X[batch], Y[batch])
                else:
                    yield X[batch]
                batch = []

def fat_lr_bg(slim_lr, fat=4, batch_size=32):
    while True:
        yield make_fat_lrs_simple(slim_lr, fat=fat, batch_size=batch_size)


def t_v_split(X, Y, ratio=0.1):
    lenx = len(X)
    if lenx != len(Y):
        raise ValueError
    lenv = int(lenx * ratio) # lent = lenx - lenv
    indices = np.arange(lenx)
    np.random.shuffle(indices)
    x_val = X[indices[:lenv]]
    y_val = Y[indices[:lenv]]
    x_train = X[indices[lenv:]]
    y_train = Y[indices[lenv:]]
    return (x_train, y_train, x_val, y_val)


def make_fat_lrs(slim_lr, fat=4):
    n,h,w,c = slim_lr.shape
    if (h!=32 or w!=32):
        raise ValueError
    index = np.arange(n)
    in1 = index * 3 
    in2 = index * 3 + 1
    in3 = index * 3 + 2

    
    np.random.shuffle(in1)
    np.random.shuffle(in2)
    np.random.shuffle(in3)

    while len(in1) >= fat:
        for i in range(fat):
            if in1[i] == 0:
                r1 = 0
            else :
                r1 = int(in1[i]/3)
            slr_temp = slim_lr[r1,:,:,0]
            slr_temp = np.expand_dims(slr_temp, axis=0)
            slr_temp = np.expand_dims(slr_temp, axis=3)
            if i==0:
                slr = slr_temp
            else:
                slr = np.concatenate((slr, slr_temp), axis=3)
        
        for i in range(fat):
            r1 = int(in2[i]/c)
            slr_temp = slim_lr[r1,:,:,1]
            slr_temp = np.expand_dims(slr_temp, axis=0)
            slr_temp = np.expand_dims(slr_temp, axis=3)
            slr = np.concatenate((slr, slr_temp), axis=3)

        for i in range(fat):
            r1 = int(in2[i]/c)
            slr_temp = slim_lr[r1,:,:,2]
            slr_temp = np.expand_dims(slr_temp, axis=0)
            slr_temp = np.expand_dims(slr_temp, axis=3)
            slr = np.concatenate((slr, slr_temp), axis=3)

        if len(in1) == n:
            fat_lr = slr
        else:
            fat_lr = np.concatenate((fat_lr, slr), axis=0)
        for i in range(fat):
            in1 = np.delete(in1, [0])
            in2 = np.delete(in2, [0])
            in3 = np.delete(in3, [0])
            
    return fat_lr

def make_fat_lrs_simple(slim_lr, fat=4, batch_size=32):
    n,h,w,c = slim_lr.shape
    if (h!=32 or w!=32):
        raise ValueError
    index = np.arange(n)
    del_list = np.arange(fat)
    np.random.shuffle(index)
    for j in range(batch_size):
        k = 0
        for i in index[:fat]:
            slr_temp = slim_lr[i,:,:,:]
            slr_temp = np.expand_dims(slr_temp, axis=0)
            if k==0:
                slr = slr_temp
            else:
                slr = np.concatenate((slr,slr_temp), axis=3)
            k+=1
        if j==0:
            fat_lr = slr
        else:
            fat_lr = np.concatenate((fat_lr,slr), axis=0)
        index = np.delete(index, del_list)
    return fat_lr


def make_fat_lrs_del(slim_lr, fat=4):
    n,h,w,c = slim_lr.shape
    if (h!=32 or w!=32):
        raise ValueError
    index = np.arange(n*c)
    np.random.shuffle(index)
    while len(index) >= fat*c:
        for i in range(fat*c):
            r1 = int(index[i]/c)
            r2 = index[i]%c
            slr_temp = slim_lr[r1,:,:,r2]
            slr_temp = np.expand_dims(slr_temp, axis=0)
            slr_temp = np.expand_dims(slr_temp, axis=3)
            if i==0:
                slr = slr_temp
            else:
                slr = np.concatenate((slr, slr_temp), axis=3)
        if len(index) == n*c:
            fat_lr = slr
        else:
            fat_lr = np.concatenate((fat_lr, slr), axis=0)
        for i in range(fat*c):
            index = np.delete(index, [0])
    return fat_lr

def save_val_imgs(v_generator, epochs, g1_model, g2_model=None, 
        path='./output_imgs', gray=False):
    a_batchg_lr_ex = v_generator
    if g2_model is None:
        for number in range(5):
            a_image = g1_model.predict_generator(a_batchg_lr_ex[number],steps=1)[0]
            a_image = (255*(a_image-np.min(a_image))/np.ptp(a_image)).astype(np.uint8)
            if number==0:
                h_img1 = a_image
            else:
                h_img1 = cv2.hconcat([h_img1, a_image])
        for number in range(5,10):
            a_image = g1_model.predict_generator(a_batchg_lr_ex[number],steps=1)[0]
            a_image = (255*(a_image-np.min(a_image))/np.ptp(a_image)).astype(np.uint8)
            if number==5:
                h_img2 = a_image
            else:
                h_img2 = cv2.hconcat([h_img2, a_image])
        if gray:
            src = cv2.vconcat([h_img1,h_img2])
            dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(path+'/G_SRGAN'+'{:04}'.format(epochs)+'.png',dst)
        else:
            cv2.imwrite(path+'/G_SRGAN'+'{:04}'.format(epochs)+'.png', 
                    cv2.vconcat([h_img1,h_img2]))
    else:
        for number in range(5):
            a_image1 = g1_model.predict_generator(a_batchg_lr_ex[number],steps=1)
            a_image11 = a_image1[0]
            a_image12 = g2_model.model.predict(a_image1)[0]
            a_image11 = (255*(a_image11-np.min(a_image11))/np.ptp(a_image11)).astype(np.uint8)
            a_image12 = (255*(a_image12-np.min(a_image12))/np.ptp(a_image12)).astype(np.uint8)
            if number==0:
                h_img11 = a_image11
                h_img12 = a_image12
            else:
                h_img11 = cv2.hconcat([h_img11, a_image11])
                h_img12 = cv2.hconcat([h_img12, a_image12])
        for number in range(5,10):
            a_image2 = g1_model.predict_generator(a_batchg_lr_ex[number],steps=1)
            a_image21 = a_image2[0]
            a_image22 = g2_model.model.predict(a_image2)[0]
            a_image21 = (255*(a_image21-np.min(a_image21))/np.ptp(a_image21)).astype(np.uint8)
            a_image22 = (255*(a_image22-np.min(a_image22))/np.ptp(a_image22)).astype(np.uint8)
            if number==5:
                h_img21 = a_image21
                h_img22 = a_image22
            else:
                h_img21 = cv2.hconcat([h_img21, a_image21])
                h_img22 = cv2.hconcat([h_img22, a_image22])
        cv2.imwrite(path+'/POST_GEN_'+'{:04}'.format(epochs)+'.png', cv2.vconcat([h_img11,h_img21,h_img12,h_img22]))
        #cv2.imwrite('./output_imgs/POST_GEN_'+'{:04}'.format(start+pt_e)+'.png', cv2.vconcat([h_img11,h_img21,h_img12,h_img22]))


if __name__ == '__main__':

    test = np.load('./CP_LR_T/lr_image_t_1.npy')
    a = make_fat_lrs_simple(test)
    print(a.shape)
    print(test.shape)
