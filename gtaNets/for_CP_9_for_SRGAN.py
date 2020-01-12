import argparse
parser = argparse.ArgumentParser(description='ArgParser')
parser.add_argument(
        '--tryout', action='store', dest='tryout',
        default=10001)
parser.add_argument(
        '--epochs', action='store', dest='epochs',
        default=1)
parser.add_argument(
        '--batch_size', action='store', dest='batch_size',
        default=16)
parser.add_argument(
        '--learning_rate', action='store', dest='learning_rate',
        default=0.00003)
parser.add_argument(
        '--checkpoint', action='store', dest='checkpoint',
        default=None)
parser.add_argument(
        '--samples', action='store', dest='samples',
        default=512)
parser.add_argument(
        '--scale', action='store', dest='scale',
        default=2)
parser.add_argument(
        '--data', action='store', dest='data',
        default=None)
parser.add_argument(
        '--GTA_train', action='store', dest='gt',
        default=False)
parser.add_argument(
        '--Post_train', action='store', dest='pt',
        default=False)
parser.add_argument(
        '--teacher_train', action='store', dest='tt',
        default=False)
parser.add_argument(
        '--test_image', action='store', dest='test_image',
        default=None)
parser.add_argument(
        '--test', action='store', dest='test',
        default=False)

parsed = parser.parse_args()
lr = float(parsed.learning_rate)
bs = int(parsed.batch_size)
eps = int(parsed.epochs)
tryout = int(parsed.tryout)
samples = int(parsed.samples)
scale = int(parsed.scale)
data = parsed.data
data = './CP_DATA/hr_image_v4.npy'
data_label = data[:-7] + '_label_v4.npy'
if data is None:
    if parsed.test_image is None:
        raise ValueError
test_image = parsed.test_image
chk = str(parsed.checkpoint)

import numpy as np
import cv2
from keras.models import Model
from keras.applications import VGG19
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, Adadelta
from networks import Discriminator_Model, SRGAN, Perceptual_Teacher
from networks import Generator_Model
from rdn import RDN, RDN_m # Generator Model
from gta_utils import batch_generator, t_v_split, make_fat_lrs
from gta_utils import fat_lr_bg, save_val_imgs

#from dcgan import DCGANG, DCGAND, DCGAN
#from time import time
#from keras.callbacks import TensorBoard
#tensorboad = TensorBoard(log_dir='logs/{}'.format(time()))

##### Load Data #####################################################
print()
print('          Load Data')
print()
data = np.load(data)
data_label = np.load(data_label)
#print(data.shape)
#print(data_label.shape)
data_lr_t_path = './CP_DATA/LRV5T/lr_image_t_0.npy'
data_lr_t_path = data_lr_t_path[:-5]
data_lr_v_path = './CP_DATA/LRV5V/lr_image_v_0.npy' #########
data_lr_v_path = data_lr_v_path[:-5]
print(data_lr_t_path)
print(data_lr_v_path)
data_lr_t = []
for i in range(10):
    data_lr_t.append(np.load(data_lr_t_path+str(i)+'.npy'))
data_lr_t = np.asarray(data_lr_t)
data_lr_v = []
for i in range(10):
    data_lr_v.append(np.load(data_lr_v_path+str(i)+'.npy'))
data_lr_v = np.asarray(data_lr_v)
lr_shape = (32,32,12)
hr_shape = (32*scale, 32*scale, 3)

##### Define Optimizer ##############################################
print()
#print('          Define Optimizer')
print()
lr = 0.00002
optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
                decay=lr/2, amsgrad=False)
optimizer1 = Adadelta()
optimizer2 = 'rmsprop'

##### RDN ###########################################################
print()
#print('          Initialize RDN (Generator)')
print()
#rdn_init = RDN_m(channel=12, patch_size=32, RDB_no=20, scale=scale)
#rdn = rdn_init.rdn_model
#rdn_init = Generator_Model()
#rdn = rdn_init.generator_model
#rdn_init.visualize()
from networks import Gen_as_per_paper, Gen_new
rdn_init = Gen_as_per_paper()
rdn = rdn_init.g_model_p
#rdn.summary()

##### Perceptual Teacher ############################################
##### teacher_model: Feature Extractor
##### interm1: Extracted Feature
#####################################################################
print()
#print('          Initialize Perceptually Teaching Model')
print()
c_model = Perceptual_Teacher(64,64,3,imagenet=False)
interm1 = Model(inputs=c_model.teacher_model.input,
               outputs=c_model.teacher_model.get_layer('block5_conv4').output,
               name='interm1')

##### Discriminator ##################################################
#print('          Initialize Discriminator')
print()
d_init = Discriminator_Model(64,64,3,filters=64)
print()
d_model = d_init.discriminator_model
#d_model.summary()

d_t_init = Discriminator_Model(64,64,3,filters=64)
d_model_t = d_t_init.discriminator_model

##### SRGAN #########################################################
print()
#print('          Construct SRGAN Structure')
print()
srgan_init = SRGAN(d_model, rdn, interm1, 64,64,32,32,12)
srgan_model = srgan_init.srgan_model

###### Model Complile ###############################################
#print('\nCheck interm1 Layers')
for layer in interm1.layers:
    layer.trainable = False
    #print(layer, layer.trainable)
interm1.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

#print('\nCheck Discriminator Layers')
for layer in d_model.layers:
    layer.trainable = False
    #print(layer, layer.trainable)
#d_model.compile(loss='mse', # or mse?
                #optimizer=Adam(0.0002,0.5), metrics=['accuracy'])
d_model.compile(loss='binary_crossentropy', # or mse?
                optimizer=optimizer, metrics=['accuracy'])
d_model_t.compile(loss='binary_crossentropy',
                optimizer=optimizer, metrics=['accuracy'])
#d_model_t.load_weights('./outputs/log01/D_RDN_weights_1500.h5')

#print('\nCheck rdn Layers')
#for layer in rdn.layers:
    #print(layer, layer.trainable)

#print('\nCheck SRGAN Layers')
srgan_model.compile(loss=['binary_crossentropy', rdn_init.EU_D_noa],
                    loss_weights=[0.001, 1],
                    optimizer=optimizer)
                    #optimizer=rdn_init.optimizer)

#for layer in srgan_model.layers:
    #print(layer.name, layer.trainable)

##### Load Weights for Models #######################################
if chk != 'None':
    print()
    #print('          Loading Checkpoint')
    print()
c_model.teacher_model.load_weights('./tt_vgg19_ep40_hr_v4_.h5')
#rdn.load_weights('./RDNG_weights500.h5')
#d_model_t.load_weights('./D_RDNG_weights_500.h5')

##### Scale and Prepare the Data ###################################
print()
#print('          Scale and Prepare the Data')
print()
data = data.astype('float32') / 255 # Scale the HR Data to [0,1]
x_t, y_t, x_v, y_v = t_v_split(data, data_label, 0.2) # For Validation
x_lr_t = []
x_lr_v = []
# Scale th LR Data to [-1,1]
for i in range(10):
    data_lr_t[i] = (data_lr_t[i].astype('float32') - 127.5) / 127.5
    x_lr_t.append(data_lr_t[i])
for i in range(10):
    data_lr_v[i] = (data_lr_v[i].astype('float32') - 127.5) / 127.5
    x_lr_v.append(data_lr_v[i])
x_lr_t = np.asarray(x_lr_t)
x_lr_v = np.asarray(x_lr_v)

a_batchg_lr_ex = []
for i in range(10):
    a_batchg_lr_ex.append(fat_lr_bg(x_lr_v[i], batch_size=1))

##### GTA Training Mode #############################################
if parsed.gt:
    batch_size_D = 32
    batch_size_G = 32
    a_batchg_hr_D = batch_generator(x_t, batch_size=batch_size_D)
    a_batchg_lr_D = []
    for i in range(10):
        a_batchg_lr_D.append(fat_lr_bg(x_lr_t[i], batch_size=batch_size_D))
        print(x_lr_t[i].shape)
    #sr_label = np.zeros((batch_size_D,) + (4,4,1))
    #hr_label = np.ones((batch_size_D,) + (4,4,1))
    sr_label = np.zeros((batch_size_D,) + (16,16,1))
    hr_label = np.ones((batch_size_D,) + (16,16,1))
    
    a_batchg_lr_G = []
    for i in range(10):
        a_batchg_lr_G.append(fat_lr_bg(x_lr_t[i], batch_size=batch_size_G))
    a_batchg_hr_G = []
    for i in range(10):
        a_batchg_hr_G.append(batch_generator(x_t, y_t, label=i, 
                                             batch_size=batch_size_G))
    #hr_vailidity = np.ones((batch_size_G,) + (4,4,1))
    hr_vailidity = np.ones((batch_size_G,) + (16,16,1))

    for GTA_e in range(5001):
        ##### Discriminator Training
        if (GTA_e%2==0 and GTA_e<200) or d_out>0.6: # 100 -> 300
            for i in range(1): # Train with False label
                j = np.random.randint(10)
                a_batch_sr = rdn.predict_generator(a_batchg_lr_D[j], steps=1)
                d_model_t.fit(a_batch_sr, sr_label, epochs=1)
            for i in range(1): # Train with True label
                a_batch_hr_D = next(a_batchg_hr_D)
                d_model_t.fit(a_batch_hr_D, hr_label, epochs=1)
            d_model.set_weights(d_model_t.get_weights())
        j = np.random.randint(10)
        a_batch_sr = rdn.predict_generator(a_batchg_lr_D[j], steps=1)
        d_model_pred = d_model.predict(a_batch_sr)
        d_out = d_model_pred.mean()
        print('Discriminator Output: ', d_out)
    
        ##### Generator Training
        for i in range(2):
            #print('G Training', i+1)
            j = np.random.randint(10)
            hr_feature = interm1.predict(next(a_batchg_hr_G[j]))
            srgan_model.fit(next(a_batchg_lr_G[j]), 
                                       [hr_vailidity, hr_feature])

        print('GLOBAL EPOCH NO. ',GTA_e,'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'*2)
        if (GTA_e)%10==0:
            save_val_imgs(a_batchg_lr_ex, GTA_e, rdn)
        if (GTA_e)%500==0:
            rdn.save_weights('G_SRGAN_weights'+str(GTA_e)+'.h5')
            d_model.save_weights('D_SRGAN_weights_'+str(GTA_e)+'.h5')

##### Perceptual Teacher Training Mode ##############################
if parsed.tt:
    c_model = Perceptual_Teacher(64,64,3,imagenet=False)
    c_model.teacher_model.compile(loss=['mse'], optimizer=optimizer, 
                                  metrics=['accuracy'])
    c_model.teacher_model.load_weights('tt_vgg19_ep20.h5')
    #c_model.base_model.summary()
    #c_model.teacher_model.summary()
    tbs = 64 # Training Batch Size
    from math import ceil
    steps_per_epoch = ceil(len(data)/tbs)
    v_steps = ceil(len(data)*0.1/tbs)
    print('STEPS PER EPOCH', steps_per_epoch)
    a_batch_t = batch_generator(x_t, y_t, batch_size=tbs)
    a_batch_v = batch_generator(x_v, y_v, batch_size=tbs)
    for ep in range(2):
        c_model.teacher_model.fit_generator(a_batch_t, 
                                steps_per_epoch=steps_per_epoch,
                                validation_data=a_batch_v, 
                                validation_steps=v_steps, epochs=20)
        c_model.teacher_model.save_weights('tt_vgg19_ep'+str((ep+1)*20)+'_hr_v4_.h5')
    print('LOADED')

##### Post Training Mode ############################################
if parsed.pt:
    post_gen = DCGANG()
    post_gen.model.compile(loss='binary_crossentropy',
                            optimizer=post_gen.optimizer,
                            metrics=['accuracy'])
    #post_gen.model.summary()
    
    post_di = DCGAND()
    post_di.model.compile(loss='binary_crossentropy',
                            optimizer=post_di.optimizer,
                            metrics=['accuracy'])

    for layer in post_di.model.layers:
        print(layer, layer.trainable)

    post_gan = DCGAN(post_di.model,post_gen.model)
    post_gan.model.compile(loss='binary_crossentropy',
                            optimizer=post_gan.optimizer,
                            metrics=['accuracy'])

    for layer in post_gan.model.layers:
        print(layer, layer.trainable)

    rdn.load_weights('./G_SRGAN_weights1000.h5') #####
    batch_size = 32
    a_batchg_lr_PD = []
    for i in range(10):
        a_batchg_lr_PD.append(fat_lr_bg(x_lr_t[i], batch_size=batch_size))
    a_batchg_lr_PG = []
    for i in range(10):
        a_batchg_lr_PG.append(fat_lr_bg(x_lr_t[i], batch_size=batch_size))
    a_batchg_hr = []
    for i in range(10):
        a_batchg_hr.append(batch_generator(x_t, y_t, label=i, 
                                             batch_size=batch_size))
    label_false = np.zeros([batch_size,1])
    label_true  = np.ones([batch_size,1])


    #post_di.model.load_weights('./weights/post_di_model500.h5')
    #post_gen.model.load_weights('./weights/post_gen_model500.h5')
    #start = 500
    start = 0
    for pt_e in range(5001):
        j = np.random.randint(10)
        a_batch_sr_PD = rdn.predict_generator(a_batchg_lr_PD[j], steps=1)
        a_batch_sr2 = post_gen.model.predict(a_batch_sr_PD)
        a_batch_hr = next(a_batchg_hr[j])
        post_di.model.fit(a_batch_sr2, label_false)
        post_di.model.fit(a_batch_hr, label_true)
        for _ in range(10):
            j = np.random.randint(10)
            a_batch_sr_PG = rdn.predict_generator(a_batchg_lr_PG[j], steps=1)
            post_gan.model.fit(a_batch_sr_PG, label_true)
        if pt_e%40==0 and pt_e>0:
            post_gen.model.save_weights('./weights/post_gen_model_'+str(start+pt_e)+'.h5')
            post_di.model.save_weights('./weights/post_di_model'+str(start+pt_e)+'.h5')
        if pt_e%20==0:
            save_val_imgs(a_batch_lr_ex, rdn, post_gen)

##### Test Mode #####################################################
if parsed.test=='1':
    #from dkimgtracking_xyplusjsd import Img_Jsd_Lee
    a_batchg_hr_F = []
    for i in range(10):
        a_batchg_hr_F.append(batch_generator(x_t, y_t, label=i, 
                                             batch_size=1))
       
    img = next(a_batchg_hr_F[0])
    imgt = (255*(img[0]-np.min(img[0]))/np.ptp(img[0])).astype(np.uint8)
    cv2.imwrite('test1.png', imgt)
    hr_feature01 = interm1.predict(img)

    print(hr_feature01.shape)

    img = next(a_batchg_hr_F[0])
    imgt = (255*(img[0]-np.min(img[0]))/np.ptp(img[0])).astype(np.uint8)
    cv2.imwrite('test2.png', imgt)
    hr_feature02 = interm1.predict(img)

    img = next(a_batchg_hr_F[3])
    imgt = (255*(img[0]-np.min(img[0]))/np.ptp(img[0])).astype(np.uint8)
    cv2.imwrite('test3.png', imgt)
    hr_feature03 = interm1.predict(img)

    print(hr_feature03[0,:,:,255])
    #mse012 = np.square(hr_feature01,hr_feature02).sum(axis=-1)
    #mse013 = np.square(hr_feature01,hr_feature03).sum(axis=-1)
    print(np.sqrt(np.sum(np.square(hr_feature01-hr_feature03))))
    print(np.sqrt(np.sum(np.square(hr_feature02-hr_feature03))))
    print(np.sqrt(np.sum(np.square(hr_feature01-hr_feature02))))
    print(np.sqrt(np.sum(np.square(hr_feature01-hr_feature03),axis=-1)))
    print(np.sqrt(np.sum(np.square(hr_feature02-hr_feature03),axis=-1)))
    print(np.sqrt(np.sum(np.square(hr_feature01-hr_feature02),axis=-1)))


    '''
    test_model = Perceptual_Teacher_N(64,64,3,imagenet=False)
    test_model.teacher_model.load_weights('./new/nasnetweights3.h5')
    test_model.teacher_model.compile(loss=['mse'], optimizer=optimizer, 
                                  metrics=['accuracy'])
    # Loading weights done above (at chk)
    predict = test_model.teacher_model.predict(x_v)
    false = 0
    dn = 0
    print(len(predict))
    for _ in range(len(predict)):
        if np.argmax(predict[_]) != np.argmax(y_v[_]):
            false = false + 1
        if predict[_][np.argmax(predict[_])] < 0.99:
            dn = dn + 1
    print(false, dn)


    test_model.teacher_model.load_weights('./new/nasnetweights2.h5')
    predict = test_model.teacher_model.predict(x_v)
    false = 0
    dn = 0
    for _ in range(len(predict)):
        if np.argmax(predict[_]) != np.argmax(y_v[_]):
            false = false + 1
        if predict[_][np.argmax(predict[_])] < 0.99:
            dn = dn + 1
    print(false, dn)
    '''

if parsed.test=='2':
    from networks import Gen_as_per_paper
    test = Gen_as_per_paper()
    test.g_model_p.compile(loss='mse', optimizer=test.optimizer)
    test.g_model_p.summary()

if parsed.test=='3':
    from networks import Gen_new
    test = Gen_new()
    test.g_model_p.compile(loss='mse', optimizer=test.optimizer)
    test.g_model_p.summary()

if parsed.test=='4':
    rdn.load_weights('./G_SRGAN_weights2000.h5')
    for i in range(100):
        save_val_imgs(a_batchg_lr_ex, 2000+i, rdn, path='./gray', gray=True)

if parsed.test=='5':
    from show import make_a_fat_lr
    a_fat_lr = make_a_fat_lr(path='./demo')
    rdn.load_weights('./G_SRGAN_weights4500.h5')
    pred = rdn.predict(a_fat_lr)[0]
    pred = (255*(pred-np.min(pred))/np.ptp(pred)).astype(np.uint8)
    cv2.imwrite('./demo/output/prediction.png', pred)

