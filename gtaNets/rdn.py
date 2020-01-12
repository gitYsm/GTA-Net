import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input,Convolution2D, Add, Reshape, Concatenate
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as k
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from subpixel import Subpixel
from PIL import Image
import cv2

from tf_subpixel import SubpixelConv2D

class RDN(object):
    def L1_loss(self, y_true, y_pred):
        return k.mean(k.abs(y_true-y_pred))

    def RDBlock(self, block_input, name, block_no=6, g=32):
        li = [block_input]
        pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1,1),
                            padding='same', activation='relu',
                            name=name+'_conv1')(block_input)

        for i in range(2, block_no+1):
            li.append(pas)
            out = Concatenate(axis=self.channel_axis)(li)
            pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1,1),
                                padding='same', activation='relu',
                                name=name+'_conv'+str(i))(out)

        li.append(pas)
        out = Concatenate(axis=self.channel_axis)(li)
        feat = Convolution2D(filters=64, kernel_size=(1,1), strides=(1,1),
                            padding='same', activation='relu', 
                            name=name+'_Local_Conv')(out)

        block_output = Add()([feat, block_input])
        return block_output

    def visualize(self):
        plot_model(self.rdn_model, to_file='rdn_model.png', show_shapes=True)

    def get_model(self):
        return self.rdn_model

    def __init__(self, channel=3, patch_size=32, RDB_no=20, scale=2):

        self.channel_axis = 3
        self.first_input = Input(shape=(patch_size,patch_size,channel))
        
        self.pass1 = Convolution2D(filters=64, kernel_size=(3,3), 
            strides=(1,1), padding='same', activation='relu')(self.first_input)
        self.pass2 = Convolution2D(filters=64, kernel_size=(3,3), 
            strides=(1,1), padding='same', activation='relu')(self.pass1)

        self.RDB = self.RDBlock(self.pass2, 'RDB1')
        self.RDBlocks_list = [self.RDB,]
        for i in range(2, RDB_no+1):
            self.RDB = self.RDBlock(self.RDB, 'RDB'+str(i))
            self.RDBlocks_list.append(self.RDB)
        self.out = Concatenate(axis=self.channel_axis)(self.RDBlocks_list)
        self.out = Convolution2D(filters=64, kernel_size=(1,1), strides=(1,1),
                                padding='same')(self.out)
        self.out = Convolution2D(filters=64, kernel_size=(3,3), strides=(1,1),
                                padding='same')(self.out)

        self.output = Add()([self.out, self.pass1])


        if scale >= 2:
            self.output = Subpixel(64, (3,3), r = 2,
                        padding='same',activation='relu')(self.output)
        if scale >= 4:
            self.output = Subpixel(64, (3,3), r = 4,
                        padding='same',activation='relu')(self.output)
        if scale >= 8:
            self.output = Subpixel(64, (3,3), r = 8,
                        padding='same',activation='relu')(self.output)

        # Original
        #self.output = Convolution2D(filters=channel, kernel_size=(3,3),
                                    #strides=(1,1), padding='same')(self.output)
        
        # GTA
        self.output = Convolution2D(filters=channel, kernel_size=(3,3),
                                    activation='tanh',
                                    strides=(1,1), padding='same')(self.output)

        self.rdn_model = Model(inputs=self.first_input, outputs=self.output)


class RDN_m(object):
    def EU_D_noa(self, y_true, y_pred):
        #return k.sqrt(k.sum(k.square(y_pred-y_true)))
        return k.mean(k.square(y_pred-y_true))

    def L1_loss(self, y_true, y_pred):
        return k.mean(k.abs(y_true-y_pred))

    def EU_D(self, y_true, y_pred):
        return k.sqrt(k.sum(k.square(y_pred - y_true), axis=-1)) 
    # axis=-1? What if 0?

    def RDBlock(self, block_input, name, block_no=6, g=32):
        li = [block_input]
        pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1,1),
                            padding='same', activation='relu',
                            name=name+'_conv1')(block_input)

        for i in range(2, block_no+1):
            li.append(pas)
            out = Concatenate(axis=self.channel_axis)(li)
            pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1,1),
                                padding='same', activation='relu',
                                name=name+'_conv'+str(i))(out)

        li.append(pas)
        out = Concatenate(axis=self.channel_axis)(li)
        feat = Convolution2D(filters=64, kernel_size=(1,1), strides=(1,1),
                            padding='same', activation='relu', 
                            name=name+'_Local_Conv')(out)

        block_output = Add()([feat, block_input])
        return block_output

    def visualize(self):
        plot_model(self.rdn_model, to_file='rdn_model.png', show_shapes=True)

    def get_model(self):
        return self.rdn_model

    def __init__(self, channel=3, patch_size=32, RDB_no=20, scale=2):

        self.channel_axis = 3
        self.first_input = Input(shape=(patch_size,patch_size,channel))
        
        self.pass1 = Convolution2D(filters=64, kernel_size=(3,3), 
            strides=(1,1), padding='same', activation='relu')(self.first_input)
        self.pass2 = Convolution2D(filters=64, kernel_size=(3,3), 
            strides=(1,1), padding='same', activation='relu')(self.pass1)

        self.RDB = self.RDBlock(self.pass2, 'RDB1')
        self.RDBlocks_list = [self.RDB,]
        for i in range(2, RDB_no+1):
            self.RDB = self.RDBlock(self.RDB, 'RDB'+str(i))
            self.RDBlocks_list.append(self.RDB)
        self.out = Concatenate(axis=self.channel_axis)(self.RDBlocks_list)
        self.out = Convolution2D(filters=64, kernel_size=(1,1), strides=(1,1),
                                padding='same')(self.out)
        self.out = Convolution2D(filters=64, kernel_size=(3,3), strides=(1,1),
                                padding='same')(self.out)

        self.output = Add()([self.out, self.pass1])


        if scale >= 2:
            ''' # Original
            self.output = Subpixel(64, (3,3), r = 2,
                        padding='same',activation='relu')(self.output)
            '''
            # Modified for GTA Project ###################################
            #self.output = Subpixel(64, (3,3), r = 2,
                        #padding='same', activation='relu')(self.output)
            self.output = SubpixelConv2D((None,32,32,256),
                                            scale=2)(self.output)
            ##############################################################
        if scale >= 4:
            self.output = Subpixel(64, (3,3), r = 4,
                        padding='same',activation='relu')(self.output)
        if scale >= 8:
            self.output = Subpixel(64, (3,3), r = 8,
                        padding='same',activation='relu')(self.output)

        '''
        # Original
        self.output = Convolution2D(filters=channel, kernel_size=(3,3),
                                    strides=(1,1), padding='same')(self.output)
        '''
        # Modified for GTA Project
        self.output = Convolution2D(filters=3, kernel_size=(3,3),
                                    strides=(1,1), padding='same')(self.output)
        
        self.rdn_model = Model(inputs=self.first_input, outputs=self.output,
                                name='RND')
















