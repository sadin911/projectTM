#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Tensorflow 2.0
"""
Created on Wed Jan  8 11:08:18 2020

@author: chonlatid
"""

import os
import shutil
import tensorflow as tf

import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
from PIL import Image
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import BatchNormalization,Dropout, MaxPooling2D , GlobalMaxPool1D,Reshape,Dropout
from tensorflow.python.keras.layers import Input,Activation, Dense, Flatten, Concatenate, LSTM, Embedding
import numpy as np
import gendata
import io
import sys
import glob2
from os.path import join
import tensorflow.keras.backend as K
import random
import tensorflow.python.keras
from tensorflow.keras.callbacks import TensorBoard
train_log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

path_input = r'D:/datasets/LSLOGO/train_and_test/train/**/'
path_test = r'/home/keng/Python/Dataset/DocData/testset/**/'
bg_path = r'/home/keng/Python/Dataset/background/**/'
save_path = r'/home/keng/docsegmentation/4connerwithsegment/'
types = ('*.bmp', '*.jpg' ,'.*gif' ,'*.png' , '*.tif') # the tuple of file types
os.makedirs('logs',exist_ok=True)
shutil.rmtree('logs')
train_log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

class DocScanner():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.gf = 16
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.input_shape = (self.img_rows, self.img_cols, self.channels)
        self.model = self.build_autoencoder()
        op = Adam(0.00001)
        self.model.load_weights(r'./cnnATECSmallDense.h5')
        # self.model.save('./tfModels/corner')
        self.model.compile(loss=['mse'],
                              optimizer=op,
                              metrics=[tf.keras.metrics.MeanSquaredError()])
        self.model.summary()
        
        if tf.test.gpu_device_name():
            print('GPU found')
        else:
            print("No GPU found")
        self.gen_data = gendata.gendata()
        
        self.pathlist = []
        self.testlist = []
        self.pathbglist = []

        #self.pathbglist = glob2.glob(bg_path)

        for files in types:
            self.pathlist.extend(glob2.glob(join(path_input, files)))
    
        print(len(self.pathlist))
        
    def build_autoencoder(self):
        input_img = Input(shape=(128, 128, 3))
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        encoded = Dense(8192)(x)
        
        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        x = Dense(8192)(encoded)
        x = Reshape((8,8,128))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Dropout(0.2)(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        return autoencoder
        
    def train(self,start_epoch, max_epoch, batch_size, viz_interval):
        log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        max_step = len(self.pathlist) // batch_size
        permu_ind = list(range(len(self.pathlist)))
        step = 0
        random.shuffle(self.pathlist)
        for epoch in range(start_epoch,max_epoch):
            permu_ind = np.random.permutation(permu_ind)
            epoch_loss = []
        
            for step_index in range(max_step):
                batch_img = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2] ))
                batch_target = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2] ))
                batch_mask = np.zeros((batch_size,self.input_shape[0],self.input_shape[1]))
                
                for batch_index in range(batch_size):
                    img,target,mask = self.gen_data.gen_data((self.pathlist[  step_index * batch_size +  batch_index ]))
                    batch_img[batch_index] = img
                    batch_target[batch_index] = target
                
                train_loss = self.model.train_on_batch(batch_img,batch_target)
                # with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss[0], step=step)
                # tf.summary.scalar('accuracy', train_loss[1], step=step)
                step = step + 1 
               
                
                # Reset metrics every epoch
           
                sys.stdout.write('\r epoch ' + str(epoch) + ' / ' + str(max_epoch) + '   ' + 'step ' + str(step_index) + ' / ' + str(max_step) + '    loss = ' + str(train_loss))

                # epoch_loss.append(loss)
                
                if(step_index % viz_interval == 0): 
                    img_viz,target,mask = self.gen_data.gen_data((self.pathlist[np.random.randint(0,len(self.pathlist))]))
                    indput_data = np.expand_dims(img_viz, axis = 0)
                    predict_mask = self.model.predict(indput_data)[0]
                    target = (target+1)*127.5
                    img_viz = (img_viz+1)*127.5
                    predict_mask = (predict_mask+1)*127.5
                    test_img = Image.fromarray(predict_mask.astype('uint8'))
                    input_img =  Image.fromarray(img_viz.astype('uint8'))
                    target_img =  Image.fromarray(target.astype('uint8'))
                    self.model.save_weights('cnnATECSmallDense.h5')
                   
                    try:
                        input_img.save("viz.jpg")
                        test_img.save("test_img.jpg")
                        target_img.save("target_img.jpg")
                        print("tested")
                
                    except IOError as e:
                        print("I/O error({0}): {1}".format(e.errno, e.strerror))
                    
        
    
if __name__ == '__main__':
  
    doc = DocScanner()
    
    doc.train(1,10000000,4,100)