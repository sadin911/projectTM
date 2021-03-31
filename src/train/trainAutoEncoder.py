#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Tensorflow 2.0
"""
Created on Wed Jan  8 11:08:18 2020

@author: chonlatid
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import tensorflow as tf

import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
from PIL import Image
from tensorflow.keras.applications import InceptionV3,InceptionResNetV2
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
import pandas as pd
from PIL import Image,ImageDraw,ImageFilter,ImageEnhance,ImageOps,ImageChops
import cv2

train_log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
df = pd.read_csv(r'D:/datasets/LSLOGO/List/train_images_root.txt', delimiter = "\t",header=None)
path2K = df[0].tolist()
path2K = [r'D:/datasets/LSLOGO/Logo-2K+/' + x for x in path2K]
os.makedirs('logs',exist_ok=True)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs('logs',exist_ok=True)
shutil.rmtree('logs')
os.makedirs('images_viz',exist_ok=True)
train_log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

class DocScanner():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.gf = 16
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.input_shape = (self.img_rows, self.img_cols, self.channels)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        Input1 = Input(shape=(256,256,3))
        x = self.encoder(Input1)
        y = self.decoder(x)
        self.model = Model(inputs=[Input1],outputs=y)
        self.model.summary()
        op = Adam(0.0002,0.5)
        
        # self.model.save('./tfModels/corner')
        self.model.compile(loss=['binary_crossentropy'],
                              optimizer=op,
                              metrics='accuracy')
        
        
        if tf.test.gpu_device_name():
            print('GPU found')
        else:
            print("No GPU found")
        self.pad_param = 5
        self.rotate_degree_param = 5

    
        
    def build_encoder(self):
        base_model=InceptionV3(input_shape=(256,256,3),weights='imagenet',include_top=False) 
        x = base_model.output
        x = Flatten()(x)
        x = Dense(2048,activation='sigmoid')(x)
        model = Model(inputs=base_model.input,outputs=x)
        return model
    
    def build_decoder(self):
        encoded = Input(shape=2048)
        x = Dense(2048)(encoded)
        x = Reshape((8,8,32))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(momentum = 0.8)(x)
        # x = Dropout(0.2)(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(momentum = 0.8)(x)
        # x = Dropout(0.2)(x)
        x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization(momentum = 0.8)(x)
        # x = Dropout(0.2)(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        model = Model(inputs=encoded,outputs=decoded)
        return model
    
    def gen_data(self,path):
     
        img_pil = Image.open(path).convert('RGB')
        img_in = np.asarray(img_pil)/255
        enhancer_color = ImageEnhance.Color(img_pil)
        color_factor = np.max([0,1-abs(np.random.normal(loc=0, scale=0.5, size=None))])
        img_pil = enhancer_color.enhance(color_factor)
        
        enhancer_color = ImageEnhance.Color(img_pil)
        
        pad_top = int(abs(np.random.uniform(0,self.pad_param)))
        pad_bottom = int(abs(np.random.uniform(0,self.pad_param)))
        pad_left = int(abs(np.random.uniform(0,self.pad_param)))
        pad_right = int(abs(np.random.uniform(0,self.pad_param)))
        rotate_param = np.random.uniform(0,self.rotate_degree_param)
        
        # flip_flag = np.random.randint(0,2)
        # print(flip_flag)
        # mirror_flag = np.random.randint(0,2)
        # print(mirror_flag)
        
        # if(flip_flag==1):
        #     img_pil = ImageOps.flip(img_pil)
        # if(mirror_flag==1):
        #     img_pil = ImageOps.mirror(img_pil)
        
        blur_rad = np.random.normal(loc=0.0, scale=1, size=None)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(blur_rad))
        
        enhancer_contrat = ImageEnhance.Contrast(img_pil)
        enhancer_brightness = ImageEnhance.Brightness(img_pil)
        
        brightness_factor = np.random.normal(loc=1.0, scale=0.25, size=None)
        contrast_factor = np.random.normal(loc=1.0, scale=0.25, size=None)
        

        translate_factor_hor = np.random.normal(loc=0, scale=5, size=None)
        translate_factor_ver = np.random.normal(loc=0, scale=5, size=None)
        

        img_pil = enhancer_contrat.enhance(contrast_factor)
        img_pil = enhancer_brightness.enhance(brightness_factor)
        
        img_pil = ImageChops.offset(img_pil, int(translate_factor_hor), int(translate_factor_ver))
        
        img_pil = img_pil.rotate(rotate_param,resample = Image.BILINEAR,expand = True, fillcolor = (255,255,255))
        
        img = np.asarray(img_pil)
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
        img = cv2.resize(img, dsize=(256,256))
        img = img/255
        return img_in, img
    
    def train(self,start_epoch, max_epoch, batch_size, viz_interval):
        max_step = len(path2K) // batch_size
        permu_ind = list(range(len(path2K)))
        step = 0
        random.shuffle(path2K)
        for epoch in range(start_epoch,max_epoch):
            permu_ind = np.random.permutation(permu_ind)
        
            for step_index in range(max_step):
                batch_img = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2] ))
                batch_target = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2] ))

                for batch_index in range(batch_size):
                    rand_f = np.random.randint(0,len(path2K))
                    target,img = self.gen_data(path2K[rand_f])
                    batch_img[batch_index] = img
                    batch_target[batch_index] = target
                
                train_loss = self.model.train_on_batch(batch_img,batch_target)
                with train_summary_writer.as_default():
                       tf.summary.scalar('loss', abs(train_loss[0]), step=step)
                       tf.summary.scalar('accuracy', abs(train_loss[1]), step=step)
                       step = step + 1 
               
                
                # Reset metrics every epoch
           
                print('\r epoch ' + str(epoch) + ' / ' + str(max_epoch) + '   ' + 'step ' + str(step_index) + ' / ' + str(max_step) + '    loss = ' + str(train_loss),end='\r')

                # epoch_loss.append(loss)
                
                if(step_index % viz_interval == 0): 
                    # rand_f = np.random.randint(0,len(path2K))
                    # target,img_viz = self.gen_data(path2K[rand_f])
                    # indput_data = np.expand_dims(img_viz, axis = 0)
                    # predict_mask = self.model.predict(indput_data)[0]
                    # test_raw =  Image.fromarray(predict_mask.astype('uint8'))
                    # target = (target+1)*127.5
                    # img_viz = (img_viz+1)*127.5
                    # predict_mask = (predict_mask+1)*127.5
                    # test_img = Image.fromarray(predict_mask.astype('uint8'))
                    # input_img =  Image.fromarray(img_viz.astype('uint8'))
                    # target_img =  Image.fromarray(target.astype('uint8'))
                    self.model.layers[1].save_weights('encoder.weights.h5')
                    self.model.save_weights('cnnATECSmallDense.weights.h5')
                    self.model.save('cnnATECSmallDense.h5')
                    try:
                        with train_summary_writer.as_default():
                            images = np.reshape(batch_img[0:5], (-1, 256, 256, 3))
                            images = images
                            mask = self.model.predict(images)
                            tf.summary.image("5 input", ((images)*255).astype('uint8'), max_outputs=5, step=step)
                            tf.summary.image("5 output", ((mask)*255).astype('uint8'), max_outputs=5, step=step)
                            
                            # b_mask = (mask[0]+1)*127.5
                            # b_img = Image.fromarray(b_mask.astype('uint8'))
                            # b_img.save('testMask0.jpg')
                            # b_mask = (mask[1]+1)*127.5
                            # b_img = Image.fromarray(b_mask.astype('uint8'))
                            # b_img.save('testMask1.jpg')
                            
                        # input_img.save("viz.jpg")
                        # test_img.save("test_img.jpg")
                        # target_img.save("target_img.jpg")
                        # test_raw.save("test_raw.jpg")
                        
                
                    except IOError as e:
                        print("I/O error({0}): {1}".format(e.errno, e.strerror))
                    
        
    
if __name__ == '__main__':
    doc = DocScanner()
    doc.model.load_weights('cnnATECSmallDense.weights.h5')
    doc.train(1,10000000,50,100)