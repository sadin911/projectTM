# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:47:05 2021

@author: chonlatid.d
"""

import tensorflow.python.keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense,GlobalAveragePooling2D,Flatten
from tensorflow.keras.applications import MobileNet,InceptionV3
from tensorflow.keras.callbacks import TensorBoard
from IPython.display import Image
import glob2
import tensorflow as tf
import datetime
from PIL import Image,ImageDraw,ImageFilter,ImageEnhance,ImageOps,ImageChops
import numpy as np
import cv2
import os
import io
import sys
import shutil
import cv2
from tensorflow.python.keras.layers.merge import add, concatenate
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Input, Dense, Activation , LeakyReLU, Flatten, BatchNormalization, Dropout
from tensorflow.python.keras.layers.recurrent import GRU 
from tensorflow.keras.optimizers import SGD, Adam

import pandas as pd
import pickle
from sklearn.preprocessing import normalize
from pathlib import Path


df = pd.read_csv(r'D:/datasets/LSLOGO/List/train_images_root.txt', delimiter = "\t",header=None)
pathlist = df[0].tolist()
classes = pd.read_csv(r'D:/datasets/LSLOGO/List/Logo-2K+classes.txt', delimiter = "\t",header=None)
classlist = classes[0].tolist()
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs('logs',exist_ok=True)
shutil.rmtree('logs')
os.makedirs('images_viz',exist_ok=True)
train_log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

class TmClassify:
    def __init__(self):
        self.input_shape = (224,224)
        self.filter_count = 32
        self.kernel_size = (3, 3)
        self.leakrelu_alpha = 0.2
        self.numclass = len(classlist)
        self.model = self.createModel()
        self.model.summary()
        self.pathlist = pathlist
        self.train_data_count = [len(i) for i in self.pathlist]
        classDict = {
            'class':classlist,
            'index': [*range(len(classlist))]
            }
        self.classdf = pd.DataFrame.from_dict(classDict)
        op = Adam(lr=0.0001)
        self.model.compile(optimizer=op,loss='categorical_crossentropy',metrics=['accuracy'])

        self.pad_param = 5
        self.rotate_degree_param = 5
        
    def createModel(self):
        base_model=InceptionV3(input_shape=(224,224,3),weights=None,include_top=False) 
        x=base_model.output
        x=Flatten()(x)
        x=Dense(4096,activation='relu')(x)
        preds=Dense(self.numclass,activation='softmax')(x) 
        
        model=Model(inputs=base_model.input,outputs=preds)
        
        return model
    
    def gen_data(self,random_index):
        img_pil = Image.open(os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',self.pathlist[random_index])).convert('RGB')
        
        # pad_top = int(abs(np.random.uniform(0,self.pad_param)))
        # pad_bottom = int(abs(np.random.uniform(0,self.pad_param)))
        # pad_left = int(abs(np.random.uniform(0,self.pad_param)))
        # pad_right = int(abs(np.random.uniform(0,self.pad_param)))
        # rotate_param = np.random.uniform(0,self.rotate_degree_param)
        
        # flip_flag = np.random.randint(0,1)
        # mirror_flag = np.random.randint(0,1)
        
        
        # if(flip_flag):
        #     img_pil = ImageOps.flip(img_pil)
        # if(mirror_flag):
        #     img_pil = ImageOps.mirror(img_pil)
        
        # blur_rad = np.random.normal(loc=0.0, scale=1, size=None)
        # img_pil = img_pil.filter(ImageFilter.GaussianBlur(blur_rad))
        
        # enhancer_contrat = ImageEnhance.Contrast(img_pil)
        # enhancer_brightness = ImageEnhance.Brightness(img_pil)
        # enhancer_color = ImageEnhance.Color(img_pil)
        # contrast_factor = np.random.normal(loc=1.0, scale=0.25, size=None)
        # color_factor = np.max([0,1-abs(np.random.normal(loc=0, scale=0.5, size=None))])

        # translate_factor_hor = np.random.normal(loc=0, scale=5, size=None)
        # translate_factor_ver = np.random.normal(loc=0, scale=5, size=None)
        # brightness_factor = np.random.normal(loc=1.0, scale=0.5, size=None)

        # img_pil = enhancer_contrat.enhance(contrast_factor)
        # img_pil = enhancer_brightness.enhance(brightness_factor)
        # img_pil = enhancer_color.enhance(color_factor)
        # img_pil = ImageChops.offset(img_pil, int(translate_factor_hor), int(translate_factor_ver))
        
        # img_pil = img_pil.rotate(rotate_param,resample = Image.BILINEAR,expand = True, fillcolor = (255,255,255))
        
        img = np.asarray(img_pil)
        # img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
        img= cv2.resize(img, dsize=(self.input_shape))
        img = img/127.5 - 1
        
        classT = self.pathlist[random_index]
        classT = classT.split('/')[1]
        targetT = self.classdf[self.classdf['class']==classT]
        targetI = targetT['index'].values[0]
        target = np.zeros(self.numclass)
        target[targetI] = 1
        return img,target
    
    def train(self,start_epoch, max_epoch, batch_size, viz_interval):
        max_step = sum([len(i) for i in self.pathlist]) // batch_size
        # permu_ind = list(range(len(self.pathlist)))
        minloss = 1000
        step = 0
        for epoch in range(start_epoch,max_epoch):
            # permu_ind = np.random.permutation(permu_ind)
            real_index = 0
            for step_index in range(max_step):
                    batch_img = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],3 ))
                    batch_target = np.zeros((batch_size,self.numclass))
                    # allclass = []
                    for batch_index in range(batch_size):
                        
                        random_index = np.random.randint(len(self.pathlist))
                        img,target = self.gen_data(random_index)
                        # allclass.append(targetI)
                        batch_target[batch_index] = target
                        real_index = real_index+1
                    
                    # print(len(np.unique(allclass)))
                    # save_img = (batch_img[np.random.randint(batch_size)]+1)*127.5
                    # save_img = Image.fromarray(save_img.astype('uint8'))
                    # save_img.save('a.png')
                    # print(batch_target)
                    train_loss = self.model.train_on_batch(batch_img,batch_target)
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss[0], step=step)
                        tf.summary.scalar('accuracy', train_loss[1], step=step)
                        step = step + 1 
                    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                    print('\r epoch ' + str(epoch) + ' / ' + str(max_epoch) + '   ' + 'step ' + str(step_index) + ' / ' + str(max_step) + '    loss = ' + str(train_loss))
                    
                    if(step_index%viz_interval==0):
                       self.model.save('2kInceptionDenseAug.h5')
                    # #     img,target = self.gen_data(io.BytesIO(self.image_buffer[rand_index]),self.path_buffer[rand_index])
                    # #     img = np.expand_dims(img, axis=0)
                    # #     test_result = self.model.predict(img)
                    # #     # print((test_result),(target))
                    # #     print(np.argmax(test_result),np.argmax(target))
                    
if __name__ == "__main__":
    TC = TmClassify()
    TC.model = load_model('2kInceptionDense.h5')
    # TC.model.save('Inw2KDen.h5')
    TC.train(1,10000,100,100)