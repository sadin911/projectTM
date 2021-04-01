# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:16:46 2021

@author: chonlatid.d
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Model, load_model
from tensorflow.keras.applications import InceptionV3,InceptionResNetV2
from IPython.display import Image
import glob2
import tensorflow as tf
import datetime
from PIL import Image,ImageDraw,ImageFilter,ImageEnhance,ImageOps,ImageChops
import numpy as np
import cv2
import shutil
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.layers.merge import add, concatenate,Dot,Multiply
from tensorflow.python.keras.layers import Input, Dense, Activation , LeakyReLU, Flatten, BatchNormalization, Dropout, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import normalize
from pathlib import Path

df = pd.read_csv(r'../../LSLOGO/List/test_images_root.txt', delimiter = "\t",header=None)
path2K = df[0].tolist()
path2K = [r'../../LSLOGO/Logo-2K+/' + x for x in path2K]
sdir = glob2.glob(r"D:/project/projectTM/src/imageGroupALLV3/*/")
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs('logs',exist_ok=True)
shutil.rmtree('logs')
os.makedirs('images_viz',exist_ok=True)
train_log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
num_batch = 5
class TmClassify:
    def __init__(self):
        self.input_shape = (224,224)
        self.encoder = self.createEncoder()
        self.encoder.summary()
        
        Input1 = Input(shape=(224,224,3))
        Input2 = Input(shape=(224,224,3))
        # target = Dot(axes=1)([self.encoder(Input1), self.encoder(Input2)])
        # x = Multiply()([self.encoder(Input1), self.encoder(Input2)])
        x = concatenate(inputs = [self.encoder(Input1), self.encoder(Input2)])
        self.discriminator = self.createDiscriminator()
        y = self.discriminator(x)
        self.modelDisc = Model(inputs=[Input1,Input2],outputs=y)
        self.modelDisc.summary()
        # op = Adam(lr=0.002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.00001)
        op = Adam(0.00001)
        self.modelDisc.compile(optimizer=op,loss='binary_crossentropy',metrics=['accuracy'])

        dot_img_file = 'modelDisc.png'
        tf.keras.utils.plot_model(self.modelDisc, to_file=dot_img_file, show_shapes=True,expand_nested=False)
        self.pad_param = 5
        self.rotate_degree_param = 5
        
    def createEncoder(self):
        base_model=InceptionV3(input_shape=(224,224,3),weights=None, include_top=False) 
        x = base_model.output
        x = Flatten()(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(2048,activation='sigmoid')(x)
        model = Model(inputs=base_model.input,outputs=x,name='Encoder')
        return model
    
    def createDiscriminator(self):
        input1 = Input(shape=4096)
        x = Dense(1024,activation='relu')(input1)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dropout(0.1)(x)
        x = Dense(512,activation='relu')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dropout(0.1)(x)
        x = Dense(224,activation='relu')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dropout(0.1)(x)
        target = Dense(1,activation='sigmoid')(x)
        model = Model(inputs=input1,outputs=target,name='Discriminator')
        return model
    
    def gen_data(self,path):
     
        img_pil = Image.open(path).convert('RGB')
        pad_top = int(abs(np.random.uniform(0,self.pad_param)))
        pad_bottom = int(abs(np.random.uniform(0,self.pad_param)))
        pad_left = int(abs(np.random.uniform(0,self.pad_param)))
        pad_right = int(abs(np.random.uniform(0,self.pad_param)))
        rotate_param = np.random.uniform(0,self.rotate_degree_param)
        
        flip_flag = np.random.randint(0,2)
        mirror_flag = np.random.randint(0,2)
        
        
        # if(flip_flag):
        #     img_pil = ImageOps.flip(img_pil)
        # if(mirror_flag):
        #     img_pil = ImageOps.mirror(img_pil)
        
        blur_rad = np.random.normal(loc=0.0, scale=2, size=None)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(blur_rad))
        
        enhancer_contrat = ImageEnhance.Contrast(img_pil)
        enhancer_brightness = ImageEnhance.Brightness(img_pil)
        enhancer_color = ImageEnhance.Color(img_pil)
        brightness_factor = np.random.normal(loc=1.0, scale=2.5, size=None)
        contrast_factor = np.random.normal(loc=1.0, scale=2.5, size=None)
        color_factor = np.max([0,1-abs(np.random.normal(loc=0, scale=1, size=None))])

        translate_factor_hor = np.random.normal(loc=0, scale=5, size=None)
        translate_factor_ver = np.random.normal(loc=0, scale=5, size=None)
        

        img_pil = enhancer_contrat.enhance(contrast_factor)
        img_pil = enhancer_brightness.enhance(brightness_factor)
        img_pil = enhancer_color.enhance(color_factor)
        img_pil = ImageChops.offset(img_pil, int(translate_factor_hor), int(translate_factor_ver))
        
        img_pil = img_pil.rotate(rotate_param,resample = Image.BILINEAR,expand = True, fillcolor = (255,255,255))
        
        img = np.asarray(img_pil)
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
        img= cv2.resize(img, dsize=(self.input_shape))
        img = img/127.5-1
        target = cv2.resize(img, dsize=(256,256))
        return img,target
    
    def train(self,start_epoch, max_epoch, batch_size, viz_interval):
        max_step = sum([len(i) for i in sdir]) // batch_size
        # permu_ind = list(range(len(self.pathlist)))
        step = 0
        for epoch in range(start_epoch,max_epoch):
            # permu_ind = np.random.permutation(permu_ind)
            # real_index = 0
            for step_index in range(max_step):
                    batch_img1 = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],3))
                    batch_img2 = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],3))
                    batch_imgTarget = np.zeros((batch_size,256,256,3))
                    batch_target = np.zeros(batch_size)
                    # print("1")
                    # Same Pic
                    for batch_index in range(0,batch_size//num_batch):
                        #print(batch_index)
                        while True:
                            try:
                                rand_2k = np.random.permutation(len(path2K))
                                fp = path2K[rand_2k[0]]
                                fp_par = Path(fp).parents[0]
                                filedir = glob2.glob(str(fp_par) + '/*')
                                random_permu = np.random.permutation(len(filedir))
                                img1, imgTarget = self.gen_data(filedir[random_permu[0]])
                                img2, tmp = self.gen_data(filedir[random_permu[0]])
                                batch_target[batch_index] = 1
                                batch_img1[batch_index] = img1
                                batch_img2[batch_index] = img2
                                batch_imgTarget[batch_index] = imgTarget
                                pass
                            except:
                                continue
                            break
                    
                    # print("2")  
                    # same class dip
                    for batch_index in range(1*batch_size//num_batch,2*batch_size//num_batch):
                        #print(batch_index)
                        while True:
                            try:
                                random_permu = np.random.permutation(len(sdir))
                                filelist = glob2.glob(sdir[random_permu[batch_index]]+'/*')
                                file_permu = np.random.permutation(len(filelist))
                                img1,imgTarget = self.gen_data(filelist[file_permu[0]])
                                img2, tmp = self.gen_data(filelist[file_permu[1]])
                                batch_target[batch_index] = 1
                                batch_img1[batch_index] = img1
                                batch_img2[batch_index] = img2
                                batch_imgTarget[batch_index] = imgTarget
                                pass
                            except:
                                continue
                            break
                    # print("3")
                    # dif class
                    for batch_index in range(2*batch_size//num_batch,3*batch_size//num_batch):
                        #print(batch_index)
                        while True:
                            try:
                                rootpath1 = 0
                                rootpath2 = 0
                                while rootpath1==rootpath2:
                                    rand_2k = np.random.permutation(len(path2K))
                                    rootpath1 = path2K[rand_2k[0]].split('/')[5]
                                    rootpath2 = path2K[rand_2k[-1]].split('/')[5]
                                    # print(rootpath1,rootpath2)
                                    fp1 = path2K[rand_2k[0]]
                                    fp2 = path2K[rand_2k[1]]
                                    img1,imgTarget = self.gen_data(fp1)
                                    img2, tmp = self.gen_data(fp2)
                                    batch_target[batch_index] = 0
                                    batch_img1[batch_index] = img1
                                    batch_img2[batch_index] = img2
                                    batch_imgTarget[batch_index] = imgTarget
                                    pass
                            except:
                                continue
                            break
                   
                    # print("4")   
                    # dif class dip
                    for batch_index in range(3*batch_size//num_batch,4*batch_size//num_batch):
                        #print(batch_index)
                        while True:
                            try:
                                random_permu = np.random.permutation(len(sdir))
                                rand_2k = np.random.permutation(len(path2K))
                                filelist1 = glob2.glob(sdir[random_permu[batch_index]]+'/*')
                                filelist2 = glob2.glob(sdir[random_permu[-1-batch_index]]+'/*')
                                file_permu1 = np.random.permutation(len(filelist1))
                                file_permu2 = np.random.permutation(len(filelist2)) 
                                img1,imgTarget = self.gen_data(filelist1[file_permu1[0]])
                                img2, tmp = self.gen_data(filelist2[file_permu2[0]])
                                batch_target[batch_index] = 0
                                batch_img1[batch_index] = img1
                                batch_img2[batch_index] = img2
                                batch_imgTarget[batch_index] = imgTarget
                                pass
                            except:
                                # print('x')
                                continue
                            break
                        
                    # print("5")
                    # dif 2k dip
                    for batch_index in range(4*batch_size//num_batch,5*batch_size//num_batch):
                        #print(batch_index)
                        while True:
                            try:
                                random_permu = np.random.permutation(len(sdir))
                                rand_2k = np.random.permutation(len(path2K))
                                fp1 = path2K[rand_2k[0]]
                                filelist1 = glob2.glob(sdir[random_permu[batch_index]]+'/*')
                                filelist2 = glob2.glob(sdir[random_permu[-1-batch_index]]+'/*')
                                file_permu1 = np.random.permutation(len(filelist1))
                                file_permu2 = np.random.permutation(len(filelist2)) 
                                img1,imgTarget = self.gen_data(filelist1[file_permu1[0]])
                                img2, tmp = self.gen_data(fp1)
                                batch_target[batch_index] = 0
                                batch_img1[batch_index] = img1
                                batch_img2[batch_index] = img2
                                batch_imgTarget[batch_index] = imgTarget
                                pass
                            except:
                                continue
                            break
        
                    train_loss = self.modelDisc.train_on_batch([batch_img1,batch_img2],batch_target)
                    
                    with train_summary_writer.as_default():
                        tf.summary.scalar('lossdisc', train_loss[0], step=step)
                        tf.summary.scalar('accuracydisc', train_loss[1], step=step)

                        step = step + 1 
                    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                    print('\r epoch ' + str(epoch) + ' / ' + str(max_epoch) + '   ' + 'step ' + str(step_index) + ' / ' + str(max_step) + '=' + str(train_loss),end='\r')
                    
                    if(step_index%viz_interval==0):
                        print('ok',end='\r')
                        self.modelDisc.layers[2].save_weights('DCencoderV2.Weights.h5')
                        self.modelDisc.layers[4].save_weights('DCdiscriminatorV2.Weights.h5')

                        with train_summary_writer.as_default():
                            images1 = np.reshape(batch_img1, (-1, 224, 224, 3))
                            images2 = np.reshape(batch_img2, (-1, 224, 224, 3))
                            # print('concat')
                            # image = np.concatenate((images1[0],images2[0]),axis=1)
                            # print('concatend')
                            result = self.modelDisc.predict([images1,images2])
                            result = result.tolist()
                            # print('write')
                            # for i in range(50):
                            #     images1[i] = cv2.putText(images1[i],result[i],(0,0), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2, cv2.LINE_AA)
                            #     images2[i] = cv2.putText(images2[i],result[i],(0,0), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2, cv2.LINE_AA)
                            # print(result)
                            tf.summary.text("1 discrimated", '\n'.join(str(result[0:20])), step=step)
                            tf.summary.text("0 discrimated", '\n'.join(str(result[20:50])), step=step)
                            tf.summary.image("1 input1", ((images1+1)*127.5).astype('uint8'), max_outputs=50, step=step)
                            tf.summary.image("2 input2", ((images2+1)*127.5).astype('uint8'), max_outputs=50, step=step)
                            # tf.summary.image("0 pair", ((image+1)*127.5).astype('uint8'), max_outputs=50, step=step)
                    
if __name__ == "__main__":
    TC = TmClassify()
    # TC.encoder = load_model('DIPEncoder.h5')
    # TC.discriminator = load_model('DIPdiscriminator.h5')
    TC.encoder.load_weights('DCencoder.Weights.h5')
    TC.discriminator.load_weights('DCdiscriminator.Weights.h5')
    # TC.decoder.load_weights('ATdecoder.Weights.h5')
    # TC.model = load_model(r'DIPMatchV9.h5')
    # TC.model.layers[2].save_weights('DIPencoderWeightsV1.h5')
    # TC.model.layers[4].save_weights('DIPdiscriminatorWeightsV1.h5')
    TC.train(1,10000,50,50)
