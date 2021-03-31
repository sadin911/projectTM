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

df = pd.read_csv(r'D:/datasets/LSLOGO/List/train_images_root.txt', delimiter = "\t",header=None)
path2K = df[0].tolist()
path_input = r'D:/project/projectTM/src/imageGroupALLV2/**/'
types = ('*.bmp', '*.jpg' ,'.*gif' ,'*.png' , '*.tif', 'jpeg')
pathlist = []
for files in types:
    pathlist.extend(glob2.glob(os.path.join(path_input, files)))
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
        self.input_shape = (256,256)
        self.encoder = self.createEncoder()
        Input1 = Input(shape=(256,256,3))
        Input2 = Input(shape=(256,256,3))
        # target = Dot(axes=1)([self.encoder(Input1), self.encoder(Input2)])
        # x = Multiply()([self.encoder(Input1), self.encoder(Input2)])
        x = concatenate(inputs = [self.encoder(Input1), self.encoder(Input2)])
        self.discriminator = self.createDiscriminator()
        y = self.discriminator(x)
        self.modelDisc = Model(inputs=[Input1,Input2],outputs=y)
        self.modelDisc.summary()
        # op = Adam(lr=0.002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.00001)
        op = Adam(0.0002,0.5)
        self.modelDisc.compile(optimizer=op,loss='binary_crossentropy',metrics=['accuracy'])
        
        self.decoder = self.createDecoder()
        en = self.encoder(Input1)
        de = self.decoder(en)
        self.modelATEC = Model(inputs=[Input1],outputs=de)
        self.modelATEC.summary()
        # self.modelATEC.compile(optimizer=op,loss='binary_crossentropy', metrics=['accuracy'])
        
        self.modelAll = Model(inputs=[Input1,Input2],outputs=[y,de])
        self.modelAll.summary()
        self.modelAll.compile(optimizer=op,loss=['binary_crossentropy','mse'], metrics=['accuracy'],loss_weights=[1,1])
        dot_img_file = 'modelATEC.png'
        tf.keras.utils.plot_model(self.modelAll, to_file=dot_img_file, show_shapes=True)
        self.pad_param = 5
        self.rotate_degree_param = 180
        
    def createEncoder(self):
        base_model=InceptionV3(input_shape=(256,256,3),weights=None, include_top=False) 
        x = base_model.output
        x = Flatten()(x)
        x = Dense(2048,activation='sigmoid')(x)
        model = Model(inputs=base_model.input,outputs=x,name='Encoder')
        return model
    def createEncoderExtend(self):
        input1 = Input(shape=2048)
        x = Dense(2048,activation='relu')(input1)
    def createDiscriminator(self):
        input1 = Input(shape=4096)
        x = Dense(1024,activation='relu')(input1)
        x = Dropout(0.1)(x)
        x = Dense(512,activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(256,activation='relu')(x)
        x = Dropout(0.1)(x)
        target = Dense(1,activation='sigmoid')(x)
        model = Model(inputs=input1,outputs=target,name='Discriminator')
        return model
    
    def createDecoder(self):
        encoded = Input(shape=2048)
        x = Dense(2048)(encoded)
        x = Reshape((8,8,32))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
        model = Model(inputs=encoded,outputs=decoded,name='Decoder')
        return model
    
    def gen_data(self,path):
     
        img_pil = Image.open(path).convert('RGB')
        img_pil.save('test.jpg')
        pad_top = int(abs(np.random.uniform(0,self.pad_param)))
        pad_bottom = int(abs(np.random.uniform(0,self.pad_param)))
        pad_left = int(abs(np.random.uniform(0,self.pad_param)))
        pad_right = int(abs(np.random.uniform(0,self.pad_param)))
        rotate_param = np.random.uniform(0,self.rotate_degree_param)
        
        flip_flag = np.random.randint(0,2)
        mirror_flag = np.random.randint(0,2)
        
        
        if(flip_flag):
            img_pil = ImageOps.flip(img_pil)
        if(mirror_flag):
            img_pil = ImageOps.mirror(img_pil)
        
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
        return img
    
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
                    batch_target = np.zeros(batch_size)
                    # print("1")
                    # Same Pic
                    for batch_index in range(0,batch_size//num_batch):
                        #print(batch_index)
                        while True:
                            try:
                                rand_2k = np.random.permutation(len(path2K))
                                fp = os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',path2K[rand_2k[0]])
                                fp_par = Path(fp).parents[0]
                                filedir = glob2.glob(str(fp_par) + '/*')
                                random_permu = np.random.permutation(len(filedir))
                                img1 = self.gen_data(filedir[random_permu[0]])
                                img2 = self.gen_data(filedir[random_permu[0]])
                                batch_target[batch_index] = 1
                                batch_img1[batch_index] = img1
                                batch_img2[batch_index] = img2
                                pass
                            except:
                                continue
                            break
                    # print("2")
                    # Same Pic
                    for batch_index in range(batch_size//num_batch,2*batch_size//num_batch):
                        #print(batch_index)
                        while True:
                            try:
                                rand_2k = np.random.permutation(len(path2K))
                                fp = os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',path2K[rand_2k[0]])
                                fp_par = Path(fp).parents[0]
                                filedir = glob2.glob(str(fp_par) + '/*')
                                random_permu = np.random.permutation(len(filedir))
                                img1 = self.gen_data(filedir[random_permu[0]])
                                img2 = self.gen_data(filedir[random_permu[0]])
                                batch_target[batch_index] = 1
                                batch_img1[batch_index] = img1
                                batch_img2[batch_index] = img2
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
                                    rootpath1 = path2K[rand_2k[0]].split('/')[1]
                                    rootpath2 = path2K[rand_2k[-1]].split('/')[1]
                                    # print(rootpath1)
                                    # print(rootpath2)
                                    fp1 = os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',path2K[rand_2k[0]])
                                    fp2 = os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',path2K[rand_2k[-1]])
                                    img1 = self.gen_data(fp1)
                                    img2 = self.gen_data(fp2)
                                    batch_target[batch_index] = 0
                                    batch_img1[batch_index] = img1
                                    batch_img2[batch_index] = img2
                                    pass
                            except:
                                continue
                            break
                    # print("4")  
                    # same class dip
                    for batch_index in range(3*batch_size//num_batch,4*batch_size//num_batch):
                        #print(batch_index)
                        while True:
                            try:
                                random_permu = np.random.permutation(len(sdir))
                                filelist = glob2.glob(sdir[random_permu[batch_index]]+'/*')
                                file_permu = np.random.permutation(len(filelist))
                                img1 = self.gen_data(filelist[file_permu[0]])
                                img2 = self.gen_data(filelist[file_permu[1]])
                                batch_target[batch_index] = 1
                                batch_img1[batch_index] = img1
                                batch_img2[batch_index] = img2
                                pass
                            except:
                                continue
                            break
                    # print("5")   
                    # dif class dip
                    for batch_index in range(4*batch_size//num_batch,5*batch_size//num_batch):
                        #print(batch_index)
                        while True:
                            try:
                                random_permu = np.random.permutation(len(sdir))
                                rand_2k = np.random.permutation(len(path2K))
                                filelist1 = glob2.glob(sdir[random_permu[batch_index]]+'/*')
                                filelist2 = glob2.glob(sdir[random_permu[-1-batch_index]]+'/*')
                                file_permu1 = np.random.permutation(len(filelist1))
                                file_permu2 = np.random.permutation(len(filelist2)) 
                                img1 = self.gen_data(filelist1[file_permu1[0]])
                                img2 = self.gen_data(filelist2[file_permu2[0]])
                                batch_target[batch_index] = 0
                                batch_img1[batch_index] = img1
                                batch_img2[batch_index] = img2
                                pass
                            except:
                                # print('x')
                                continue
                            break
                        
                    # train_loss_disc = self.modelDisc.train_on_batch([batch_img1,batch_img2],batch_target)
                    # train_loss_atec = self.modelATEC.train_on_batch([batch_img1],batch_img1)
                    train_loss = self.modelAll.train_on_batch([batch_img1,batch_img2],[batch_target,batch_img1])
                    
                    with train_summary_writer.as_default():
                        tf.summary.scalar('losstotal', train_loss[0], step=step)
                        tf.summary.scalar('lossdisc', train_loss[1], step=step)
                        tf.summary.scalar('lossatec', train_loss[2], step=step)
                        tf.summary.scalar('accuracydisc', train_loss[3], step=step)
                        tf.summary.scalar('accuracyatec', train_loss[4], step=step)
                        step = step + 1 
                    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                    print('\r epoch ' + str(epoch) + ' / ' + str(max_epoch) + '   ' + 'step ' + str(step_index) + ' / ' + str(max_step) + '=' + str(train_loss),end='\r')
                    
                    if(step_index%viz_interval==0):
                        print('ok',end='\r')
                        self.modelAll.layers[2].save_weights('ATencoder.Weights.h5')
                        self.modelAll.layers[4].save_weights('ATdiscriminator.Weights.h5')
                        self.modelAll.layers[5].save_weights('ATdecoder.Weights.h5')
                        with train_summary_writer.as_default():
                            images1 = np.reshape(batch_img1, (-1, 256, 256, 3))
                            images2 = np.reshape(batch_img2, (-1, 256, 256, 3))
                            
                            mask1 = self.modelATEC.predict([images1])
                            mask2 = self.modelATEC.predict([images2])
                            tf.summary.image("5 input1", ((images1+1)*127.5).astype('uint8'), max_outputs=20, step=step)
                            tf.summary.image("5 output1", ((mask1+1)*127.5).astype('uint8'), max_outputs=20, step=step)
                            tf.summary.image("5 input2", ((images2+1)*127.5).astype('uint8'), max_outputs=20, step=step)
                            tf.summary.image("5 output2", ((mask2+1)*127.5).astype('uint8'), max_outputs=20, step=step)
                    
if __name__ == "__main__":
    TC = TmClassify()
    # TC.encoder = load_model('DIPEncoder.h5')
    # TC.discriminator = load_model('DIPdiscriminator.h5')
    # TC.encoder.load_weights('ATencoder.Weights.h5')
    # TC.discriminator.load_weights('ATdiscriminator.Weights.h5')
    # TC.decoder.load_weights('ATdecoder.Weights.h5')
    # TC.model = load_model(r'DIPMatchV9.h5')
    # TC.model.layers[2].save_weights('DIPencoderWeightsV1.h5')
    # TC.model.layers[4].save_weights('DIPdiscriminatorWeightsV1.h5')
    TC.train(1,10000,20,50)
