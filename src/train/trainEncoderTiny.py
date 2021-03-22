import tensorflow.python.keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense,GlobalAveragePooling2D,Flatten,Concatenate,LayerNormalization
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
from tensorflow.python.keras.layers.merge import add, concatenate,Dot,Multiply
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
path2K = df[0].tolist()
path_input = r'D:/datasets/TradeMark/trainingSet/imageGroupALL/**/'
types = ('*.bmp', '*.jpg' ,'.*gif' ,'*.png' , '*.tif', 'jpeg')
pathlist = []
for files in types:
    pathlist.extend(glob2.glob(os.path.join(path_input, files)))
sdir = glob2.glob(r"D:/datasets/TradeMark/trainingSet/imageGroupALL/*/")
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
        self.encoder = self.createEncoder()
        Input1 = Input(shape=(224,224,3))
        Input2 = Input(shape=(224,224,3))
        # target = Dot(axes=1)([self.encoder(Input1), self.encoder(Input2)])
        # x = Multiply()([self.encoder(Input1), self.encoder(Input2)])
        # x = concatenate(inputs = [self.encoder(Input1), self.encoder(Input2)])
        self.discriminator = self.createDiscriminator()
        
        x1 = self.encoder(Input1)
        x2 = self.encoder(Input2)
        y = self.discriminator([x1,x2])
        self.model = Model(inputs=[Input1,Input2],outputs=y)
        self.model.summary()
        op = Adam(lr=0.01)
        self.model.compile(optimizer=op,loss='mse',metrics=['accuracy'])
        self.pad_param = 10
        self.rotate_degree_param = 45
        
    def createEncoder(self):
        base_model=InceptionV3(input_shape=(224,224,3),weights=None,include_top=False) 
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048,activation='sigmoid')(x)
        x = LayerNormalization()(x)
        model = Model(inputs=base_model.input,outputs=x)
        return model
    
    def createDiscriminator(self):
        input1 = Input(shape=2048)
        input2 = Input(shape=2048)
        target = Dot(axes=1)([input1, input2])
        model = Model(inputs=[input1,input2],outputs=target)
        return model
    
    
    def gen_data(self,path):
        img_pil = Image.open(path).convert('RGB')
        # img_pil.save("test.jpg")
        pad_top = int(abs(np.random.uniform(0,self.pad_param)))
        pad_bottom = int(abs(np.random.uniform(0,self.pad_param)))
        pad_left = int(abs(np.random.uniform(0,self.pad_param)))
        pad_right = int(abs(np.random.uniform(0,self.pad_param)))
        rotate_param = np.random.uniform(0,self.rotate_degree_param)
        
        flip_flag = np.random.randint(0,1)
        mirror_flag = np.random.randint(0,1)
        
        
        if(flip_flag):
            img_pil = ImageOps.flip(img_pil)
        if(mirror_flag):
            img_pil = ImageOps.mirror(img_pil)
        
        blur_rad = np.random.normal(loc=0.0, scale=1, size=None)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(blur_rad))
        
        enhancer_contrat = ImageEnhance.Contrast(img_pil)
        enhancer_brightness = ImageEnhance.Brightness(img_pil)
        enhancer_color = ImageEnhance.Color(img_pil)
        brightness_factor = np.random.normal(loc=1.0, scale=0.5, size=None)
        contrast_factor = np.random.normal(loc=1.0, scale=0.5, size=None)
        color_factor = np.max([0,1-abs(np.random.normal(loc=0, scale=0.5, size=None))])

        translate_factor_hor = np.random.normal(loc=0, scale=5, size=None)
        translate_factor_ver = np.random.normal(loc=0, scale=5, size=None)
        

        img_pil = enhancer_contrat.enhance(contrast_factor)
        img_pil = enhancer_brightness.enhance(brightness_factor)
        img_pil = enhancer_color.enhance(color_factor)
        img_pil = ImageChops.offset(img_pil, int(translate_factor_hor), int(translate_factor_ver))
        
        img_pil = img_pil.rotate(rotate_param,resample = Image.BILINEAR,expand = True, fillcolor = (255))
        
        img = np.asarray(img_pil)
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
        img= cv2.resize(img, dsize=(self.input_shape))
        img = img/127.5 - 1
        
        return img
    
    def train(self,start_epoch, max_epoch, batch_size, viz_interval):
        max_step = sum([len(i) for i in sdir]) // batch_size
        # permu_ind = list(range(len(self.pathlist)))
        step = 0
        for epoch in range(start_epoch,max_epoch):
            # permu_ind = np.random.permutation(permu_ind)
            real_index = 0
            for step_index in range(max_step):
                    batch_img1 = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],3))
                    batch_img2 = np.zeros((batch_size,self.input_shape[0],self.input_shape[1],3))
                    batch_target = np.zeros(batch_size)
                    
                    for batch_index in range(batch_size//4):
                        #print(batch_index)
                        random_permu = np.random.permutation(len(sdir))
                        filelist = glob2.glob(sdir[random_permu[batch_index]]+'/*')
                        file_permu = np.random.permutation(len(filelist))
                        img1 = self.gen_data(filelist[file_permu[0]])
                        img2 = self.gen_data(filelist[file_permu[1]])
                        batch_target[batch_index] = 1
                        batch_img1[batch_index] = img1
                        batch_img2[batch_index] = img2
                     
                    
                    for batch_index in range(batch_size//4,batch_size//3):
                        #print(batch_index)
                        random_permu = np.random.permutation(len(sdir))
                        rand_2k = np.random.permutation(len(path2K))
                        fp = os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',path2K[rand_2k[0]])
                        filelist1 = glob2.glob(sdir[random_permu[batch_index]]+'/*')
                        filelist2 = glob2.glob(sdir[random_permu[-1-batch_index]]+'/*')
                        file_permu1 = np.random.permutation(len(filelist1))
                        file_permu2 = np.random.permutation(len(filelist2))
                        img1 = self.gen_data(filelist1[file_permu1[0]])
                        img2 = self.gen_data(filelist2[file_permu2[0]])
                        batch_target[batch_index] = 0
                        batch_img1[batch_index] = img1
                        batch_img2[batch_index] = img2
                    
                    for batch_index in range(batch_size//3,batch_size//2):
                        #print(batch_index)
                        random_permu = np.random.permutation(len(sdir))
                        rand_2k = np.random.permutation(len(path2K))
                        fp = os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',path2K[rand_2k[0]])
                        filelist1 = glob2.glob(sdir[random_permu[batch_index]]+'/*')
                        filelist2 = glob2.glob(sdir[random_permu[-1-batch_index]]+'/*')
                        file_permu1 = np.random.permutation(len(filelist1))
                        file_permu2 = np.random.permutation(len(filelist2))
                        img1 = self.gen_data(filelist1[file_permu1[0]])
                        img2 = self.gen_data(fp)
                        batch_target[batch_index] = 0
                        batch_img1[batch_index] = img1
                        batch_img2[batch_index] = img2
                    
                    for batch_index in range(batch_size//2,3*batch_size//4):
                        #print(batch_index)
                        rand_2k = np.random.permutation(len(path2K))
                        fp = os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',path2K[rand_2k[0]])
                        img1 = self.gen_data(fp)
                        img2 = self.gen_data(fp)
                        batch_target[batch_index] = 1
                        batch_img1[batch_index] = img1
                        batch_img2[batch_index] = img2
                      
                    for batch_index in range(3*batch_size//4,batch_size):
                        #print(batch_index)
                        rootpath1 = 0
                        rootpath2 = 0
                        while rootpath1==rootpath2:
                            rand_2k = np.random.permutation(len(path2K))
                            rootpath1 = path2K[rand_2k[0]].split('/')[1]
                            rootpath2 = path2K[rand_2k[-1]].split('/')[1]
                            fp1 = os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',path2K[rand_2k[0]])
                            fp2 = os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',path2K[rand_2k[-1]])
                            img1 = self.gen_data(fp1)
                            img2 = self.gen_data(fp2)
                            batch_target[batch_index] = 0
                            batch_img1[batch_index] = img1
                            batch_img2[batch_index] = img2
                            pass
                       
                        
                    train_loss = self.model.train_on_batch([batch_img1,batch_img2],batch_target)
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss[0], step=step)
                        tf.summary.scalar('accuracy', train_loss[1], step=step)
                        step = step + 1 
                    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                    print('\r epoch ' + str(epoch) + ' / ' + str(max_epoch) + '   ' + 'step ' + str(step_index) + ' / ' + str(max_step) + '    loss = ' + str(train_loss))
                    
                    if(step_index%viz_interval==0):
                        print('ok')
                        self.model.layers[2].save_weights('DIPencoderWeightsTinyV1.h5')
                        self.model.layers[3].save_weights('DIPdiscriminatorWeightsTinyV1.h5')
                        # self.model.save('DIPMatch.h5')
                    
if __name__ == "__main__":
    TC = TmClassify()
    # TC.encoder = load_model('DIPEncoder.h5')
    # TC.discriminator = load_model('DIPdiscriminator.h5')
    # TC.encoder.load_weights('DIPencoderWeights.h5')
    # TC.discriminator.load_weights('DIPdiscriminatorWeights.h5')
    # TC.model = load_model(r'DIPMatchTinyV1.h5')
    # TC.model.layers[2].save_weights('DIPencoderWeightsV1.h5')
    # TC.model.layers[4].save_weights('DIPdiscriminatorWeightsV1.h5')
    TC.train(1,10000,40,200)