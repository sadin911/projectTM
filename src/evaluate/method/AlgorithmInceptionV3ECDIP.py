# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.models import Model
from PIL import Image
import pickle
import sys
import math
import pandas as pd
from numba import cuda 

def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r',flush=True)
    
class InceptionV3ENDIP:
    def __init__(self):
        print("[init] InceptionV3ENDIP")
        self.Encoder=self.createEncoder()
        self.Encoder.load_weights(r'../DIPencoderWeightsV5.h5')
        # self.Discriminator=self.createDiscriminator()
        # self.Discriminator.load_weights(r'../DIPdiscriminatorWeightsV5.h5')
        self.database = []
        self.ID=[]
        
    def createEncoder(self):
        base_model=keras.applications.InceptionV3(input_shape=(224,224,3),weights=None,include_top=False) 
        x = base_model.output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(2048,activation='sigmoid')(x)
        model = Model(inputs=base_model.input,outputs=x)
        return model
    
    def createDiscriminator(self):
        input1 = keras.layers.Input(shape=4096)
        x = keras.layers.Dense(1024,activation='relu')(input1)
        x = keras.layers.Dense(512,activation='relu')(x)
        x = keras.layers.Dense(256,activation='relu')(x)
        target = keras.layers.Dense(1,activation='sigmoid')(x)
        model = Model(inputs=input1,outputs=target)
        return model
    
    def feature_extract(self,image_path):
        img_pil = Image.open(image_path).convert('RGB')
        img_pil = img_pil.resize((224,224))
        img_cv = np.asarray(img_pil)
        indput_data = img_cv/127.5-1
        indput_data = np.expand_dims(indput_data, axis = 0)
        out = self.Encoder.predict(indput_data)[0]
        return out
    
    def feature_extract_batch(self,ImagePathList,batch_size,pkl_save_path=None):
        print("[featureExtract][Batch] InceptionV3ENDIP")
        featureList = []
        real_index = 0
        ns = 1
        max_iter = math.ceil(len(ImagePathList)/ns/batch_size)
        for i in range(max_iter):
            batchImg = np.zeros((batch_size,224,224,3))
            for batch_index in range(batch_size):
                try:
                    img_input_path = ImagePathList[real_index]
                    img_pil = Image.open(img_input_path).convert('RGB')
                    img_pil = img_pil.resize((224,224))
                    img_cv = np.asarray(img_pil)
                    batchImg[batch_index] = (img_cv/127.5-1)  
                    print('', end='\r')
                    progressBar(real_index+1,len(ImagePathList)/ns,20)
                    real_index += 1
                    # print(f'{real_index}//{len(ImagePathList)}')
                except:
                    print('', end='\r')
                    batchImg = batchImg[:batch_index]
                    # print('', end='\r')
                    # print("[featureExtract][Batch][End of List]")  
                    break
            print('', end='\r')
            out = self.Encoder.predict(batchImg)
            for n in range(len(out)):
                featureList.append(out[n])
        
        if(pkl_save_path!=None):
            with open(pkl_save_path, 'wb') as file:
                pickle.dump(pkl_save_path, file)
                
        sys.stdout.write("]\n")
        featureList = np.asarray(featureList)
        return featureList
    
    def match_1to1_batch(self,template_search,template_ref,save_pic=False):
        f = np.asarray(template_search)
        scores = []
        for n in range(len(template_search)):
            input_f = f[n]
           
            batch_feature = np.zeros((len(template_ref),4096))
            for j in range(len(template_ref)):
                match_feature = np.concatenate((input_f,template_ref[j]), axis=0)
                batch_feature[j] = match_feature
                
            score = self.Discriminator.predict(batch_feature)  
            score = score.tolist()
            scores.append(score)
            print('', end='\r')
            progressBar(n+1,len(template_search),20)
            
        return scores
    
    def match_1to1(self,template1,template2):
        score = np.dot(template1, template2)
        return score
    
    def enroll(self):
        self.searchDB=np.transpose(np.array(self.database))
        
    def enroll_to_DB(self,template,_id):
        self.database.append(template)
        self.ID.append(_id)
        
    def clear_model(self):
        del self.Encoder
        self.Discriminator=self.createDiscriminator()
        self.Discriminator.load_weights(r'../DIPdiscriminatorWeightsV5.h5')
        
    def retriev(self,template,num_result):
        input_f = template
        batch_feature = np.zeros((len(self.database),4096))
        for i in range(len(self.database)):
            match_feature = np.concatenate((input_f,self.database[i]), axis=0)
            batch_feature[i] = match_feature
            
        score = self.Discriminator.predict(batch_feature)  
        score = score.tolist()
        pd_dict = {
            'id':self.ID,
            'score':score
            }
        df = pd.DataFrame.from_dict(pd_dict)
        df_sort = df.sort_values(by="score",ascending=False)
        df_sort = df_sort.reset_index(drop=True)
        topScore = df_sort['score'].tolist()
        topID = df_sort['id'].tolist()
        
        return (topID,topScore)
    
    def retrievDot(self,template,num_result):
        score = np.matmul(template,self.searchDB)
        score = score.tolist()
        pd_dict = {
            'id':self.ID,
            'score':score
            }
        df = pd.DataFrame.from_dict(pd_dict)
        df_sort = df.sort_values(by="score",ascending=False)
        df_sort = df_sort.reset_index(drop=True)
        topScore = df_sort['score'].tolist()
        topID = df_sort['id'].tolist()
        
        return (topID,topScore)

