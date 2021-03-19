# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from PIL import Image

class InceptionV3ENDIP:
    def __init__(self):
        print("[init] InceptionV3ENDIP")
        self.Encoder=self.createEncoder()
        self.Discriminator=self.createDiscriminator()
        self.database = []
        self.ID=[]
        self.searchDB = None

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
    
    def feature_extract(self,ImagePathList):
        print("[featureExtract] InceptionV3ENDIP")
        img_pil = Image.open(image_path).convert('RGB')
        img_pil = img_pil.resize((224,224))
        img_cv = np.asarray(img_pil)
        indput_data = tf.keras.applications.resnet.preprocess_input(img_cv)
        indput_data = np.expand_dims(indput_data, axis = 0)
        out = self.model.predict(indput_data)[0]
        return out
    
    def match_1to1(self,template1,template2):
        score = np.dot(template1, template2)
        return score
    
    def enroll(self):
        
        self.searchDB=np.transpose(np.array(self.database))
        
    def enroll_to_DB(self,template,_id):
        self.database.append(template)
        self.ID.append(_id)
        
        
    
    def retriev(self,template,num_result):
        if(self.searchDB==None):
            raise Exception("you must call enroll() first before you can retrieve")
        score = np.matmul(template,self.searchDB())
        l = zip(score, self.ID)
        l.sort()
        # 'unzip'
        score, _id = zip(*l)
        return (_id,score)

if __name__ == '__main__':
    print("main")
    cat = Restnet()