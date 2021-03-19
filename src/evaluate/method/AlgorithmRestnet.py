# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from PIL import Image
from sklearn.preprocessing import normalize

class Restnet:
    def __init__(self):
        print("init")
        base_model=keras.applications.ResNet152V2(input_shape=(224,224,3),weights='imagenet',include_top=False) 
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        self.model=Model(inputs=base_model.input,outputs=x)
        self.database = []
        self.ID=[]
        self.searchDB = None

    def feature_extract(self,image_path):

        img_pil = Image.open(image_path).convert('RGB')
        img_pil = img_pil.resize((224,224))
        img_cv = np.asarray(img_pil)
        indput_data = tf.keras.applications.resnet.preprocess_input(img_cv)
        indput_data = np.expand_dims(indput_data, axis = 0)
        out = self.model.predict(indput_data)[0]
        normalized_v = out / np.sqrt(np.sum(out**2))
        return normalized_v
        
    def feature_extract_batch(self,image_path_list,batch_size):
        image_pool = image_path_list
        result_list=[]
        batch_data=[]
        while(len(image_pool)>0):
            temp= image_pool.pop(0)
            img_pil = Image.open(temp).convert('RGB')
            img_pil = img_pil.resize((224,224))
            img_cv = np.asarray(img_pil)
            indput_data = tf.keras.applications.resnet.preprocess_input(img_cv)
            batch_data.append(indput_data)
            if(len(batch_data)==batch_size or len(image_pool)==0):
                print("batch processing")
                batch_result = self.model.predict(np.array(batch_data))
                batch_data=[]
                result_list.extend(batch_result)
        # indput_data = np.expand_dims(indput_data, axis = 0)
        result = normalize(np.array(result_list), norm="l2")

        return result
    
    def match_1to1(self,template1,template2):
        score = np.dot(template1, template2)
        return score
    
    def enroll(self):
        
        self.searchDB=np.transpose(np.array(self.database))
        
    def enroll_to_DB(self,template,_id):
        self.database.append(template)
        self.ID.append(_id)
        
        
    
    def retriev(self,template,num_result):
        # if(self.searchDB==None):
        #     raise Exception("you must call enroll() first before you can retrieve")
        score = np.matmul(template,self.searchDB)
        # score = np.random.rand(score.shape[0])
        l = zip(score, self.ID)
        l=sorted(l, key = lambda x:x[0],reverse=True)
        # 'unzip'
        score, _id = zip(*l)
        return (_id,score)

if __name__ == '__main__':
    print("main")
    cat = Restnet()
    feature = cat.feature_extract(r"D:\project\trademark\projectTM\src\imageGroupALL\2d_160115081\0.jpg")
    feature = cat.feature_extract_batch([r"D:\project\trademark\projectTM\src\imageGroupALL\2d_160115081\0.jpg" for x in range(100)], 30)
    # cat.match_1to1(feature, feature)
    # for i in range(1000):
    #     cat.enroll_to_DB(feature,str(i))
    # cat.enroll()
    # (_id,score) = cat.retriev(feature, 100)
