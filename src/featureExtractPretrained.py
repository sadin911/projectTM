
from tensorflow.keras.applications import MobileNet,InceptionV3,ResNet152V2
from tensorflow.keras.callbacks import TensorBoard
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
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import normalize
from pathlib import Path


path_input = r'images/*.png'
df = pd.read_csv(r'D:/datasets/LSLOGO/List/train_images_root.txt', delimiter = "\t",header=None)
pathlist = df[0].tolist()
pathlist = glob2.glob(path_input)
def createModel():
        base_model=ResNet152V2(input_shape=(224,224,3),weights='imagenet',include_top=False) 
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        model=Model(inputs=base_model.input,outputs=x)
        return model
    
def distance(pointA, pointB):
    dist = np.linalg.norm(pointA - pointB)
    return dist

model = createModel()
model.summary()
features = []
for file in pathlist:
    img_pil = Image.open(file).convert('RGB')
    img_pil = img_pil.resize((224,224))
    img_cv = np.asarray(img_pil)
    indput_data = tf.keras.applications.resnet.preprocess_input(img_cv)
    indput_data = np.expand_dims(indput_data, axis = 0)
    out = model.predict(indput_data)[0]
    # normalized_v = out / np.sqrt(np.sum(out**2))
    normalized_v = out
    filename = file.split('/')
    str1 = "_"
    filename = str1.join(filename)
    print(filename)
    features.append(normalized_v)


pkl_filename = "scb_no.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(features, file)
    
pkl_filename = "pathlist_no.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(pathlist, file)
    
features = pickle.load(open(r'scb_no.pkl', 'rb'))
pathlist = pickle.load(open(r'pathlist_no.pkl', 'rb'))


f = np.asarray(features)

for n in range(len(features)):
    print(n)
    input_f = f[n]
    pair = np.transpose(f)
    # score = np.matmul(input_f,pair)
    score = []
    for j in range(len(features)):
        dist = distance(input_f, f[j])
        score.append(dist)
        
    pd_dict = {
        'path':pathlist,
        'score':score
        }
    
    df = pd.DataFrame.from_dict(pd_dict)
    paths = df.path.tolist()
    df_sort = df.sort_values(by="score",ascending=True)
    df_sort = df_sort.reset_index(drop=True)
    pathsTop = df_sort.path.tolist()
    scoreTop = df_sort.score.tolist()
    in_name = os.path.basename(paths[n])
    Path(f"output/{in_name}").mkdir(parents=True, exist_ok=True)
    for i in range(20):
        img = Image.open(pathsTop[i]).convert('RGB')
        filename = os.path.basename(pathsTop[i])
        sc = scoreTop[i]
        img.save(f"output/{in_name}/{i}_{sc}.jpg")