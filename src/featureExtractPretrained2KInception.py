
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
from tensorflow.python.keras.models import Model, load_model
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
path_input = r'images/*.png'
df = pd.read_csv(r'D:/datasets/LSLOGO/List/test_images_root.txt', delimiter = "\t",header=None)
pathlist = df[0].tolist()
# pathlist = glob2.glob(path_input)
def createModel():
        base_model=ResNet152V2(input_shape=(224,224,3),weights='imagenet',include_top=False) 
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        model=Model(inputs=base_model.input,outputs=x)
        return model
    
def distance(pointA, pointB):
    dist = np.linalg.norm(pointA - pointB)
    return dist

model_ori = load_model('2kInception.h5')
model_ori.summary()

model = Model(model_ori.input,model_ori.layers[-2].output)
model.summary()

features = []
path_num = len(pathlist)//1
for file in range(path_num):
    fp = os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',pathlist[file])
    img_pil = Image.open(fp).convert('RGB')
    img_pil = img_pil.resize((224,224))
    img_cv = np.asarray(img_pil)
    indput_data = img_cv/127.5-1
    indput_data = np.expand_dims(indput_data, axis = 0)
    out = model.predict(indput_data)[0]
    # normalized_v = out / np.sqrt(np.sum(out**2))
    normalized_v = out
    filename = pathlist[file].split('/')
    str1 = "_"
    filename = str1.join(filename)
    print(f'{file}//{path_num}')
    features.append(normalized_v)


pkl_filename = "scb2K_no_inceptionV3test.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(features, file)
    
pkl_filename = "pathlist2K_no_inceptionV3test.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(pathlist[0:path_num], file)
    
features = pickle.load(open(r'scb2K_no_inceptionV3test.pkl', 'rb'))
pathlist = pickle.load(open(r'pathlist2K_no_inceptionV3test.pkl', 'rb'))


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
    # in_name = os.path.basename(paths[n])
    inpath =  paths[n].split('/')
    in_par = paths[n].split('/')[0]
    in_name = "_".join(paths[n].split('/')[-2])
    in_name = inpath[2].split(".")[0]
    Path(f"output2KInceptionV3test/{inpath[0]}/{inpath[1]}/{in_name}").mkdir(parents=True, exist_ok=True)
    for i in range(50):
        img = Image.open(os.path.join(r'D:\datasets\LSLOGO\Logo-2K+',pathsTop[i])).convert('RGB')
        filename = os.path.basename(pathsTop[i])
        sc = scoreTop[i]
        img.save(f"output2KInceptionV3test/{inpath[0]}/{inpath[1]}/{in_name}/{i}_{sc}.jpg")