
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
path_input = r'D:/project/projectTM/src/ImagesALL/DIP/N/**/'
path_ref = r'D:/project/projectTM/src/ImagesALL/DIP/R/**/'
types = ('*.bmp', '*.jpg' ,'.*gif' ,'*.png' , '*.tif','*.jpeg')
pathlist = []
pathref = []
shutil.rmtree('outputDIPInceptionV3LAUG')


for files in types:
    pathlist.extend(glob2.glob(os.path.join(path_input, files)))
    pathref.extend(glob2.glob(os.path.join(path_ref, files)))
    
def distance(pointA, pointB):
    dist = np.linalg.norm(pointA - pointB)
    return dist

model_ori = load_model('2kInception.h5')
model_ori.summary()

model = Model(model_ori.input,model_ori.layers[-2].output)
model.summary()

features = []
feature_ref = []
path_num = len(pathlist)//1
for file in range(path_num):
    fp = pathlist[file]
    img_pil = Image.open(fp).convert('RGB')
    img_pil = img_pil.resize((224,224))
    img_cv = np.asarray(img_pil)
    indput_data = img_cv/127.5-1
    indput_data = np.expand_dims(indput_data, axis = 0)
    out = model.predict(indput_data)[0]
    normalized_v = out / np.sqrt(np.sum(out**2))
    normalized_v = out
    filename = pathlist[file].split('/')
    str1 = "_"
    filename = str1.join(filename)
    print(f'{file}//{path_num}')
    features.append(normalized_v)

path_num = len(pathref)//1
for file in range(path_num):
    fp = pathref[file]
    img_pil = Image.open(fp).convert('RGB')
    img_pil = img_pil.resize((224,224))
    img_cv = np.asarray(img_pil)
    indput_data = img_cv/127.5-1
    indput_data = np.expand_dims(indput_data, axis = 0)
    out = model.predict(indput_data)[0]
    normalized_v = out / np.sqrt(np.sum(out**2))
    normalized_v = out
    filename = pathref[file].split('/')
    str1 = "_"
    filename = str1.join(filename)
    print(f'{file}//{path_num}')
    feature_ref.append(normalized_v)



pkl_filename = "scbDIP_no_inceptionV3.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(features, file)
    
pkl_filename = "scbDIP_no_inceptionV3ref.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(feature_ref, file)
    
pkl_filename = "pathlistDIP_no_inceptionV3.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(pathlist[0:path_num], file)
    
pkl_filename = "pathlistDIP_no_inceptionV3ref.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(pathref[0:path_num], file)
    
features = pickle.load(open(r'scbDIP_no_inceptionV3.pkl', 'rb'))
features_ref = pickle.load(open(r'scbDIP_no_inceptionV3ref.pkl', 'rb'))
pathlist = pickle.load(open(r'pathlistDIP_no_inceptionV3.pkl', 'rb'))
pathref = pickle.load(open(r'pathlistDIP_no_inceptionV3ref.pkl', 'rb'))


f = np.asarray(features)
f_ref = np.asarray(features_ref)
for n in range(len(features)):
    print(n)
    input_f = f[n]
    # pair = np.transpose(f)
    # score = np.matmul(input_f,pair)
    score = []
    for j in range(len(features_ref)):
        dist = distance(input_f, f_ref[j])
        score.append(dist)
        
    pd_dict = {
        'path':pathref,
        'score':score
        }
    
    df = pd.DataFrame.from_dict(pd_dict)
    paths = pathlist
    df_sort = df.sort_values(by="score",ascending=True)
    df_sort = df_sort.reset_index(drop=True)
    pathsTop = df_sort.path.tolist()
    scoreTop = df_sort.score.tolist()
    in_name = os.path.basename(paths[n])
    in_name = in_name.split('.')[0]
    Path(f"outputDIPInceptionV3/{in_name}").mkdir(parents=True, exist_ok=True)
    for i in range(50):
        img = Image.open(pathsTop[i]).convert('RGB')
        img_input = Image.open(paths[n]).convert('RGB')
        filename = os.path.basename(pathsTop[i])
        sc = scoreTop[i]
        img.save(f"outputDIPInceptionV3/{in_name}/{i}_{sc}.jpg")
        img_input.save(f"outputDIPInceptionV3/{in_name}/00.jpg")