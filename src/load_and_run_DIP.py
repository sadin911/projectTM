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
import shutil
features = pickle.load(open(r'scbDIP_no_inceptionV3test_N.pkl', 'rb'))
features_old = pickle.load(open(r'scbDIP_no_inceptionV3test.pkl', 'rb'))
pathlist = pickle.load(open(r'pathlistDIP_no_inceptionV3test_N.pkl', 'rb'))
pathlist_old = pickle.load(open(r'pathlistDIP_no_inceptionV3test.pkl', 'rb'))
f = np.asarray(features)
shutil.rmtree('outputDIPInceptionV3test')

def distance(pointA, pointB):
    dist = np.linalg.norm(pointA - pointB)
    return dist

for n in range(len(features)):
    print(n)
    input_f = f[n]
    # pair = np.transpose(f)
    # score = np.matmul(input_f,pair)
    score = []
    for j in range(len(features_old)):
        dist = distance(input_f, features_old[j])
        score.append(dist)
        
    pd_dict = {
        'path':pathlist_old,
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
    Path(f"outputDIPInceptionV3test/{in_name}").mkdir(parents=True, exist_ok=True)
    img = Image.open(pathlist[n]).convert('RGB')
    # img.save(f"outputDIPInceptionV3test/{in_name}/0_in_name.jpg")
    for i in range(50):
        img = Image.open(pathsTop[i]).convert('RGB')
        filename = os.path.basename(pathsTop[i])
        sc = scoreTop[i]
        img.save(f"outputDIPInceptionV3test/{in_name}/{i}_{sc}.jpg")