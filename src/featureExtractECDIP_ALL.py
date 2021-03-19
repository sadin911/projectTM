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
from tensorflow.python.keras.layers import Input, Dense, Activation , LeakyReLU, Flatten, BatchNormalization, Dropout,LayerNormalization
from tensorflow.keras.applications import MobileNet,InceptionV3
from CMC import CMC

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
path_input = r'D:/project/projectTM/src/ImagesALLDB/DIP/N/**/'
path_ref = r'D:/project/projectTM/src/ImagesALLDB/DIP/R/**/'
types = ('*.bmp', '*.jpg' ,'.*gif' ,'*.png' , '*.tif','*.jpeg')
df = pd.read_csv(r'D:/datasets/LSLOGO/List/test_images_root.txt', delimiter = "\t",header=None)
path2K = df[0].tolist()
pathlist = []
pathref = []
dirpath = 'outputDIPEnV5'
if os.path.exists(dirpath):
    shutil.rmtree(dirpath)

path2K = [r'D:/datasets/LSLOGO/Logo-2K+/' + x for x in path2K]

for files in types:
    pathlist.extend(glob2.glob(os.path.join(path_input, files)))
    pathref.extend(glob2.glob(os.path.join(path_ref, files)))
    
pathref.extend(path2K)

def createEncoder():
    base_model=InceptionV3(input_shape=(224,224,3),weights=None,include_top=False) 
    x = base_model.output
    x = Flatten()(x)
    x = Dense(2048,activation='sigmoid')(x)
    model = Model(inputs=base_model.input,outputs=x)
    return model
    
def createDiscriminator():
    input1 = Input(shape=4096)
    x = Dense(1024,activation='relu')(input1)
    x = Dense(512,activation='relu')(x)
    x = Dense(256,activation='relu')(x)
    target = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=input1,outputs=target)
    return model

model = createEncoder()
model_discriminator = createDiscriminator()

model_discriminator.load_weights(r'DIPdiscriminatorWeightsV5.h5')
model.load_weights(r'DIPencoderWeightsV5.h5')

feature = []
feature_ref = []
IterNum_N = 7
batchsize_N = len(pathlist)//IterNum_N

real_index = 0

for iteration in range(IterNum_N):
    batchImg = np.zeros((batchsize_N,224,224,3))
    for batch_index in range(batchsize_N):
        fp = pathlist[real_index]
        img_pil = Image.open(fp).convert('RGB')
        img_pil = img_pil.resize((224,224))
        img_cv = np.asarray(img_pil)
        batchImg[batch_index] = img_cv/127.5-1   
        real_index = real_index + 1
        print(f'{real_index}//{len(pathlist)}')
        
        
    out = model.predict(batchImg)
    for n in range(len(out)):
        feature.append(out[n])
    

IterNum_R = 10
batchsize_R = len(pathref)//IterNum_R
batchImg = np.zeros((batchsize_R,224,224,3))
real_index = 0

for iteration in range(IterNum_R):
    batchImg = np.zeros((batchsize_R,224,224,3))
    for batch_index in range(batchsize_R):
        fp = pathref[real_index]
        img_pil = Image.open(fp).convert('RGB')
        img_pil = img_pil.resize((224,224))
        img_cv = np.asarray(img_pil)
        batchImg[batch_index] = img_cv/127.5-1  
        real_index = real_index + 1
        print(f'{real_index}//{len(pathref)}')
    out = model.predict(batchImg)
    for n in range(len(out)):
        feature_ref.append(out[n])

Path(r"models/DIPEnV5All/").mkdir(parents=True, exist_ok=True)

pkl_filename = r"models/DIPEnV5All/scbDIP_no_EnV5.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(feature, file)
    
pkl_filename = r"models/DIPEnV5All/scbDIP_no_EnV5ref.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(feature_ref, file)
    
pkl_filename = r"models/DIPEnV5All/pathlistDIP_no_EnV5.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(pathlist, file)
    
pkl_filename = r"models/DIPEnV5All/pathlistDIP_no_EnV5ref.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(pathref, file)
    
features = pickle.load(open(r'models/DIPEnV5All/scbDIP_no_EnV5.pkl', 'rb'))
features_ref = pickle.load(open(r'models/DIPEnV5All/scbDIP_no_EnV5ref.pkl', 'rb'))
pathlist = pickle.load(open(r'models/DIPEnV5All/pathlistDIP_no_EnV5.pkl', 'rb'))
pathref = pickle.load(open(r'models/DIPEnV5All/pathlistDIP_no_EnV5ref.pkl', 'rb'))


f = np.asarray(features)
f_ref = np.asarray(features_ref)
Total_AP = 0
Rank_CMC = np.zeros(len(features_ref))
for n in range(len(features)):
    # print(n)
    input_f = f[n]
    # pair = np.transpose(f)
    # score = np.matmul(input_f,pair)
    score = []
    batch_feature = np.zeros((len(features_ref),4096))
    for j in range(len(features_ref)):
        match_feature = np.concatenate((features_ref[j],input_f), axis=0)
        # match_feature = np.expand_dims(match_feature,axis=0)
        batch_feature[j] = match_feature
        # score.append(dist)
    score = model_discriminator.predict(batch_feature)  
    score = score.tolist()
    pd_dict = {
        'path':pathref[0:len(features_ref)],
        'score':score
        }
    
    df = pd.DataFrame.from_dict(pd_dict)
    paths = pathlist
    df_sort = df.sort_values(by="score",ascending=False)
    df_sort = df_sort.reset_index(drop=True)
    pathsTop = df_sort.path.tolist()
    scoreTop = df_sort.score.tolist()
    in_name = os.path.basename(paths[n])
    in_name = in_name.split('.')[0]
    Path(f"outputDIPEnV5/{in_name}").mkdir(parents=True, exist_ok=True)
    P_total = 0
    index_T = 1
    list_find = []
    for k in range(len(pathsTop)):
        path = pathsTop[k].split('_')[0].split('\\')[-1]
        if(path==in_name.split('_')[0]):
            P_total = P_total + float(index_T)/float(k+1)
            Rank_CMC[k] += 1
            list_find.append(k)
            index_T += 1
            
            
    ngenuine = index_T - 1
    AP = float(P_total) / float(ngenuine)
    print({f"index={in_name.split('_')[0]} AP={AP} Rank={list_find}"})
    Total_AP = Total_AP + AP
    for i in range(50):
        img = Image.open(pathsTop[i]).convert('RGB')
        img_input = Image.open(paths[n]).convert('RGB')
        filename = os.path.basename(pathsTop[i])
        sc = scoreTop[i]
        img.save(f"outputDIPEnV5/{in_name}/{i}_{sc}.jpg")
        img_input.save(f"outputDIPEnV5/{in_name}/00.jpg")
        
MAPAP = Total_AP/len(features_ref)
print(MAPAP)

sum_CMC = np.zeros(len(features))
for i in range(len(sum_CMC)):
    temp = Rank_CMC[:i]
    sum_CMC[i] = temp.sum()/len(features)

cmc_dict = {
    'InceptionV3ShareEC (rank50)': sum_CMC
}

cmc = CMC(cmc_dict)
cmc.plot(title = 'CMC on Search Rank\n', rank=100,
         xlabel='Rank',
         ylabel='Hit Rate', show_grid=False)

pkl_filename = r"models/DIPEnV5All/sum_CMC.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(sum_CMC, file)
    
pkl_filename = r"models/DIPEnV5All/CMC_Rank.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(Rank_CMC, file)
    
pkl_filename = r"models/DIPEnV5All/MAPAP.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(MAPAP, file)