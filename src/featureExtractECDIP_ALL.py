import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.callbacks import TensorBoard
import glob2
import tensorflow as tf
import datetime
from PIL import Image,ImageDraw,ImageFilter,ImageEnhance,ImageOps,ImageChops
import numpy as np
import cv2
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


path_input = r'D:/project/projectTM/src/imageGroupALLV3/**/0*.jpg'
path_ref = r'D:/project/projectTM/src/imageGroupALLV3/**/[1-9]*.jpg'

types = ('*.jpg')
df = pd.read_csv(r'D:/datasets/LSLOGO/List/test_images_root.txt', delimiter = "\t",header=None)
path2K = df[0].tolist()
pathlist = []
pathref = []
dirpath = 'V2outputDIPEnV9'


if os.path.exists(dirpath):
    shutil.rmtree(dirpath)

path2K = [r'D:/datasets/LSLOGO/Logo-2K+/' + x for x in path2K]
pathlist.extend(glob2.glob(path_input))
pathref.extend(glob2.glob(path_ref))
    
pathref.extend(path2K)

def progressBar(current, total, barLength = 20):
    print('', end='\r')
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r',flush=True)

def createEncoder():
    base_model=InceptionV3(input_shape=(224,224,3),weights=None, include_top=False) 
    x = base_model.output
    x = Flatten()(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(2048,activation='sigmoid')(x)
    model = Model(inputs=base_model.input,outputs=x,name='Encoder')
    return model
    
def createDiscriminator():
    input1 = Input(shape=4096)
    x = Dense(1024,activation='relu')(input1)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.1)(x)
    x = Dense(512,activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.1)(x)
    x = Dense(224,activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.1)(x)
    target = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=input1,outputs=target,name='Discriminator')
    return model

model = createEncoder()
model_discriminator = createDiscriminator()

model_discriminator.load_weights(r'DCdiscriminator.Weights.h5')
model.load_weights(r'DCencoder.Weights.h5')

feature = []
feature_ref = []

IterNum_N = 10
batchsize_N = len(pathlist)//IterNum_N

real_index = 0
real_pathinput = []
real_pathref = []

for iteration in range(IterNum_N):
    batchImg = np.zeros((batchsize_N,224,224,3))
    for batch_index in range(batchsize_N):
        fp = pathlist[real_index]
        img_pil = Image.open(fp).convert('RGB')
        img_pil = img_pil.resize((224,224))
        img_cv = np.asarray(img_pil)
        batchImg[batch_index] = img_cv/127.5-1   
        real_pathinput.append(fp)
        real_index+=1
        print(f'{real_index}//{len(pathlist)}') 
    out = model.predict(batchImg)
    for n in range(len(out)):
        feature.append(out[n])
    

IterNum_R = 50
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
        real_pathref.append(fp)
        real_index+=1
        print(f'{real_index}//{len(pathref)}')
    out = model.predict(batchImg)
    for n in range(len(out)):
        feature_ref.append(out[n])

Path(r"models/DIPEnV9All/").mkdir(parents=True, exist_ok=True)

pkl_filename = r"models/DIPEnV9All/V2scbDIP_no_EnV9.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(feature, file)
    
pkl_filename = r"models/DIPEnV9All/V2scbDIP_no_EnV9ref.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(feature_ref, file)
    
pkl_filename = r"models/DIPEnV9All/V2pathlistDIP_no_EnV9.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(real_pathinput, file)
    
pkl_filename = r"models/DIPEnV9All/V2pathlistDIP_no_EnV9ref.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(real_pathref, file)
    
features = pickle.load(open(r'models/DIPEnV9All/V2scbDIP_no_EnV9.pkl', 'rb'))
features_ref = pickle.load(open(r'models/DIPEnV9All/V2scbDIP_no_EnV9ref.pkl', 'rb'))
pathlist = pickle.load(open(r'models/DIPEnV9All/V2pathlistDIP_no_EnV9.pkl', 'rb'))
pathref = pickle.load(open(r'models/DIPEnV9All/V2pathlistDIP_no_EnV9ref.pkl', 'rb'))


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
        match_feature = np.concatenate((input_f,features_ref[j]), axis=0)
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
    # in_name = os.path.basename(paths[n])
    # in_name = in_name.split('.')[0]
    in_name = Path(paths[n])._cparts[-2]
    P_total = 0
    index_T = 1
    list_find = []
    for k in range(len(pathsTop)):
        path = Path(pathsTop[k])._cparts[-2]
        if(path==in_name):
            P_total = P_total + float(index_T)/float(k+1)
            Rank_CMC[k] += 1
            list_find.append(k)
            index_T += 1
            
            
    ngenuine = index_T
    AP = float(P_total) / float(ngenuine)
    print({f"index={in_name} AP={AP} Rank={list_find}"})
    # progressBar(n, len(features))
    Total_AP = Total_AP + AP
    Path(f"V2outputDIPEnV9/{in_name}_{list_find[0]}").mkdir(parents=True, exist_ok=True)
    img_input = Image.open(paths[n]).convert('RGB')
    img_input.save(f"V2outputDIPEnV9/{in_name}_{list_find[0]}/0000.jpg")
    for i in range(len(list_find)):
        img_input = Image.open(pathsTop[list_find[i]]).convert('RGB')
        img_input.save(f"V2outputDIPEnV9/{in_name}_{list_find[0]}/000_{list_find[i]}.jpg")
    for i in range(50):
        img = Image.open(pathsTop[i]).convert('RGB')
        filename = os.path.basename(pathsTop[i])
        sc = scoreTop[i]
        img.save(f"V2outputDIPEnV9/{in_name}_{list_find[0]}/{i}_{sc}.jpg")
    
        
        
        
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

pkl_filename = r"models/DIPEnV9All/V2sum_CMC.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(sum_CMC, file)
    
pkl_filename = r"models/DIPEnV9All/V2CMC_Rank.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(Rank_CMC, file)
    
pkl_filename = r"models/DIPEnV9All/V2MAPAP.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(MAPAP, file)