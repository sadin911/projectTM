# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 17:14:38 2021

@author: chonlatid.d
"""
import glob2
from os.path import join
import numpy as np
import os
import shutil
from PIL import Image
import pandas as pd
from pathlib import Path
import requests
from io import BytesIO

df = pd.read_csv(r"C:\Users\chonlatid.d\Downloads\DIP_PIC_DATA\drive-download-20210311T064932Z-001\dip_tr_like.csv",sep=";")
df_file = pd.read_csv(r"C:\Users\chonlatid.d\Downloads\DIP_PIC_DATA\drive-download-20210311T064932Z-001\dip_tr_file.csv",sep=";")
df_drop = df[pd.notnull(df['COND_PIC'])]

TN_List = df_drop['TR_NO'].tolist()
TR_List = df_drop['TR_NO_R'].tolist()

print(len(np.unique(TN_List)))
print(len(np.unique(TR_List)))

def progressBar(current, total, barLength = 20):
    print('', end='\r')
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r',flush=True)
    
group_path = 'imageGroupALLV2'
sep_path = 'ImagesALLV2'
if os.path.exists(group_path):
    shutil.rmtree(group_path)
if os.path.exists(sep_path):
    shutil.rmtree(sep_path)
    
for i in range(len(TN_List)):
    try:
        progressBar(i, len(TN_List))
        dfRef = df_file[df_file['TR_NO']==TR_List[i]]
        pathR = dfRef['FILE_PATH'].tolist()[0]
        fileR = dfRef['FILE_NAME'].tolist()[0]
        # if(len(fileR)>1):
        #     print(fileR)
        response = requests.get(r'https://madrid.ipthailand.go.th/download/trademark?file=/image/TRS_SCAN/'+pathR+'/'+fileR)
        Refimg = Image.open(BytesIO(response.content))
        fileR = fileR.split('.')[0]
        
        dfN = df_file[df_file['TR_NO']==TN_List[i]]
        pathN = dfN['FILE_PATH'].tolist()[0]
        fileN = dfN['FILE_NAME'].tolist()[0]
        
        # if(len(fileN)>1):
        #    print(fileN)
        response = requests.get(r'https://madrid.ipthailand.go.th/download/trademark?file=/image/TRS_SCAN/'+pathN+'/'+fileN)
        imgN = Image.open(BytesIO(response.content))
        fileN = fileN.split('.')[0]
           
        Path(f"{group_path}/{fileN}").mkdir(parents=True, exist_ok=True)
        Path(f"{sep_path}/DIP/Search").mkdir(parents=True, exist_ok=True)
        Path(f"{sep_path}/DIP/Reference").mkdir(parents=True, exist_ok=True)
        
        fref1 = f"{group_path}/{fileN}/{fileR}.jpg"
        if not os.path.exists(fref1):
            Refimg.save(fref1)
            
        fref2 = f"{sep_path}/DIP/Reference/{fileR}.jpg"
        if not os.path.exists(fref2):
            Refimg.save(fref2)
        
        fn1=f"{group_path}/{fileN}/0.jpg"
        if not os.path.exists(fn1):
            imgN.save(fn1)
        
        fn2=f"{sep_path}/DIP/Search/{fileN}.jpg"
        if not os.path.exists(fn2):
            imgN.save(fn2)
    except:
        a = 1