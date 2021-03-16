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

df = pd.read_csv(r"C:\Users\chonlatid.d\Downloads\DIP_PIC_DATA\drive-download-20210311T064932Z-001\dip_tr_like.csv",sep=";")
df_file = pd.read_csv(r"C:\Users\chonlatid.d\Downloads\DIP_PIC_DATA\drive-download-20210311T064932Z-001\dip_tr_file.csv",sep=";")
df_drop = df[pd.notnull(df['COND_PIC'])]

TN_List = df_drop['TR_NO'].tolist()
TR_List = df_drop['TR_NO_R'].tolist()

print(len(np.unique(TN_List)))
print(len(np.unique(TR_List)))
shutil.rmtree('imageGroupALL')
shutil.rmtree('ImagesALL')
for i in range(len(TN_List)):
    try:
        dfRef = df_file[df_file['TR_NO']==TR_List[i]]
        pathR = dfRef['FILE_PATH'].tolist()[0]
        fileR = dfRef['FILE_NAME'].tolist()[0]
        if(len(fileR)>1):
            print(fileR)
        Refimg = Image.open(os.path.join(r'C:\Users\chonlatid.d\Downloads\DIP_PIC_DATA',pathR,fileR))
        fileR = fileR.split('.')[0]
        
        dfN = df_file[df_file['TR_NO']==TN_List[i]]
        pathN = dfN['FILE_PATH'].tolist()[0]
        fileN = dfN['FILE_NAME'].tolist()[0]
        
        if(len(fileN)>1):
           print(fileN)
        imgN = Image.open(os.path.join(r'C:\Users\chonlatid.d\Downloads\DIP_PIC_DATA',pathN,fileN))
        fileN = fileN.split('.')[0]
           
        Path(f"imageGroupALL/{fileN}").mkdir(parents=True, exist_ok=True)
        Path(f"ImagesALL/DIP/N").mkdir(parents=True, exist_ok=True)
        Path(f"ImagesALL/DIP/R").mkdir(parents=True, exist_ok=True)
        Refimg.save(f"imageGroupALL/{fileN}/{fileR}.jpg")
        Refimg.save(f"imagesALL/DIP/R/{fileR}.jpg")
        imgN.save(f"imageGroupALL/{fileN}/0.jpg")
        imgN.save(f"imagesALL/DIP/N/{fileN}.jpg")
    except:
        b=2