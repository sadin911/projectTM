# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:17:12 2021

@author: chonlatid.d
"""

import glob2
from PIL import Image
import shutil
from pathlib import Path
import os
pathlist = []
path_input = r'D:/project/projectTM/src/imageGroupALLV2/**/*.jpg'
pathlist.extend(glob2.glob(path_input))

for file in pathlist:
    try:
        img = Image.open(file)
    except:
        path = Path(file)
        os.remove(path)
        print(path)
        
