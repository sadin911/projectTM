# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:44:16 2021

@author: chonlatid.d
"""

import glob2
from os.path import join
import numpy as np
import os
import shutil
from PIL import Image
path_input = r'C:\\Users\chonlatid.d\\Downloads\\DIP_PIC_DATA\\**\\'
types = ('*.bmp', '*.jpg' ,'.*gif' ,'*.png' , '*.tif')
pathlist = []
for files in types:
    pathlist.extend(glob2.glob(join(path_input, files)))
    
for path in pathlist:
    img = Image.open(path)
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)
    img.save(r'images/{filename}.png')