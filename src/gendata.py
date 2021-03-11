#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:03:52 2020

@author: chonlatid
"""
from PIL import Image,ImageDraw
import numpy as np
import cv2
import math

class gendata():
    def __init__(self):
        # self.sx = 1.2
        # self.sy = 1.2
        self.pad_param = 20
        self.rotate_degree_param = 5
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.input_shape = (self.img_rows, self.img_cols, self.channels)
    



    def scale_and_rotate_image(self,im, sx, sy, deg_ccw,fill):
        sx = 1
        sy = 1
        im_orig = im
        im = Image.new('RGB', im_orig.size, (0, 0, 0))
        im.paste(im_orig)
    
        w, h = im.size
        angle = math.radians(-deg_ccw)
    
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
    
        scaled_w, scaled_h = w * sx, h * sy
    
        new_w = int(math.ceil(math.fabs(cos_theta * scaled_w) + math.fabs(sin_theta * scaled_h)))
        new_h = int(math.ceil(math.fabs(sin_theta * scaled_w) + math.fabs(cos_theta * scaled_h)))
    
        cx = w / 2.
        cy = h / 2.
        tx = new_w / 2.
        ty = new_h / 2.
    
        a = cos_theta / sx
        b = sin_theta / sx
        c = cx - tx * a - ty * b
        d = -sin_theta / sy
        e = cos_theta / sy
        f = cy - tx * d - ty * e
    
        return im.transform(
            (new_w, new_h),
            Image.AFFINE,
            (a, b, c, d, e, f),
            resample=Image.BILINEAR,
            fillcolor = fill
        )
    
    def gen_data(self,input_path,isPerspective = False):
       
        img = Image.open(input_path).convert('RGB')
        img = np.asarray(img)

        pad_top = int(abs(np.random.uniform(0,self.pad_param)))
        pad_bottom = int(abs(np.random.uniform(0,self.pad_param)))
        pad_left = int(abs(np.random.uniform(0,self.pad_param)))
        pad_right = int(abs(np.random.uniform(0,self.pad_param)))
        # sx = int(abs(np.random.uniform(0.8,self.sx)))
        # sy = int(abs(np.random.uniform(0.8,self.sy)))
        rotate_param = np.random.uniform(0,self.rotate_degree_param)
        src_points = np.float32([[0,0], [img.shape[1],0], [0,img.shape[0]], [img.shape[1],img.shape[0]]])
        
        # dx0 = 0 + np.random.uniform(1,0.1*img.shape[1])
        # dy0 = 0 + np.random.uniform(1,0.1*img.shape[0])
        
        # dx1 = img.shape[0] - np.random.uniform(100,0.1*img.shape[1])
        # dy1 = 0 + np.random.uniform(1,0.1*img.shape[1])
        
        # dx2 = 0 + np.random.uniform(1,0.1*img.shape[0])
        # dy2 = img.shape[1] - np.random.uniform(100,0.1*img.shape[1])
        
        # dx3 = img.shape[0] - np.random.uniform(1,0.1*img.shape[0])
        # dy3 = img.shape[1] - np.random.uniform(1,0.1*img.shape[1])
        
        dx0 = 0 + np.random.uniform(0,0.2*img.shape[0])
        dy0 = 0 + np.random.uniform(0,0.2*img.shape[1])
        
        dx1 = 0 + np.random.uniform(0,0.2*img.shape[0])
        dy1 = img.shape[1] - np.random.uniform(0,0.2*img.shape[1])
        
        dx2 = img.shape[0] - np.random.uniform(0,0.1*img.shape[0])
        dy2 = 0 + np.random.uniform(0,0.1*img.shape[1])
        
        dx3 = img.shape[0] - np.random.uniform(0,0.1*img.shape[0])
        dy3 = img.shape[1] - np.random.uniform(0,0.1*img.shape[1])
        
        dst_points = np.float32([[dy0,dx0],[dy1,dx1],[dy2,dx2],[dy3,dx3]]) 
        projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        if(isPerspective):
            img = cv2.warpPerspective(np.float32(Image.fromarray(img.astype('uint8'))), projective_matrix, (img.shape[1],img.shape[0]),cv2.BORDER_CONSTANT,borderValue=(255,255,255))
        ret_img = Image.fromarray(img.astype('uint8')).rotate(rotate_param,resample = Image.BILINEAR,expand = True, fillcolor = (255,255,255))
        ret_img = cv2.copyMakeBorder( np.asarray(ret_img), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
        ret_img= cv2.resize(ret_img, dsize=(self.input_shape[0], self.input_shape[1]))
        
        
        ####compute mask
        mask_img = np.zeros((img.shape[0],img.shape[1],3))
        if(isPerspective):
            mask_img = cv2.warpPerspective(np.float32(Image.fromarray(mask_img.astype('uint8'))), projective_matrix, (img.shape[1],img.shape[0]),cv2.BORDER_CONSTANT,borderValue=(255,255,255))
        mask_2 = Image.fromarray(mask_img.astype('uint8')).rotate(rotate_param,resample = Image.NEAREST,expand = True, fillcolor = (255,255,255))
        mask_2 = cv2.copyMakeBorder( np.asarray(mask_2), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
        
      
            
       
        mask_3 = cv2.resize(mask_2, dsize=(self.input_shape[0], self.input_shape[1]))
        mask_3 = mask_3/255
        mask_out = 1-mask_3
        mask_out = np.array(mask_out, dtype=np.uint8)
        mask_out = cv2.cvtColor(mask_out,cv2.COLOR_BGR2GRAY)
        ret_img = (ret_img/127.5)-1
        
        
        loc_topleft_img = np.zeros((img.shape[0],img.shape[1],3))
        loc_topleft_img[0][0]=255
        
        loc_bottomleft_img = np.zeros((img.shape[0],img.shape[1],3))
        loc_bottomleft_img[-1][0]=255
        
        loc_topright_img = np.zeros((img.shape[0],img.shape[1],3))
        loc_topright_img[0][-1]=255
        
        loc_bottomright_img = np.zeros((img.shape[0],img.shape[1],3))
        loc_bottomright_img[-1][-1]=255
        
        
        array_image = [loc_topleft_img,loc_bottomleft_img,loc_topright_img,loc_bottomright_img ]
        
        
        for i in range(len(array_image)):
                
            ## rotate
            if(isPerspective): 
                array_image[i] = cv2.warpPerspective(np.float32(Image.fromarray(array_image[i].astype('uint8'))), projective_matrix, (img.shape[1],img.shape[0]),cv2.BORDER_CONSTANT,borderValue=(0,0,0))
            img2 = Image.fromarray(array_image[i].astype('uint8')).rotate(rotate_param,resample = Image.BILINEAR,expand = True, fillcolor = (0,0,0))
            
            # pad border
            img3 = cv2.copyMakeBorder( np.asarray(img2), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(0,0,0))
            array_image[i] = img3
        
        target = cv2.resize(img/127.5-1, dsize=(self.input_shape[0], self.input_shape[1]))
        return ret_img, target, mask_out
    
    def plot_corner(self,img,target,color = (255,0,0)):
        ret_img = (img+1)*127.5
        
        ret_img = Image.fromarray(ret_img.astype('uint8'))
        draw = ImageDraw.Draw(ret_img)
        draw.polygon([(target[0],target[1]),(target[2],target[3]),(target[4],target[5]),(target[6],target[7]) ],  outline=color)
        draw.ellipse((target[0]-5, target[1]-5, target[0]+5, target[1]+5), fill = 'blue', outline ='blue')
        draw.ellipse((target[4]-5, target[5]-5, target[4]+5, target[5]+5), fill = 'green', outline ='green')
        return ret_img
    
    def plot_corner_seg(self,img,target,color = (255,0,0)):
        ret_img = (img+1)*127.5
        ret_img = cv2.cvtColor(ret_img,cv2.COLOR_GRAY2RGB)
        ret_img = Image.fromarray(ret_img.astype('uint8'))
        draw = ImageDraw.Draw(ret_img)
        draw.polygon([(target[0],target[1]),(target[2],target[3]),(target[4],target[5]),(target[6],target[7]) ],  outline=color)
        draw.ellipse((target[0]-5, target[1]-5, target[0]+5, target[1]+5), fill = 'blue', outline ='blue')
        draw.ellipse((target[4]-5, target[5]-5, target[4]+5, target[5]+5), fill = 'green', outline ='green')
        return ret_img