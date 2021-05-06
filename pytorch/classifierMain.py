# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:07:45 2021

@author: chonlatid.d
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import os
import glob2
from PIL import Image,ImageDraw,ImageFilter,ImageEnhance,ImageOps,ImageChops
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import time
import copy
""" environment inittial """
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:0" if cuda else "cpu")

""" Augmented Param """
pad_param = 5
rotate_degree_param = 5
input_shape = (224,224)

""" Inputs Initialize """
classes = pd.read_csv(r'../LSLOGO/List/Logo-2K+classes.txt', delimiter = "\t",header=None)
classlist = classes[0].tolist()
numclass = len(classlist)
classDict = {
    'class':classlist,
    'index': [*range(numclass)]
}
classdf = pd.DataFrame.from_dict(classDict)

""" DataLoader """
class ImageDataset(Dataset):
    def __init__(self, root, transforms_= None, mode = 'samepic'):
        self.transform = transforms.Compose(transforms_)
        df = pd.read_csv(r'../LSLOGO/List/test_images_root.txt', delimiter = "\t",header=None)
        path2K = df[0].tolist()
        path2K = [r'../LSLOGO/Logo-2K+/' + x for x in path2K]
        self.files = path2K
        
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_A = self.transform(gen_data(img))
        classT = self.files[index % len(self.files)]
        classT = classT.split('/')[-2]
        targetT = classdf[classdf['class']==classT]
        targetI = targetT['index'].values[0]
        
        
        return {"img": img_A , "label" : targetI}
    
    def __len__(self):
        return len(self.files)



transformations = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


train_loader = DataLoader(
    ImageDataset(r"D:/project/projectTM/LSLOGO/train_and_test/train", transforms_=transformations),
    batch_size=5,
    shuffle=True,
    num_workers=0,
)


""" Util Function """
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def gen_data(img_pil):
    
    pad_top = int(abs(np.random.uniform(0,pad_param)))
    pad_bottom = int(abs(np.random.uniform(0,pad_param)))
    pad_left = int(abs(np.random.uniform(0,pad_param)))
    pad_right = int(abs(np.random.uniform(0,pad_param)))
    rotate_param = np.random.uniform(0,rotate_degree_param)
    blur_rad = np.random.normal(loc=0.0, scale=2, size=None)
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(blur_rad))
    enhancer_contrat = ImageEnhance.Contrast(img_pil)
    enhancer_brightness = ImageEnhance.Brightness(img_pil)
    enhancer_color = ImageEnhance.Color(img_pil)
    brightness_factor = np.random.normal(loc=1.0, scale=2.5, size=None)
    contrast_factor = np.random.normal(loc=1.0, scale=2.5, size=None)

    translate_factor_hor = np.random.normal(loc=0, scale=5, size=None)
    translate_factor_ver = np.random.normal(loc=0, scale=5, size=None)
    
    img_pil = enhancer_contrat.enhance(contrast_factor)
    img_pil = enhancer_brightness.enhance(brightness_factor)
    img_pil = enhancer_color.enhance(0)
    img_pil = ImageChops.offset(img_pil, int(translate_factor_hor), int(translate_factor_ver))
    img_pil = img_pil.rotate(rotate_param,resample = Image.BILINEAR,expand = True, fillcolor = (255,255,255))
    
    img = np.asarray(img_pil)
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=(255,255,255))
    img= cv2.resize(img, dsize=input_shape)

    return img

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            inputs = Variable(batch["img"].type(Tensor))
            labels = Variable(batch["label"].type(Tensor))

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {classlist[preds[j]]} / {classlist[labels[j].int()]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


""" Model Build """

model_conv = torchvision.models.resnet18(pretrained=False)
# for param in model_conv.parameters():
#     param.requires_grad = False
    
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(classlist))
model_conv = model_conv.to(device)
print(model_conv)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model_conv = torchvision.models.resnet18(pretrained=False)
        self.model_conv.fc = torchvision.models.resnet18(pretrained=False)

    def forward(self, x):
        x = self.model_conv(x)
        x = self.model_conv.fc(x)
        return x

# net = Net()
# net.to(device)
# print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_conv.parameters(), lr=0.001)
imgs = next(iter(train_loader))
inputs = Variable(imgs["img"].cpu())
labels = Variable(imgs["label"].cpu())
                  
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[classlist[x] for x in labels])

""" training """
for epoch in range(2000):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, batch in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = Variable(batch["img"].type(Tensor))
        labels = Variable(batch["label"].type(Tensor))

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model_conv(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            visualize_model(model_conv)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
