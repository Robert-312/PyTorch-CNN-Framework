import numpy as np
from collections import defaultdict


import os, os.path
from PIL import Image

import torch.utils.data

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage

class DatasetAndLoader():
    def __init__(self):
        exec(open("CheXpertCleanMetaData.py").read())
        self.cheXpertCleanMetaData = CheXpertCleanMetaData()
        self.df = cheXpertCleanMetaData.getCleanDF()
               
        self.image_width = 320
        self.image_height = 320
        self.pixel_mean = 0.5064167
        self.pixel_sd = 0.16673872
        
        self.transform = None
        self.Dataset = None
        self.DataLoader = None
        
    def getTransformations(self):
        if not self.transform:
            self.transform = transforms.Compose(
                                  [transforms.Resize(size=(self.image_height,self.image_width), interpolation=2),
                                  transforms.Grayscale(1),
                                  transforms.ToTensor(),
                                  transforms.Normalize((self.pixel_mean,), (self.pixel_sd,))])
        return self.transform
    
    def getDataSet(self):
        if not self.Dataset:
            exec(open("ChestXRayDataset.py").read())
            transform = self.getTransformations()
            self.Dataset = ChestXRayDataset(df, transform)
        
        return self.Dataset
    
    def getDataLoader(self, batch_size=64):
        if not self.DataLoader:
            ds = self.getDataSet()
            self.DataLoader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
            
        return self.DataLoader 
        
        