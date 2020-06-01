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
               
        self.image_width = 320
        self.image_height = 320
        self.pixel_mean = 0.5064167
        self.pixel_sd = 0.16673872
        
        self.transform = None
        self.DataFrame = None
        self.Dataset = None
        self.DataLoader = None
        self.DataTrainValidateLoaders = None
        
    def getTransformations(self):
        if not self.transform:
            self.transform = transforms.Compose(
                                  [transforms.Resize(size=(self.image_height,self.image_width), interpolation=2),
                                  transforms.Grayscale(1),
                                  transforms.ToTensor(),
                                  transforms.Normalize((self.pixel_mean,), (self.pixel_sd,))])
        return self.transform
    
    def setDataFrame(self, n_random_rows = 0):
        exec(open("CheXpertCleanMetaData.py").read())
        self.cheXpertCleanMetaData = CheXpertCleanMetaData()
        self.DataFrame = cheXpertCleanMetaData.getCleanDF(n_random_rows)
            
    def getDataSet(self, n_random_rows = 0):
        if not self.Dataset:
            exec(open("ChestXRayDataset.py").read())
            transform = self.getTransformations()
            if not self.DataFrame:
                self.setDataFrame(n_random_rows)
            self.Dataset = ChestXRayDataset(self.DataFrame, transform)
        
        return self.Dataset
    
    def getDataLoader(self, batch_size=64, n_random_rows = 0):
        if not self.DataLoader:
            ds = self.getDataSet(n_random_rows)
            self.DataLoader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
            
        return self.DataLoader
    
    def getDataTrainValidateLoaders(self, batch_size=64, train_percent = 0.8, n_random_rows = 0):
        if not self.DataTrainValidateLoaders:
            ds = self.getDataSet(n_random_rows)
            
            train_length = int(len(ds) * train_percent)
            val_length = int(len(ds) - train_length)
            train_dataset, val_dataset = torch.utils.data.random_split(ds, [train_length, val_length])
            
            train_load = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
            val_load = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
            self.DataTrainValidateLoaders = (train_load, val_load)
            
        return self.DataTrainValidateLoaders 
        
        