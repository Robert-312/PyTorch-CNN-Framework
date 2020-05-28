import numpy as np
from collections import defaultdict


import os, os.path
from PIL import Image

import torch.utils.data

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage

class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.Enlarged_Cardiomediastinum = []
        self.Cardiomegaly = []
        self.Lung_Opacity = []
        self.Lung_Lesion = []
        self.Edema = []
        self.Consolidation = []
        self.Pneumonia = []
        self.Atelectasis = []
        self.Pneumothorax = []
        self.Pleural_Effusion = []
        self.Pleural_Other = []
        self.Fracture = []
        
        self.transform = transform
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

        # Load the image and lables for each row in the DataFrame
        for _, row in df.iterrows():
            self.data.append(row.Image_Path)
            self.Enlarged_Cardiomediastinum.append(row['Enlarged_Cardiomediastinum'])
            self.Cardiomegaly.append(row['Cardiomegaly'])
            self.Lung_Opacity.append(row['Lung_Opacity'])
            self.Lung_Lesion.append(row['Lung_Lesion'])
            self.Edema.append(row['Edema'])
            self.Consolidation.append(row['Consolidation'])
            self.Pneumonia.append(row['Pneumonia'])
            self.Atelectasis.append(row['Atelectasis'])
            self.Pneumothorax.append(row['Pneumothorax'])
            self.Pleural_Effusion.append(row['Pleural_Effusion'])
            self.Pleural_Other.append(row['Pleural_Other'])
            self.Fracture.append(row['Fracture'])
            
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]

        # read image
        img = Image.open(img_path)

        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)

        # return the image and all the associated labels
        result = {
            'img': img,
            'labels': {
                        'Enlarged_Cardiomediastinum': self.Enlarged_Cardiomediastinum[idx],
                        'Cardiomegaly': self.Cardiomegaly[idx],
                        'Lung_Opacity': self.Lung_Opacity[idx],
                        'Lung_Lesion': self.Lung_Lesion[idx],
                        'Edema': self.Edema[idx],
                        'Consolidation': self.Consolidation[idx],
                        'Pneumonia': self.Pneumonia[idx],
                        'Atelectasis': self.Atelectasis[idx],
                        'Pneumothorax': self.Pneumothorax[idx],
                        'Pleural_Effusion': self.Pleural_Effusion[idx],
                        'Pleural_Other': self.Pleural_Other[idx],
                        'Fracture': self.Fracture[idx]
                    }
        }
        return result            