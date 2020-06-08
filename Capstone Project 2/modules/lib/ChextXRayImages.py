import numpy as np
import pandas as pd
from collections import defaultdict

import os, os.path
from os.path import dirname, basename, isfile, join
import glob
from PIL import Image
from skimage import io

import torch.utils.data

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage

class CleanMetaData():
    
    def __init__(self): 
        self.df = None
        self.df_reduced = None
        self.df_clean = None
        
        self.data_path = 'data/'
        self.raw_path = None
        self.train_image_path = None
        self.raw_train_csv = None
        self.intermediate_path = None
        self.intermediate_train_csv = None
        self.getPaths()
        
        self.feature_columns = ['PatientID',
                                'StudyID',
                                'Age',
                                'Sex_Male',
                                'Sex_Unknown',
                                'Orientation_PA',
                                'Support Devices']
        self.meta_columns = ['Image_Path']
        self.target_columns = [ 'Enlarged_Cardiomediastinum',
                                'Cardiomegaly',
                                'Lung_Opacity',
                                'Lung_Lesion',
                                'Edema',
                                'Consolidation',
                                'Pneumonia',
                                'Atelectasis',
                                'Pneumothorax',
                                'Pleural_Effusion',
                                'Pleural_Other',
                                'Fracture'
                               ]
        
    def getPaths(self):
        
        self.raw_path = self.data_path + "raw/"
        self.train_image_path = self.raw_path + "train/"
        self.train_csv = self.raw_path + "train.csv"

        self.intermediate_path = self.data_path + "intermediate/"
        self.intermediate_train_csv =  self.intermediate_path + "CleanMetaData.csv"
        
    def getDF(self):
        # load CheXpert Image metadata
        # Though CheXpert gave use a validation set with it's own metadata, there were too few images to be useful
        # So we are only going to use the training images and split this into train and val/test

        df = pd.read_csv(self.train_csv)

        # Rename columns to be more Python friendly
        df = df.rename(columns={'Frontal/Lateral': 'View', 
                                'AP/PA': 'Orientation',
                                'No Finding':'No_Finding',
                                'Enlarged Cardiomediastinum':'Enlarged_Cardiomediastinum',
                                'Lung Opacity':'Lung_Opacity',
                                'Lung Lesion':'Lung_Lesion',
                                'Pleural Effusion':'Pleural_Effusion',
                                'Pleural Other':'Pleural_Other',
                                'Support Device':'Support_Devices'})

        # Make sure the path prefix is consistent before apply parsing logic
        path_prefix = 'CheXpert-v1.0-small/train/'
        path_prefix_replace = path_prefix + '/data'
        patient_prefix = path_prefix + 'patient'
        paths_with_different_prefix = df[~df.Path.str.contains(patient_prefix)].shape[0]
        if paths_with_different_prefix > 0:
          raise Exception("Path prefix is not consistent, please update parsing logic") 

        # Parse path to get PatientID
        path_start_index_patient = len(patient_prefix)
        path_end_index_patient = path_start_index_patient + 5 # Patient ID is fixed width of size 5
        df['PatientID'] = df.Path.str.slice(path_start_index_patient,path_end_index_patient).astype(int)

        # Parse path to get StudyID (Count of x-ray sessions for a single patient)
        path_start_index_study = path_end_index_patient + 1 # Start off with end of PartientID + '/'
        path_end_index_patient = path_start_index_study + len('studyxx') # Here we assume the studyID will only be 2 digits
        df['StudyID'] = df.Path.str.slice(path_start_index_study,path_end_index_patient) \
                                            .str.replace('study', '') \
                                            .str.replace('/', '').astype(int)

        # Add image path with proper relative URI
        df['Image_Path'] = df.Path.str.replace(path_prefix, self.imageFolderPath())
        
        df.index.names = ['ImageID']
        
        self.df = df
        
    def reduceDF(self):
        self.df_reduced = self.df[
                (self.df.View == 'Frontal') & # Don't show x-rays from the side
                (self.df.StudyID < 5) & # Don't include more than 4 studies for a single patient
                ((self.df.Orientation == 'AP') | (self.df.Orientation == 'PA')) # Don't show Left or Right Lateral
                ]
     
    def cleanDF(self):
        # The only columns with null values are the targets
        # -1 = negative finding (the patient does NOT have this finding)
        #  1 = positive finding (the patient HAS this finding)
        #  0 = no position on the finding
        # So it should be safe to change all the nan in the df to 0 (no position)
        df_clean = self.df_reduced.fillna(0, inplace=False)

        # We already extracted all the information we needed from path and view, so we can drop these columns
        df_clean.drop(['Path', 'View'], axis=1, inplace=True)

        # This leaves us with just 2 categorical columns
        # So lest hot-one enode these
        df_clean = pd.get_dummies(df_clean, columns=['Sex', 'Orientation'], drop_first=True)
        # Sex: Not Male or Unknown = Female
        # Orientation: Not PA = AP

        # Let's clean up the column order with features first
        df_clean = df_clean[self.feature_columns + self.meta_columns + self.target_columns]

        for c in self.target_columns:
            df_clean[c] = df_clean[c].astype('int8')

        self.df_clean = df_clean
            
    def imageFolderPath(self):
        return self.train_image_path
            
    def intermediateFilePath(self):
        return self.intermediate_train_csv
            
    def intermediateFileExists(self):
        return glob.glob(self.intermediateFilePath())
    
    def saveCleanDF(self):
        self.df_clean.to_csv(self.intermediateFilePath())
            
    def getCleanDF(self, n_random_rows = 0):
        if self.df_clean is None:
            if not self.intermediateFileExists():
                self.getDF()
                self.reduceDF()
                self.cleanDF()
                self.saveCleanDF()

            self.df_clean = pd.read_csv(self.intermediateFilePath(), index_col=0)
        
        
        if n_random_rows == 0:
            return self.df_clean
        else:
            return self.df_clean.sample(n=n_random_rows, random_state = 42) 
    
    def displayImage(self, idx):
        return io.imread(os.path.join(os.getcwd(), self.getCleanDF().iloc[idx].Image_Path), as_gray=False)

class Dataset(torch.utils.data.Dataset):
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
        img_path = os.path.join(os.getcwd(), self.data[idx])

        # read image
        img = Image.open(img_path, 'r')

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

class Loaders():
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
        md = CleanMetaData()
        self.DataFrame = md.getCleanDF(n_random_rows)
            
    def getDataSet(self, n_random_rows = 0):
        if not self.Dataset:
            transform = self.getTransformations()
            if not self.DataFrame:
                self.setDataFrame(n_random_rows)
            self.Dataset = Dataset(self.DataFrame, transform)
        
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
        
        