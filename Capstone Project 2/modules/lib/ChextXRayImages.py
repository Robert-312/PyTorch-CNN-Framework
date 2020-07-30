import numpy as np
import pandas as pd
from collections import defaultdict

from enum import Enum

import os, os.path
import warnings
from os.path import dirname, basename, isfile, join
import glob
import cv2
from PIL import Image
from skimage import io

import torch.utils.data

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage

class CleanMetaData():
    '''
    This class will load the csv file from CheXpert.
    The main call to this call will be the getCleanDF() function.
    This class will attemp to pull the clean DataFrame from memory first.  
    If it is not loaded, it will pull the clean df from a file.
    If this file does not exist, it will make build the clean df from scratch and safe the file.
    The goal is to be able to make multiple calls to this class and only do the work that is necessary.
    
    Subsetting and train/val splits are done in this DataFrame based class.  
    This is primarly done since transformations in PyTorch are easier to apply at the Dataset level.
    So creating different Datasets from different DataFrames was found to be the easiest approach.
    
    Hierarchal Path:  This is primarly benefits running this code in Google CoLab.
    Google Drive uses IDs for Directories.  
    Walking images directly on an OS such as Windows is very efficent since directory navagation is a primary function of the OS.
    But Google Drive looks like it has to do a ID lookup every time it is presented with a path.
    This lookup is very inefficent if there are thousands of items in the root path to lookup.
    So the Hierarchal Path column is a way of having only about 50 items in any folder.  
    See the Capstone Project 2/notebooks/Support Notebooks for Modules/HierarchicalPath.ipynb notebook for more details.
    
    Note:  If you change code in this class, you should delete the file or you will simply reload the old DataFrame.
    '''
    
    def __init__(self, binary_targets=True, target_columns=None): 
        
        # Leave target as is [-1, 0, 1] or force it to be Boolean [0,1]
        self.binary_targets = binary_targets
        
        # Init class variables
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
        
        self.random_state = 42
        
        # From EDA, we know we will have these columns in the DataFrame
        self.feature_columns = ['PatientID',
                                'StudyID',
                                'Age',
                                'Sex_Male',
                                'Sex_Unknown',
                                'Orientation_PA',
                                'Support Devices']
        self.meta_columns = ['Image_Path', 'Hierarchical_Path']
        
        if target_columns is None:
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
        else:
            self.target_columns = target_columns
        
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
        
        # Add Hierarchal Path
        df['FirstFolderID'] = (df.apply(self.rowIndex__, axis=1) % 50).astype('int')
        df['SecondFolderID'] = (df.PatientID % 50).astype('int')
        df['Hierarchical_Path'] = 'data/d' + \
                                    df['FirstFolderID'].apply(str) + '/d' + \
                                    df['SecondFolderID'].apply(str) + '/i' + \
                                    df.apply(self.rowIndex__, axis=1).apply(str) + '.jpg'

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
            
        if self.binary_targets:
            for c in self.target_columns:
                df_clean[c] = df_clean[c].map({-1:0, 0:0, 1:1})

        self.df_clean = df_clean.sort_index()
            
    def imageFolderPath(self):
        return self.train_image_path
            
    def intermediateFilePath(self):
        return self.intermediate_train_csv
            
    def intermediateFileExists(self):
        return glob.glob(self.intermediateFilePath())
    
    def saveCleanDF(self):
        self.df_clean.to_csv(self.intermediateFilePath())
     
    def warnFeatureImbalance(self, train, val, threshold=0.02):
        tdf, vdf = train[self.target_columns], val[self.target_columns]
        tlen, vlen = len(tdf), len(vdf)
        warning_message = ''
        for c in self.target_columns:
            tp = len(tdf[tdf[c] == 1]) / tlen
            vp = len(vdf[vdf[c] == 1]) / vlen
            if tp-vp >= threshold:
                warning_message += f'   {c}: {tp-vp:.2%}\n'
        
        if len(warning_message) > 0:
            warnings.warn("\nFeature Imbalance Detected (train % - val %):\n" + warning_message, stacklevel=2)
            
    def getCleanDF(self, n_random_rows=0, val_percent=0):
        '''
        Main function.  
        
        n_random_rows lets you randomly pick the number of rows you want to use.  Often times used to do quick tests on model changes
        
        val_percent will split the result into 2 DataFrames, train and val
        
        Note:  Only the complete DataFrame is stored in memory.  
        So recalling this method with random rows or the validation percent will not be deterministic.
        '''
        if self.df_clean is None:
            if not self.intermediateFileExists():
                self.getDF()
                self.reduceDF()
                self.cleanDF()
                self.saveCleanDF()

            self.df_clean = pd.read_csv(self.intermediateFilePath(), index_col=0)
            
            for c in self.target_columns:
                self.df_clean[c] = self.df_clean[c].astype('int8')
        
        result = None
        if n_random_rows == 0:
            # All rows in df_Clean
            result =  self.df_clean
        else:
            # Subset of df_clean
            result =  self.df_clean.sample(n=n_random_rows, random_state = self.random_state).sort_index()
            
        if val_percent > 0 and val_percent < 1:
            # train/val split
            # Note: The split can be done of full df_Clean or subset of df_clean
            np.random.RandomState(self.random_state)
            val_mask = np.random.rand(len(result)) < val_percent
            train, value = result[~val_mask].sort_index(), result[val_mask].sort_index()
            
            result = (train, value)
            
            self.warnFeatureImbalance(train, value)
    
        return result
    
    def displayImage(self, idx, use_hierarchical_path=False):
        '''
        Mostly a check to make sure we can retrieve an image
        
        See the docstrings at the class level about Hierarchical Path
        '''
        
        if use_hierarchical_path:
            return cv2.imread(os.path.join(os.getcwd(), self.getCleanDF().iloc[idx].Hierarchical_Path))
        else:
            return cv2.imread(os.path.join(os.getcwd(), self.getCleanDF().iloc[idx].Image_Path))
    
    def rowIndex__(self, row):
        # Private function to get value of the index for a row
        return row.name

class Dataset(torch.utils.data.Dataset):
    
    '''
    We build our own derived call of PyTorch's Dataset class.
    This allows us to do 2 primary things:
     - It allows us to read the clean DataFrame from above and store the multiple lables
     - It also allows us to format the output of the iteration to return the lables along with the image
     
    Since we pass in the transform into the constructor, the Dataset can have only one transformation.
    
    So we use the train/split from the CleanMetaData class to create multiple DataFrames.
    Each DataFrame along with a transformation is used to create a seperate Dataset for train and val.
    '''
    
    def __init__(self, df, transform=None, target_columns=None):

        # initialize the arrays to store the ground truth labels and paths to the images
        self.ImageID = []
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
        
        self.target_columns = target_columns

        # Load the image and lables for each row in the DataFrame
        for _, row in df.iterrows():
            self.ImageID.append(row.name)
            
            self.data.append(row.Hierarchical_Path) #Use Hierarchical Path so it will work for CoLab and local OS.
            
            #todo: Find loss function that combines multi-class (null, -1, 0, 1) and multi-label (Edema & Fracture & Cardiomegaly)
            # Their are 4 possible values for each target (null=No Mention, -1=Negative, 0=Uncertain, 1=Positive)
            # For now: we will just make null=No Mention, -1=Negative, 0=Uncertain all have a value of 0
            # i.e. Of the 12 targets, each will have a simple binary classification (Positive, Not-Positive)
            self.Enlarged_Cardiomediastinum.append(np.maximum(int(row['Enlarged_Cardiomediastinum']),0))
            self.Cardiomegaly.append(np.maximum(int(row['Cardiomegaly']),0))
            self.Lung_Opacity.append(np.maximum(int(row['Lung_Opacity']),0))
            self.Lung_Lesion.append(np.maximum(int(row['Lung_Lesion']),0))
            self.Edema.append(np.maximum(int(row['Edema']),0))
            self.Consolidation.append(np.maximum(int(row['Consolidation']),0))
            self.Pneumonia.append(np.maximum(int(row['Pneumonia']),0))
            self.Atelectasis.append(np.maximum(int(row['Atelectasis']),0))
            self.Pneumothorax.append(np.maximum(int(row['Pneumothorax']),0))
            self.Pleural_Effusion.append(np.maximum(int(row['Pleural_Effusion']),0))
            self.Pleural_Other.append(np.maximum(int(row['Pleural_Other']),0))
            self.Fracture.append(np.maximum(int(row['Fracture']),0))
            
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):
        
        result = None
        
        # take the data sample by its index
        img_path = self.data[idx]
        
        # read image
        if glob.glob(img_path):
            try:
                img = cv2.imread(img_path)
                img = Image.fromarray(img)
            except:
                try:
                    img = Image.open(img_path, 'r')
                except:
                    self.__next__
                pass

            # apply the image augmentations if needed
            if self.transform:
                img = self.transform(img)

            # return the image and all the associated labels
            if self.target_columns is None:
                labels = [
                            self.Enlarged_Cardiomediastinum[idx], #Enlarged_Cardiomediastinum
                            self.Cardiomegaly[idx], #Cardiomegaly
                            self.Lung_Opacity[idx], #Lung_Opacity
                            self.Lung_Lesion[idx], #Lung_Lesion
                            self.Edema[idx], #Edema
                            self.Consolidation[idx], #Consolidation
                            self.Pneumonia[idx], #Pneumonia
                            self.Atelectasis[idx], #Atelectasis
                            self.Pneumothorax[idx], #Pneumothorax
                            self.Pleural_Effusion[idx], #Pleural_Effusion
                            self.Pleural_Other[idx], #Pleural_Other
                            self.Fracture[idx] #Fracture
                          ]
            else:
                labels = []
                if 'Enlarged_Cardiomediastinum' in self.target_columns:
                    labels.append(self.Enlarged_Cardiomediastinum[idx])
                if 'Cardiomegaly' in self.target_columns:
                    labels.append(self.Cardiomegaly[idx])
                if 'Lung_Opacity' in self.target_columns:
                    labels.append(self.Lung_Opacity[idx])
                if 'Lung_Lesion' in self.target_columns:
                    labels.append(self.Lung_Lesion[idx])
                if 'Edema' in self.target_columns:
                    labels.append(self.Edema[idx])
                if 'Consolidation' in self.target_columns:
                    labels.append(self.Consolidation[idx])
                if 'Pneumonia' in self.target_columns:
                    labels.append(self.Pneumonia[idx])
                if 'Atelectasis' in self.target_columns:
                    labels.append(self.Atelectasis[idx])
                if 'Pneumothorax' in self.target_columns:
                    labels.append(self.Pneumothorax[idx])
                if 'Pleural_Effusion' in self.target_columns:
                    labels.append(self.Pleural_Effusion[idx])
                if 'Pleural_Other' in self.target_columns:
                    labels.append(self.Pleural_Other[idx])
                if 'Fracture' in self.target_columns:
                    labels.append(self.Fracture[idx])
            
            result = {
                'id': self.ImageID[idx],
                'img': img,
                'labels': torch.FloatTensor(labels)}
        return result            

class TranformType(Enum):
    '''
    Simple Enum to indictate which transformat we should use for a specific Dataset
    '''
    
    No = 0
    Train = 1
    Val = 2    

    
    
class Loaders():
    
    '''
    This class helps us create the data loaders for our PyTorch training loops.
    
    There are 3 variations of loaders we can build:
     - Loader with all images
     - Loader with a subset of images chossen at random
     - 2 Loaders split into train/val
     
     We can combine the subsetting with train/val.  
     This allows us to build training loops that are small for quick accessments of model changes.
     
     The image sizes, mean and SD are hard coded.  
     See the Capstone Project 2/notebooks/Support Notebooks for Modules/ChextXRayImages_NB.ipynb notebook for the caculations.
     
     Unlike the CleanMetaData class, this class does not cache any objects in memory.  
     This allows us to change things like batch size and row counts
    
    '''
    
    def __init__(self, 
                 image_width=320,
                 image_height=320,
                 affineDegrees=5, 
                 translatePrecent=0.05, 
                 shearDegrees=4, 
                 brightnessJitter=0.2, 
                 contrastJitter=0.1, 
                 augPercent=0.2,
                 observation_min_count = None,
                 target_columns=None):
            

        self.target_columns=target_columns
        self.cleanMetaData = CleanMetaData(target_columns=self.target_columns)    
        
        self.feature_columns = self.cleanMetaData.feature_columns
        self.meta_columns = self.cleanMetaData.meta_columns
        self.target_columns = self.cleanMetaData.target_columns
            
        self.pixel_mean = 0.5064167
        self.pixel_sd = 0.16673872
        
        self.image_width = image_width
        self.image_height = image_height
        
        self.translatePrecent=translatePrecent
        self.shearDegrees=shearDegrees
        self.brightnessJitter=brightnessJitter
        self.contrastJitter=contrastJitter
        self.augPercent=augPercent
        self.observation_min_count=observation_min_count
        
        self.train_transform = None
        self.val_transform = None
        
        self.df = None
        self.train_df = None
        self.val_df = None
        
    def oversampleDF(self, df, observation_min_count = 50):
        df_combined = pd.DataFrame(df.index) #builds a new df with the same index as df (same row count)
        total_rows = df_combined.shape[0]

        # Let's do this via concatenation instead of a groupby
        # This give us an easier way to see the combination without spaning multiple columns
        df_combined['Combined_Targets'] = ''
        for c in self.target_columns:
            df_combined['Combined_Targets'] = df_combined['Combined_Targets'] + df[c].map({-1:'N', 0:'0', 1:'P'})

        # Now do the groupby on the concatenated column
        # Remember, df_combined has the same cardinality as the passed in df
        df_combinations = pd.DataFrame(df_combined.groupby(['Combined_Targets']).count())
        df_combinations.columns = ['Observations']

        # We only want the combinations that have less rows than the threshold
        df_combinations_und = df_combinations[df_combinations.Observations < observation_min_count].copy()
        df_combinations_und['AddCount'] = (observation_min_count - df_combinations_und.Observations).astype('int')

        # Use above result to filter passed in df
        df_in = df.copy()
        df_in['Combined_Targets'] = df_combined['Combined_Targets'].copy()
        df_underrep = df_in[df_in.Combined_Targets.isin(df_combinations_und.index)].copy()

        # Let's leave the passed in df alone so we do a copy
        df_oversample = df.copy()
        df_dup_rows = df.iloc[0:0,:].copy()

        # Walk every combination with a deficit
        # Randomly pick rows with the same combination
        # Insert these random rows to fill in deficit
        combinations_with_deficit = df_combinations_und[df_combinations_und.AddCount>0].AddCount
        for combination, add_count in combinations_with_deficit.items():
            rows_to_copy = df_underrep[df_underrep.Combined_Targets==combination] \
                            .sample(add_count, replace=True).copy()
            df_dup_rows = df_dup_rows.append(rows_to_copy.copy(), sort=False)

        # We need to unique index for these rows
        # Let's make them negative so that we know they are oversample rows
        dup_row_count = len(df_dup_rows)
        new_index_values = np.linspace(1, dup_row_count, dup_row_count) * -1
        df_dup_rows.index = new_index_values

        # Append the dup rows into the oversample df
        df_oversample = df_oversample.append(df_dup_rows, sort=False)
        df_oversample = df_oversample.sort_index()

        return (df_oversample, len(df_dup_rows))       
        
    def getTransformations(self, tranformType=TranformType.No):
        
        if tranformType == TranformType.Train:
            if not self.train_transform:
                affineTrans = transforms.RandomApply(
                                        [transforms.RandomAffine(degrees=5, 
                                                                 translate=(self.translatePrecent,self.translatePrecent), 
                                                                 shear=self.shearDegrees)],
                                        p=self.augPercent)
                brightnessContrastTrans = transforms.RandomApply(
                                        [transforms.ColorJitter(brightness=self.brightnessJitter, 
                                                                contrast=self.contrastJitter)],
                                        p=self.augPercent)
                self.train_transform = transforms.Compose(
                                      [affineTrans,
                                      brightnessContrastTrans,
                                      transforms.Resize(size=(self.image_height,self.image_width), interpolation=2),
                                      transforms.Grayscale(1),
                                      transforms.ToTensor(),
                                      transforms.Normalize((self.pixel_mean,), (self.pixel_sd,))])              
            return self.train_transform
        elif tranformType == TranformType.Val:
            if not self.val_transform:
                self.val_transform = transforms.Compose(
                                      [transforms.Resize(size=(self.image_height,self.image_width), interpolation=2),
                                      transforms.Grayscale(1),
                                      transforms.ToTensor(),
                                      transforms.Normalize((self.pixel_mean,), (self.pixel_sd,))])
            return self.val_transform
        else:
            return None
    
           
    def getDataSet(self, n_random_rows = 0, tranformType=TranformType.No):
        md = self.cleanMetaData
        
        self.df = md.getCleanDF(n_random_rows)
        
        if self.observation_min_count is not None:
            self.df, total_rows_added = self.oversampleDF(self.df, 
                                                          observation_min_count=self.observation_min_count)
            print(f'Total Oversampled Rows Added: {total_rows_added:,}\n')
            
        self.feature_columns = md.feature_columns
        self.meta_columns = md.meta_columns
        self.target_columns = md.target_columns
        
        transform = self.getTransformations(tranformType)
        return Dataset(self.df, transform, target_columns=self.target_columns)
    
    def getDataLoader(self, batch_size=64, n_random_rows = 0, tranformType=TranformType.No):
        ds = self.getDataSet(n_random_rows, tranformType)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    def getTranValDataFrames(self, val_percent, n_random_rows = 0):
        md = self.cleanMetaData 
        
        df_train, df_val = md.getCleanDF(n_random_rows=n_random_rows, val_percent=val_percent)
        
        if self.observation_min_count is not None:
            df_train, total_rows_added = self.oversampleDF(df_train, 
                                                          observation_min_count=self.observation_min_count)
            print(f'Total Oversampled Rows Added to Train: {total_rows_added:,}\n')
            
        return (df_train, df_val)
    
    def getTrainValDataSets(self, val_percent, n_random_rows = 0):
        self.train_df, self.val_df = self.getTranValDataFrames(val_percent, n_random_rows)
           
        train_transform = self.getTransformations(TranformType.Train)
        val_transform = self.getTransformations(TranformType.Val)
        
        return (
                    Dataset(self.train_df, train_transform, target_columns=self.target_columns),
                    Dataset(self.val_df, val_transform, target_columns=self.target_columns)
                )
    
                
    def getDataTrainValidateLoaders(self, batch_size=64, val_percent = 0.2, n_random_rows = 0):
        train_dataset, val_dataset = self.getTrainValDataSets(val_percent, n_random_rows)

        train_load = torch.utils.data.DataLoader(train_dataset, 
                                                 batch_size=batch_size, 
                                                 shuffle=True)

        val_load = torch.utils.data.DataLoader(val_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=False)
        
        return (train_load, val_load)
 
        
