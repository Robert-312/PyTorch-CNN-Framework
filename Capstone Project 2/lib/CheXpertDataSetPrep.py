import os, os.path
import pandas as pd
import glob
from skimage import io

class CheXpertDataSetPrep():
    
    
    def __init__(self): 
        self.df = None
        self.df_reduced = None
        self.df_clean = None
    
    def getDF(self):
        # load CheXpert Image metadata
        # Though CheXpert gave use a validation set with it's own metadata, there were too few images to be useful
        # So we are only going to use the training images and split this into train and val/test

        # drive.mount('/content/drive')
        root_path = "../data/raw/"
        train_impage_path = root_path + "train/"
        train_csv = root_path + "train.csv"

        df = pd.read_csv(train_csv)

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
        df['Image_Path'] = df.Path.str.replace(path_prefix, train_impage_path)
        test_image_path = df.iloc[0]['Image_Path']


        self.df = df
        
    def reduceDF(self):
        self.df_reduced = self.df[
                (df.View == 'Frontal') & # Don't show x-rays from the side
                (df.StudyID < 5) & # Don't include more than 4 studies for a single patient
                ((df.Orientation == 'AP') | (df.Orientation == 'PA')) # Don't show Left or Right Lateral
                ]
     
    def cleanDF(self):
        # The only columns with null values are the targets
        # -1 = negative finding (the patient does NOT have this finding)
        #  1 = positive finding (the patient HAS this finding)
        #  0 = no position on the finding
        # So it should be safe to change all the nan in the df to 0 (no position)
        df_clean = df_reduced.fillna(0, inplace=False)

        # We already extracted all the information we needed from path and view, so we can drop these columns
        df_clean.drop(['Path', 'View'], axis=1, inplace=True)

        # This leaves us with just 2 categorical columns
        # So lest hot-one enode these
        df_clean = pd.get_dummies(df_clean, columns=['Sex', 'Orientation'], drop_first=True)
        # Sex: Not Male or Unknown = Female
        # Orientation: Not PA = AP

        # Let's clean up the column order with features first
        feature_columns = ['PatientID',
                            'StudyID',
                            'Age',
                            'Sex_Male',
                            'Sex_Unknown',
                            'Orientation_PA']
        meta_columns = ['Image_Path']
        target_columns = ['No_Finding',
                            'Enlarged_Cardiomediastinum',
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
                            'Fracture',
                            'Support Devices'
                           ]
        df_clean = df_clean[feature_columns + meta_columns + target_columns]

        for c in target_columns:
            df_clean[c] = df_clean[c].astype('int8')

        self.df_clean = df_clean
            
    def intermediateFilePath(self):
        # drive.mount('/content/drive')
        root_path = "../data/intermediate/"
        intermediate_train_csv = root_path + "train.csv"
        return intermediate_train_csv
            
    def intermediateFileExists(self):
        return glob.glob(self.intermediateFilePath())
    
    def saveCleanDF(self):
        self.df_clean.to_csv(self.intermediateFilePath())
            
    def getCleanDF(self):
        if self.df_clean is None:
            if not self.intermediateFileExists():
                self.getDF()
                self.reduceDF()
                self.cleanDF()
                self.saveCleanDF()

            self.df_clean = pd.read_csv(self.intermediateFilePath(), index_col=0)
        
        return self.df_clean
    
    def displayImage(self, idx):
        return io.imread(self.getCleanDF().iloc[idx].Image_Path, as_gray=False)
        