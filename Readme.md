# Overview

This is a CNN project for the **CheXpert** dataset consisting of labeled chest x-ray images.  There are over 200k x-rays in the dataset, each labeled with 14 findings.  The dataset in its raw form is both multi-label and multi-class in that each x-ray can have more than one label, and each of these labels can be one of 4 values (not mentuned, negative, uncertain and positive).

This project will reduce this dataset to about 137k images with 12 binary labels making this project just a multi-label CNN problem.

The dataset comes from the Stanford Machine Learning group and can be accessed via this link:
https://stanfordmlgroup.github.io/competitions/chexpert/

# Purpose

The purpose of this project is to build a framework that will easily allow repeatable runs of the dataset with different parameters, images counts, models, etc.  This framework will be used to determine the best starting model and it's initial hypterparameters.  

Once a good starting point is found, a new project can be created that will fine tune and productionize the winning approach from this project.  

# Framework ERD

Below if a high level entity relationship diagram of how the framework was built:

![alt text](Diagrams/StandardTraining.png "Standard Training Overview") 


# Table of Contents

- [Goals](#Goals)
- [Educational Notebooks](#Educational_Notebooks)
- [Dataset](#Dataset)
- [Project Directory Structure](#Project_Directory_Structure)
- [Environments](#Environments)
- [EDA](#EDA)
- [CleanMetaData](#CleanMetaData)
	- [Intrinsic Leaks](#Intrinsic_Leaks)
	- [Target Interdependence](#Target_Interdependence)
	- [Imbalance](#Imbalance)
- [Modules](#Modules)
    - [CheXpertData Module](#CheXpertData_module)
    - [TrainingLoop and Metrics_Modules](#TrainingLoop_and_Metrics_Modules)
    - [Standard raining Module](#Standard_Training_Module)
- [Criterion and Optimizer](#Criterion_Optimizer)
- [Models](#Models)
	- [Custom](#Custom)
	- [ResNet](#ResNet)
	- [DenseNet](#DenseNet)
- [Results](#Results)
    - [Single Target Runs](#Single_Target_Runs)
    - [5 Targets](#5_Targets)
    - [All 12 Targets](#All_12_Targets)
    - [20 Epochs](#20_Epochs)
    - [5 Independent Models](#5_Independent_Models)
- [Conclusions](#Conclusions)
- [Future Areas of Exploration](#Future_Areas_of_Exploration)
- [Final Thoughts](#Final_Thoughts)

# Goals <a class="anchor" id="Goals"></a>

### Build a Framework for better Understanding

One of the primary goals of this project is to gain a deeper understanding of CNN and PyTorch.  There are already a lot of frameworks out there that do a lot more than what is built into this project, Keras and TensorBoard are some good examples.

But it is very difficult to get a good understanding of what is happening behind the scenes if the work is already done for you.  So in this project, preference is given to manual coding over pre-built packages.  

The coding mistakes and wrong assumptions made in this journey are invaluable.  In fact, these missteps are the key to better understanding.  Things like:
- Why can't I build the ROC curve from just my actual and predicted labels?
- Why is my accuracy over 90% when all my predictions are negative?
- Why can't I use pre-trained models with my grayscale images?
- How do I get my model's output to match my loss function?

There is no intention of making a framework that is better than ones already available.  Chances are for future CNN projects, I will use a lot more pre-built packages.  But the hope is the understanding gained with this project will allow me to leverage features and parameters in these packages much more accurately and efficiently.

### Code Reuse / Repeatable Model Runs

Jupyter Notebooks are a great way of combining code, visualizations and descriptive text in a single location.  But using notebooks for repeatable runs is not the most efficient approach.  You can copy/paste code or entire notebooks, but if you need to make a change, you have to change every notebook.  

Building Python modules to house the repeatable code is a much better approach and the route taken with this project.

Notebooks are used for specific runs, i.e. DenseNet vs. ResNet, Learning Rate 1e-4 vs. Learning Rate 1e-6.  But since most of the code is in modules, these notebooks (other than a couple of bootstrap blocks) contain only the model and training loop configurations.

### Model Goals

Though this project is more focused on understanding and building a framework, it is still important to have a target model goals to shoot for.

The order of importance for the model results are:
- Recall
- ROC Area Under the Curve
- Precision

This is somewhat arbitrary, but the thought is it would be better to suggest a false positive finding to a Radiologist then to miss a possible diagnosis.  Radiologists tend to make their own judgments about x-rays, but having a suggested finding might influence the doctor to spend more time analyzing these suggestions.  For this reason, Recall was chosen as the primary metric.

ROC is always important since this is more of a true metric of how well the model performed since it is based on probabilities and not predictions.  You can have the same Recall values for two models, but one can have a much high AUC.  This is because the better model might have most of its probability scores closer to 1 while the lesser model has most of its probabilities closer to 0.5. But this might not be a score the Radiologist sees, so we put it into the second priority slot.

We can easily make our recall values 100% by always predicting a positive finding.  So Recall is of no value if we don't keep the precision under control.  So Precision cannot be forgotten.  Looking at the F1 score might be a good approach for this.  But since we wan't to favor Recall, we might use a weighted harmonic mean, i.e. F2 (see sklearn.metrics.fbeta_score).  


# Educational Notebooks <a class="anchor" id="Educational_Notebooks"></a>

### Linear Regression from Scratch using PyTorch

<a href="notebooks/Educational/Pytorch%20Linear%20Regression%20from%20Scratch.ipynb" >Pytorch Linear Regression from Scratch</a>

This notebook explored a simple linear regression using PyTorch.  A simple model is built from scratch using manually built linear and loss functions and a simple gradient descent class that inherits from torch.optim.Optimizer.

This notebook then goes into polynomial regression to predict the coefficients of a 3rd degree polynomial.
    

### Autograd package in PyTorch    
<a href="notebooks/Educational/Pytorch%20Automatic%20Differentiation.ipynb" >Pytorch Automatic Differentiation</a>

This notebook looks at the torch.autograd package and explores how PyTorch implements automatic differentiation.  Various functions including polynomials are looked at and how the computational graphs along with the chain rule are used during forward and backward passes to determine the gradient of the function with respect to the input variables.

# Dataset <a class="anchor" id="Dataset"></a>

The raw dataset comes as a set of images along with a csv index file.  There is one row in the csv file for each x-ray.
There are both train and validation dataset with 223,415 and 234 images respectively.  Since the validation set was so small, only the train dataset was used.  These 223,415 images were randomly split into train and val during the model runs.

The csv contains some tabular features such as sex and age, aspectes of the image such as orientation and view, and a path for the image file.

The csv also contains the 14 labels:
- Enlarged_Cardiomediastinum
- Cardiomegaly
- Lung_Opacity
- Lung_Lesion
- Edema
- Consolidation
- Pneumonia
- Atelectasis
- Pneumothorax
- Pleural_Effusion
- Pleural_Other
- Fracture
- No Finding
- Support Devices (i.e. pacemakers)

Any image can have any number of labels making this a multi-label dataset.  i.e. An x-ray can show both Pleural Effusion and Cardiomegaly.

Also, each of these labels can have one of 4 values:
- Not Mentioned by Radiologists = Null
- Negative = -1
- Uncertain = 0
- Positive = 1

How the x-ray was taken is stored in 2 columns, "Frontal/Lateral" and "AP/PA".  Lateral x-rays are taken from the patient's side.  AP/PA stands for Anterior and Posterior.  AP means the x-rays entered the patient's front and exited out of the patient's back and PA is the opposite.  

From the image path, we can parse out a PatientID and StudyID.  A study is a set of x-rays taken at the same time.  A study can have just a single frontal x-ray, a single lateral x-ray or have both a frontal and lateral set of x-rays.
Patients who only had one set of x-rays will have only a StudyID of 1.  Patients with multiple studies will have StudyIDs 1,2,3,4,....  The max number of studies for a single patient in this dataset was 72.

#### Competition
CheXpert was set up as a competition, but only 5 of the targets are included:
['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural_Effusion']

So many of model runs will only focus and/or train only on these 5 labels.

*Note: This project is NOT intended to enter this competition*

# CheXpert User Agreement

The images in this dataset cannot be made publically available.  Because of this, the images are not included in this project.  But you can obtain the dataset yourself and run this code.  You may need to adjust the working directories in the top bootstrap code block at the start of every notebook.

*Note:  See Hierarchical Path below for special directory structure*

# ═════════════════════════
# Project Directory Structure <a class="anchor" id="Project_Directory_Structure"></a>


## <a href="data" >data</a>

*   **<a href="data/raw" >raw</a>**
    -   Holds metadata csv files from CheXpert
    -   Holds download x-ray images

*    **<a href="data/intermediate" >intermediate</a>**
    -   Holds Clean DF

*    **<a href="data/d0" >d0</a> through <a href="data/d49" >d49</a>**
    -   Holds same x-ray images, but in a hierarchical directory structure for CoLab

## <a href="modules" >modules</a>

* **<a href="modules/lib" >lib</a>**
    -   **<a href="modules/lib/CheXpertData.py" >CheXpertData.py</a>**
        * Clean Meta Data (DataFrame)
        * PyTorch DataSet (Subclass of torch.utils.data.Dataset)
        * PyTorch DataLoaders
    -   **<a href="modules/lib/Metrics.py" >Metrics.py</a>**
        * Holds results passed to it via training
        * Displays metrics of the run along with things like ROC curves
    -   **<a href="modules/lib/StandardTraining.py" >StandardTraining.py</a>**
        * Runs the training loop with overridable default parameters
        * Allow consistency between different run
        * ModelLoop class - Run multiple combinations of parameters in same notebook
    -   **<a href="modules/lib/TrainingLoop.py" >TrainingLoop.py</a>**
        * Runs the training
        * Pass in, NN, cuda device, optimizer, loss function and Metrics class described above

* **<a href="modules/models" >models</a>**
    -   **<a href="modules/models/CustomPneumonia.py" >CustomPneumonia.py</a>**
        * Custom build CNN model
        * Intended to be more educational rather than effective
        * Goal was to make sure:
            - Shapes match between each layer and between convolution and fully connected layers
            - Ability to calculate the number of trainable parameters
    -   **<a href="modules/models/DenseNet.py" >DenseNet.py</a>**
        * A fairly standard structure of DenseNet (see below for more details)
    -   **<a href="modules/models/ResNet.py" >ResNet.py</a>**
        * ResNet_GrayScale
            - Uses torchvision.models.resnetXX models
            - Overrides conv1 to take in 1 input channel
            - Add fully connected layer at end of forward to reduce the 1000 changes to the desired output length
            - Registers a forward hook to add dropout2d is drop_out_precent is passed in
        * ResNet_Pretrained
            - Uses torchvision.models.resnetXX models with pretrained=True
            - Keeps input channels as 3
            - Add fully connected layer at end of forward to reduce the 1000 changes to the desired output length
            - Registers a forward hook to add dropout2d is drop_out_precent is passed in
            - Uses torch.repeat_interleave to duplicate the 1 grayscale channel to 3 channels

## <a href="notebooks" >notebooks</a>
-   **<a href="notebooks/Educational" >Educational</a>** - Two notebooks exploring PyTorch functionality
    * <a href="notebooks/Educational/Pytorch%20Linear%20Regression%20from%20Scratch.ipynb" >Pytorch Linear Regression from Scratch</a>
    * <a href="notebooks/Educational/Pytorch%20Automatic%20Differentiation.ipynb" >Pytorch Automatic Differentiation</a>
-   **<a href="notebooks/Kaggle%20Pneumonia" >Kaggle Pneumonia</a>**
    * Very early work with CNN on Kaggle Pneumonia chest x-rays
    * Basis of CustomPneumonia model
-   **<a href="notebooks/ModelLoop" >ModelLoop</a>**
    * Notebooks that run combinations of different modes and/or parameters with a reduced dataset
-   **<a href="notebooks/ModelRuns" >ModelRuns</a>**
    * Notebooks that train a single model and parameters
    * **<a href="notebooks/ModelRuns/saved" >saved</a>**
        - Holds the pickle serialization of the model runs
        - *Note: Due to size constraints, this folder was not committed to GitHub*
-   **<a href="notebooks/Support%20Notebooks%20for%20Modules" >Support Notebooks for Modules</a>**
    * The workspace used to help build the modules
    * <a href="notebooks/Support%20Notebooks%20for%20Modules/CheXpertData_NB.ipynb" >CheXpertData_NB</a>
    * <a href="notebooks/Support%20Notebooks%20for%20Modules/EDA.ipynb" >EDA</a>
    * <a href="notebooks/Support%20Notebooks%20for%20Modules/HierarchicalPath.ipynb" >HierarchicalPath</a>
    * <a href="notebooks/Support%20Notebooks%20for%20Modules/Metrics%20and%20Training%20Loop.ipynb" >Metrics and Training Loop</a>
    * <a href="notebooks/Support%20Notebooks%20for%20Modules/Metrics_NB.ipynb" >Metrics_NB</a>
    * <a href="notebooks/Support%20Notebooks%20for%20Modules/ModelLoop_NB.ipynb" >ModelLoop_NB</a>
    * <a href="notebooks/Support%20Notebooks%20for%20Modules/Oversampling.ipynb" >Oversampling</a>
    * <a href="notebooks/Support%20Notebooks%20for%20Modules/StandardTraining_NB.ipynb" >StandardTraining_NB</a>
    * <a href="notebooks/Support%20Notebooks%20for%20Modules/TrainingLoop_NB.ipynb" >TrainingLoop_NB</a>

# ═════════════════════════

# Environments <a class="anchor" id="Environments"></a>

This project is intended to run both locally with GPU and on Google CoLab.  Locally (GEFORCE RTX 2080 - 8GB), most models run very well, but might need to adjust the batch size or image size to fit in the available memory. The same notebooks can also run on Google CoLab without any changes.  The bootstrap code block at the top of the notebooks determines the working directory on both environments and sets the PyTorch device to cuda. 

Most if not all of the notebooks in this project were completed locally.  Though CoLab has the advantage from a computational power perspective, it is very slow accessing the image files.

## Hierarchical Directory

To see how to implement the hierarchical path, see: 
<a href="notebooks/Support%20Notebooks%20for%20Modules/HierarchicalPath.ipynb" >Hierarchical Path Notebook</a>

In Google CoLab, the images need to be accessible.  Since the dataset is not public and there is no API to retrieve the images, you need to mount a drive in order for CoLab to get access to these images.  In the CoLab environment, the images are stored on Google Drive.  There are plenty of posts on how to mount a drive in this way.

But the directory structure on Google Drive is not the same as your local file system.  Since Google Drive is primarily a web based application, it cannot navigate directly to a file via a path.  So every time you give it a path, a call must be made to obtain an ID for the folder and or path before it can access it.  That means that every image read during the training loop must do a lookup of the item ID first.

With over 200k files, the training loop would have to find the ID out of 200,000 IDs.  But it has to do this for every image, so the number of lookups in a single epoch would be 200k X 200k which is over 40 billion lookups.

In fact, just uploading the images to Google Drive was extremely slow.  After several days of upload, the process was still not done.  A silent timeout error was occuring most of the time so many of the uploaded images didn't get to the drive.  See top of HierarchicalPath.ipynb (link above).

After a fair amount of research, the approach taken to resolve this was to copy the files into a hierarchical folder structure where no more than 50 items existed in any one folder.  To get the proper ID, there are only 3 ID lookups, each with only 50 items.  So instead of 200k lookup, we only needed to do 150.  

Though this did resolve the timeout issue, the training loops were still much slower than running locally.  So all models were run locally as long as there was enough memory to support it.

*Note:  The hierarchical directories greatly helped the upload process!*

# EDA  <a class="anchor" id="EDA"></a>

*↓ Bootstrap code to set working directory and mount drive in CoLab ↓*


```python
import sys
import os, os.path

sys.path.append(os.path.join(os.getcwd() ,'/modules'))
root_path = "C:/git/Springboard-Public/Capstone Project 2/"
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    root_path = "/content/drive/My Drive/Capstone Project 2/"

print('Current Working Dir: ', os.getcwd())
print('Root Path: ', root_path)

# We need to set the working directory since we are using relative paths from various locations
if os.getcwd() != root_path:
  os.chdir(root_path)
```

    Current Working Dir:  C:\git\Springboard-Public\Capstone Project 2
    Root Path:  C:/git/Springboard-Public/Capstone Project 2/
    


```python
import sys
import os, os.path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from modules.lib.CheXpertData import CleanMetaData
from modules.lib.TrainingLoop import *
from modules.lib.Metrics import *
from modules.lib.StandardTraining import *

from modules.models.CustomPneumonia import CustomPneumoniaNN
from modules.models.ResNet import ResNet_GrayScale, ResNet_PreTrained
from modules.models.DenseNet import DenseNet

from torchsummary import summary

%matplotlib inline
```

# Data Cleansing

To see the process of data cleaning, please go to the supporing notebook: 
<a href="notebooks/Support%20Notebooks%20for%20Modules/EDA.ipynb" >EDA Support Notebook</a>

To see the module, please go to:
<a href="modules/lib/CheXpertData.py" >CheXpertData.py</a>

### Frontal only
To reduce the complexity of learning and to reduce the overall number of images, only frontal views were included.  

### Multiple Studies
To prevent a patients with serial studies biasing the results, only the first 4 studies were included in the clean dataset. 

These 2 filters reduced the image count from 223,415 to 131,748.

    df_reduced = df[
                (df.View == 'Frontal') & # Don't show x-rays from the side
                (df.StudyID < 5) & # Don't include more than 4 studies for a single patient
                ((df.Orientation == 'AP') | (df.Orientation == 'PA')) # Don't show Left or Right Lateral
                ]

### 12 Targets
Two targets were removed from the original 14:
- No Finding
- Support Devices

The first was removed for the same reason we do with one-hot encoding.
Support Devices (i.e. Pacemakers, LVAD, etc.) was removed since these findings could be considered more of a feature than a diagnosis.

### Binary Labels
To reduce the complexity of combining multi-label and multi-class classifications, the labels were converted into Boolean values.

- Not Mentioned = 0
- Negative = 0
- Uncertain = 0
- Positive = 1

# CleanMetaData <a class="anchor" id="CleanMetaData"></a>


```python
metaData = CleanMetaData()
target_columns = metaData.target_columns
df_clean= metaData.getCleanDF()
display(df_clean)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientID</th>
      <th>StudyID</th>
      <th>Age</th>
      <th>Sex_Male</th>
      <th>Sex_Unknown</th>
      <th>Orientation_PA</th>
      <th>Support Devices</th>
      <th>Image_Path</th>
      <th>Hierarchical_Path</th>
      <th>Enlarged_Cardiomediastinum</th>
      <th>...</th>
      <th>Lung_Opacity</th>
      <th>Lung_Lesion</th>
      <th>Edema</th>
      <th>Consolidation</th>
      <th>Pneumonia</th>
      <th>Atelectasis</th>
      <th>Pneumothorax</th>
      <th>Pleural_Effusion</th>
      <th>Pleural_Other</th>
      <th>Fracture</th>
    </tr>
    <tr>
      <th>ImageID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>68</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>data/raw/train/patient00001/study1/view1_front...</td>
      <td>data/d0/d1/i0.jpg</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>87</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>data/raw/train/patient00002/study2/view1_front...</td>
      <td>data/d1/d2/i1.jpg</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>83</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>data/raw/train/patient00002/study1/view1_front...</td>
      <td>data/d2/d2/i2.jpg</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>41</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>data/raw/train/patient00003/study1/view1_front...</td>
      <td>data/d4/d3/i4.jpg</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>data/raw/train/patient00004/study1/view1_front...</td>
      <td>data/d5/d4/i5.jpg</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>223409</td>
      <td>64537</td>
      <td>2</td>
      <td>59</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>data/raw/train/patient64537/study2/view1_front...</td>
      <td>data/d9/d37/i223409.jpg</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>223410</td>
      <td>64537</td>
      <td>1</td>
      <td>59</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>data/raw/train/patient64537/study1/view1_front...</td>
      <td>data/d10/d37/i223410.jpg</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>223411</td>
      <td>64538</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>data/raw/train/patient64538/study1/view1_front...</td>
      <td>data/d11/d38/i223411.jpg</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>223412</td>
      <td>64539</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>data/raw/train/patient64539/study1/view1_front...</td>
      <td>data/d12/d39/i223412.jpg</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>223413</td>
      <td>64540</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>data/raw/train/patient64540/study1/view1_front...</td>
      <td>data/d13/d40/i223413.jpg</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>131748 rows × 21 columns</p>
</div>


# Intrinsic Leaks <a class="anchor" id="Intrinsic_Leaks"></a>
Though there is nothing we can do about this, there are intrinsic leaks in the images that we should be aware of.

In-patients are much more likely to have positive findings then out-patients, especially out-patient routine screening x-rays. In-patient x-rays oftentimes can have leaks in the images.  ECG leads, IVs and gown snaps are often present.  Higher acuity patients will oftentimes have their x-rays taken in bed with AP orientation.  PA is preferred since it does not tend to exaggerate the cardiac silhouette.  Also, it might be harder to properly align very sick patients.  All of these can leak features more common in positive diagnoses.  

### X-Ray with ECG Leads indicating an In-patient

*Note:  What looks like a watermark on the patient's right side (left side of image) is most likely a defibrillator pad.*



```python
img = metaData.displayImage(123)
imgplot = plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
```


![png](Readme%20Images/output_19_0.png)


# Target Interdependence <a class="anchor" id="Target_Interdependence"></a>
Though each of the 12 labels can be found independently of each other, it is very common to have findings that correlate with other findings.  For example, it would be more common to find atelectasis in patients with pneumonia or pleural effusions.  Rib fractures may make it painful to ambulate potentially leading to other lung diseases such as atelectasis or pneumonia.

Patients with high comorbidities may have several of these findings.  Diagnoses like Cardiomeglia tend to not be transient and can persist for years.  These longer term diseases may or may not correlate to other pathologies.


```python
ax = df_clean.iloc[:, -12:-1].sum(axis=1).hist(bins=20) 
ax.set_title('Frequency of Multiple Targets')
ax.set_xlabel('Number of Positive Findings')
ax.set_ylabel('Frequency')
plt.show()
```


![png](Readme%20Images/output_21_0.png)


# Imbalance <a class="anchor" id="Imbalance"></a>

One of the biggest challenges is the imbalance of the 12 targets.  The most frequent target, Lung Opacity, occurs in about 45% of the x-rays.  But as you can see below, the majority of the labels have less than a 15% positive occurrence.

This imbalance along with non-independent multi-label classification will comprise the majority of the obstacles for this project.



```python
ax =pd.DataFrame(df_clean.iloc[:, -12:].sum() / len(df_clean) * 100, 
             columns=['% Pos']).sort_values('% Pos').plot.barh(figsize=(10,6))
ax.set_title("Frequency of a Positive Diagnosis")
ax.set_xlabel("Positive Rate %")
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 5))
ax.grid()
```


![png](Readme%20Images/output_23_0.png)


# Modules  <a class="anchor" id="Modules"></a>



# CheXpertData module (DataFrame, Dataset and DataLoaders)  <a class="anchor" id="CheXpertData_module"></a>

Please go to the supporting notebook for these classes: 
<a href="notebooks/Support%20Notebooks%20for%20Modules/CheXpertData_NB.ipynb" >CheXpertData Notebook</a>

To see the module, please go to:
<a href="modules/lib/ChextXRayImages.py" >ChextXRayImages</a>

### CleanMetaData (DataFrame)
In addition to the data cleansing, the CleanMetaData class performs 2 critical functions:
- Randomly select rows from metadata 
    * This allow us to do model runs with reduced size for debugging and parameter evaluation
- Splits the rows randomly into train and validation
    * This is be done after the the dataset has been reduced from the above step
    * The balance is checked between train and val
        - For each feature, the % positives is found
        - If the difference in this percentage between train and val exceeds the default 2%, a warning is displayed
        - The warning lists all features that have an imbalance
        - This is often seen with very low row counts
        - See **Warnings** section of <a href="notebooks/Support%20Notebooks%20for%20Modules/EDA.ipynb#Warnings">EDA</a>:


### Dataset
This is a subclass of torch.utils.data.Dataset.  The main purpose of the derived class to use the clean metadata to get the image path, the ground truth for the 12 targets and the ImageID index value of the image.

One the init method, lists are created to hold the above data.  The clean dataframe is walked to load these lists.

In the getitem method, 3 tasks are performed:
- Get the ImageID
- Pull the PIL image from the image path and apply any transformations
- Build a vector of the target values

The output of the getitem is a dictionary with 3 keys, 'id', 'img' and 'labels'.  During training, this dictionary has to be parsed manually before feeing the images into the model.

### Loaders

This class is used to obtain instances of torch.utils.data.DataLoader.  A single loader for all images in the clean dataframe can be returned, but most of the time 2 loaders are returned for train and validation.

This class uses the other 2 classes in the CheXpertData module, **CleanMetaData** and **Dataset**.  Because of these, these 2 classes are usually never called directly.

On instantiation of this class, you pass in the following parameters:

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

The image shape parameters are self explanatory.  The next 5 parameters are used to add image augmentations to the train transformation sequence (RandomAffine and ColorJitter).  The augPercent is the percentage that these 2 augmentation types are applied to the dataset.

The observation_min_count parameter is for oversampling (see below).  

The target_columns let you reduce the labels of the dataset.  
If you pass in:
- **training_columns = ['Pleural_Effusion', 'Edema']**-

only these 2 labels will be used for training and metrics.

In addition, the pixel mean and standard deviation are hard coded with values found in the support notebook:

        self.pixel_mean = 0.5064167
        self.pixel_sd = 0.16673872


The most common method used in this class is:
    
    def getDataTrainValidateLoaders(self, batch_size=64, 
                                    val_percent = 0.2, 
                                    n_random_rows = 0):
                                    
This method returns a tuple of (train, val) data loaders with the total size and split percent passed in.                                    
        
#### Oversampling
The Loaders class also does the oversampling. We cannot use tools like SMOTE for this.  Our features are embedded in the images and are derived via the convolutions during training.  Because of this, we have nothing metrizable to do a nearest neighbors concept.  

But we do have the combination of targets.  There are 802 unique ground truth vectors in our clean dataframe.  If you look at the frequencies, you see the typical exponential decal pattern.  The all negative vector [0,0,0,0,0,0,0,0,0,0,0,0] is the most common followed by "Lung Opacity only" and "Lung Opacity, Pleural_Effusion only".  The counts very quickly drop to very low numbers after the first few combinations.

So the approach to oversampling is to group rows by unique target vectors.  A parameter **observation_min_count** can be passed into the class.  This parameter is used to find any target vector who has less observations than this threshold.  Then the observations in this group are sampled with replacement to add duplicate rows to the dataframe to reach the min count required.

Please go to the supporting notebook that help derived this method: 
<a href="notebooks/Support%20Notebooks%20for%20Modules/Oversampling.ipynb" >Oversampling Notebook</a>



```python
total_rows = df_clean.shape[0]
df_combined = pd.DataFrame(df_clean.PatientID)
df_combined['Combined_Targets'] = ''
for c in target_columns:
    df_combined['Combined_Targets'] = df_combined['Combined_Targets'] + df_clean[c].map({-1:'0', 0:'0', 1:'1'})
   
df_combinations = pd.DataFrame(df_combined.groupby(['Combined_Targets']).count())
df_combinations.columns = ['Frequency']
df_combinations['Percent'] = df_combinations.Frequency / total_rows
print(f'Total Unique Target Combinations {df_combinations.shape[0]:,}\n')
df_combinations = df_combinations.sort_values(['Frequency'], ascending=False)
print('\n'.join(target_columns))
display(df_combinations.head(15))
display(df_combinations.tail(5))
```

    Total Unique Target Combinations 802
    
    Enlarged_Cardiomediastinum
    Cardiomegaly
    Lung_Opacity
    Lung_Lesion
    Edema
    Consolidation
    Pneumonia
    Atelectasis
    Pneumothorax
    Pleural_Effusion
    Pleural_Other
    Fracture
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Frequency</th>
      <th>Percent</th>
    </tr>
    <tr>
      <th>Combined_Targets</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>000000000000</td>
      <td>25805</td>
      <td>0.195866</td>
    </tr>
    <tr>
      <td>001000000000</td>
      <td>12816</td>
      <td>0.097277</td>
    </tr>
    <tr>
      <td>001000000100</td>
      <td>10586</td>
      <td>0.080350</td>
    </tr>
    <tr>
      <td>000010000000</td>
      <td>4834</td>
      <td>0.036691</td>
    </tr>
    <tr>
      <td>000000000100</td>
      <td>4770</td>
      <td>0.036205</td>
    </tr>
    <tr>
      <td>001010000100</td>
      <td>4584</td>
      <td>0.034794</td>
    </tr>
    <tr>
      <td>001010000000</td>
      <td>4116</td>
      <td>0.031241</td>
    </tr>
    <tr>
      <td>000000010000</td>
      <td>3409</td>
      <td>0.025875</td>
    </tr>
    <tr>
      <td>000000001000</td>
      <td>2900</td>
      <td>0.022012</td>
    </tr>
    <tr>
      <td>000010000100</td>
      <td>2479</td>
      <td>0.018816</td>
    </tr>
    <tr>
      <td>001000010000</td>
      <td>2476</td>
      <td>0.018793</td>
    </tr>
    <tr>
      <td>010000000000</td>
      <td>2293</td>
      <td>0.017404</td>
    </tr>
    <tr>
      <td>000000010100</td>
      <td>2206</td>
      <td>0.016744</td>
    </tr>
    <tr>
      <td>001000010100</td>
      <td>1766</td>
      <td>0.013404</td>
    </tr>
    <tr>
      <td>000000000001</td>
      <td>1716</td>
      <td>0.013025</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Frequency</th>
      <th>Percent</th>
    </tr>
    <tr>
      <th>Combined_Targets</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>010000100010</td>
      <td>1</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <td>010000100001</td>
      <td>1</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <td>100101000000</td>
      <td>1</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <td>010000001101</td>
      <td>1</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <td>111110100100</td>
      <td>1</td>
      <td>0.000008</td>
    </tr>
  </tbody>
</table>
</div>


# TrainingLoop and Metrics Modules  <a class="anchor" id="TrainingLoop_and_Metrics_Modules"></a>

Both of the modules hold a single class with the same name.

To see the modules, please go to:

<a href="modules/lib/TrainingLoop.py" >TrainingLoop</a>

<a href="modules/lib/Metrics.py" >Metrics</a>

Please go to the supporting notebooks, please go to: 

<a href="notebooks/Support%20Notebooks%20for%20Modules/Metrics_NB.ipynb" >Metrics Notebook</a>

<a href="notebooks/Support%20Notebooks%20for%20Modules/TrainingLoop_NB.ipynb" >TrainingLoop Notebook</a>

<a href="notebooks/Support%20Notebooks%20for%20Modules/Metrics and Training Loop.ipynb" >Metrics and Training Loop Notebook</a>

The TrainingLoop performs the training epochs and passes the results to an instance of the Metrics object.  The Metrics object holds the probabilities and predictions for each epoch run.  It also displays and saves this data.  The Metrics instance can be used in the training loop to display selected results after each epoch.  It also allows you to display all metrics including epoch by epoch plots after training is completed.


### TrainingLoop

This is a relatively simple class.  On instantiation, you pass in the device (CPU or CUDA), NN, optimizer and loss functions and an instance of the Metrics class.

    def __init__(self, device, net, optimizer, criterion, metrics):
    
The **train** method gets passed in the number of epochs and the train and validation data loaders.

    def train(self, num_epochs, train_loader, val_loader):

For each epoch, both the training and validation loaders are enumerated.  The size of the batches is a parameter in the 2 data loaders, so the train code is unaware of this value.  The train() and eval() methods are called on the model for train and val respectively.  Only the train gets back propagation while val calls torch.no_grad().

The main calls in the epoch loops are below:

    self.net.train()
    for i, data in enumerate(train_loader, 0): 
        ids, inputs, labels, outputs = self.processBatch(data)
        self.metrics.appendEpochBatchData(ids, outputs)
        self.epoch_loss += self.backProp(outputs, labels)

    ...

    self.net.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):         
            ids, inputs, labels, outputs = self.processBatch(data, is_validation=True)
            self.metrics.appendEpochBatchData(ids, outputs, is_validation=True)        
        
The processBatch method is:

    def processBatch(self, data, is_validation=False):
        # Convert output from loader
        ids, inputs, labels = self.parseLoaderData(data)
        if not is_validation:
            # zero the parameter gradients
            self.optimizer.zero_grad()
        # Get outputs from Model
        outputs = self.net(inputs)
        return ids, inputs, labels, outputs 
                
Since the dataset subclass returns a dictionary with the getitem method, the values are parsed with:

    ids, inputs, labels = data['id'], data['img'], data['labels']
    # move data to device GPU OR CPU
    inputs, labels = inputs.to(self.device), labels.to(self.device)
    
The inputs variable is the transformed images for the batch.  
The shape of this tensor is **[batch_size, channels, height, width**]  

The outputs tensor from the net call has a shape of **[batch_size, target_count]**.  There is no activation function on the last **nn.Linear** layer in the models, so the outputs are raw numbers between $-\infty$ and $\infty$.  So the assumption is that the loss function will do the squashing of this output (see Criterion below).

The metrics.appendEpochBatchData(ids, outputs) hands off the batch results to the Metrics object.

For train, backprop is done with this method:

    def backProp(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()   


### Metrics

#### Some Metric Background
This class is key to understanding how our model is doing, both on completion and for each epoch.

Looking at multiple values after each epoch is critical in this dataset.  Since most targets are close to 90% negative, chances are you are going to get very high accuracy scores.  The model is trying to predict the positive labels.  A bad model that can't do this very well will still get ~90% of the predictions right.

There are also multiple ways to look at accuracy for multi-label data.  You can say the prediction is accurate only if all 12 targets are correct.  This is a very high bar and doesn't reflect the effectiveness of the model.  If every image correctly identifies 11 out of 12 labels, the accuracy would still be 0% since no image got all 12 correct.  This type of accuracy score is returned with **sklearn.metrics.accuracy_score**.  Note, that if only one target is looked at during training, this accuracy score becomes the traditional accuracy we see with binary classification.

You can also use the number of correct predictions divided by the total number of predictions. This type of accuracy score is returned with **1 - sklearn.metrics.hamming_loss**.  But the Hamming approach gets us back to the 90% accuracy problem we just described above.  

The bottom line is that accuracy is probably not the best way to access the model effectiveness for multi-label classification.  

#### Model probability vs. predictions
We need to be sure of the differences between these two before we discuss the next metric scores.

The CNN model outputs produce one value for each target.  These values can be anything from from $-\infty$ to $\infty$.  So we take these outputs and squash them using the sigmoid function to produce probabilities between 0 and 1.  

To get a prediction of either 1 or 0, we take this probability and check it against a threshold (usually 0.50).  If the prob is >= 0.5, we predict positive or 1.  If below the threshold, we predict negative or 0.

The end result is predictions make a single assertion about an image, i.e. either it is a dog or it is not.  But the predictions don't show how close we got.  i.e. it is not a dog if the prob was 0.49999 which is very close to 0.5.

Probabilities show how close we got. Scores based on probabilities more accurately reflect effectiveness of our model.  But if your desired output is a single assertion, prediction based scores are probably what we will use.



#### Combined Metrics vs. Itemized Metrics
Like we tried to do above with accuracy, we can combine the probabilities and/or predictions of the 12 labels into a single score, i.e. simply show the score 12 times, one for each target.  We tend to prefer a single score since it is easier to make a statement about the effectiveness of the entire model.  But taking some kind of average of these 12 scores can be very misleading.  A bad prediction for a few targets can suggest the model is a poor performer.  

For this reason, the itemized scores will be the primary approach we take in this project for accessing model performance.

#### Four Primary Sores

- Recall (Sensitivity)
- Precision
- ROC AUC
- Average Precision

The first 2 are based only on the predictions and do not take probabilities into account.  
- $Recall = \frac{TP}{TP + FN}$ - What % of the actual positives were found
- $Precision = \frac{TP}{TP + FP}$ - What % of the true predictions were correct

The AUC and Average Precision scores are based on probability. Because the scores use probabilities, they gives us many combinations values as opposed to only the 4 prediction values (TP. TN, FP, FN).  

A TP can have many different probabilities (.12, .45, .78, .98, .87. ...).  These floats allow us to plot both the Receiver Operator Curves (ROC) and the Precision/Recall curves.  To get a value from these curves, we take the area under the curve.  

If our recall is 50%, half the actual positives would be true positives and half would be false negatives.
If all the false negatives have a probability of 0.4999, and all the true positives have a probability of 0.9999, then the ROC AUD and Average Precision would be much greater than 50%.

Since our stated model goal was to try to get the Recall scores, this is the primary score used to determine the model's success.  But we can't just look at a single value.   A good recall with a bad precision means we won't miss any actual positives, but we won't be able to trust these predictions very much.  We can simply make the model always predict Positive for all observations.  This would give us a 100% Recall, but a very low Precision.


### Metrics Class

To see the Metrics modules, please go to:

<a href="modules/lib/Metrics.py" >Metrics</a>

Please go to the supporting notebook, please go to: 

<a href="notebooks/Support%20Notebooks%20for%20Modules/Metrics_NB.ipynb" >Metrics Notebook</a>

<a href="notebooks/Support%20Notebooks%20for%20Modules/Metrics and Training Loop.ipynb" >Metrics and Training Loop Notebook</a>



On instantiation, this class is given the target columns, the actual label values, and optionally the target thresholds.

    def __init__(self, target_columns, train_actual, val_actual, target_thresholds=None, ...

The training loop passes the model outputs after each batch iteration.  It also passes the set of ImageIDs so that these results can be matched back to the original metadata.  

    def appendEpochBatchData(self, ids, outputs, is_validation=False):

This batch data is stored temporarily in a dictionary of ID:Output.  Each batch appends to this dictionary until the epoch is done.  Then the dictionaries for train and val are converted to a dataframe and appended to the prediction and probability histories:

    def closeEpoch(self, epochNumber, is_validation=False):
        if is_validation:
            self.df_val_prediction = self.getPredictionDataFrame(self.epoch_val_predictions)
            self.val_prediction_hx[epochNumber] = self.df_val_prediction.copy()
            self.epoch_val_predictions = {}

            self.df_val_probability = self.getProbilityDataFrame(self.epoch_val_probabilities)
            self.val_probability_hx[epochNumber] = self.df_val_probability.copy()
            self.epoch_val_probabilities = {}
        else:
            self.df_train_prediction = self.getPredictionDataFrame(self.epoch_train_predictions)
            self.train_prediction_hx[epochNumber] = self.df_train_prediction.copy()
            self.epoch_train_predictions = {}

            self.df_train_probability = self.getProbilityDataFrame(self.epoch_train_probabilities)
            self.train_probability_hx[epochNumber] = self.df_train_probability.copy()
            self.epoch_train_probabilities = {}


The Metrics class assumes the final output of the model has not been passed through any activation function (see Criterion below).  So the Metrics class need to generate probabilities and predictions from these raw outputs:

    def getPredictionsFromOutput(self, outputs):
        """
        We are using BCEWithLogitsLoss for our loss
        In this loss function, each label gets the sigmoid (inverse of Logit) before the CE loss
        So our model outputs the raw values on the last FC layer
        This means we have to apply sigmoid to our outputs to squash them between 0 and 1
        We then take values >= .5 as Positive and < .5 as Negative (or optional target thresholds)
        """

        probabilities = torch.sigmoid(outputs.data) 
        predictions = probabilities.clone()
        
        #We need to make sure all tensors are in the same memory space
        if self.target_thresholds.device != predictions.device:
            self.target_thresholds = self.target_thresholds.to(predictions.device)
        
        predictions[predictions >= self.target_thresholds] = 1 # assign 1 label to those with gt or equal to threshold
        predictions[predictions < self.target_thresholds] = 0 # assign 0 label to those with less than threshold   
        
        return probabilities, predictions   

The target thresholds are optional with default values of 0.5:

        if target_thresholds is None:
            #Default thresholds for all targets is .5
            self.target_thresholds = np.ones(len(self.target_columns)) * .5
        else:
            #Custom Thresholds
            #i.e. set Edema threshold to .2.  If Edema probability >= .2, prediction = 1
            self.target_thresholds = target_thresholds

These thresholds are a trick.  This is done entirely in the Metrics class and has nothing to do with the loss function.  So the predictions displayed in this class are affected, but training and weight adjustments are completely unaware of these thresholds.  

At the end of the epoch, the current dataframes are used to build the scores and optionally displayed in the training loop display.  This display method has many different parameters to allow you to pick which items to display:

    def displayMetrics(self, 
                       metricDataSource = MetricDataSource.Both, #train, val, both
                       showCombinedMetrics=True,
                       showMetricDataFrame=True,
                       showROCCurves=True,
                       showPrecisionRecallCurves=True,
                       include_targets=None,
                       combinedAverageMethod='samples',
                       gridSpecColumnCount=4,
                       gridSpecHeight=3,
                       gridSpecWidth=20):

The include_targets parameter lest you display only a subset of the targets.  For example, instead of showing all 12 label scores after each epoch, you can just display the 5 targets identified in the CheXpert competation.

At the end of training, the last epoch results can be displayed.  The display for the epoch and the final results are the same method (displayMetrics), but with different display options passed in.

The Metrics class also has a method (displayEpochProgression) to display plots by epoch.  i.e. display the progression of the recall score plotted against the epoch number.  

    def displayEpochProgression(self, 
                                showResultDataFrames=False,
                                showAccuracyProgression=True,
                                showRecallProgression=True,
                                showPrecisionProgression=True,
                                showF1Progression=True,
                                showROCAUCProgression=True,
                                showAvgPrecisionProgression=True,
                                include_targets=None):



## Standard Training Module <a class="anchor" id="Standard_Training_Module"></a>

This module is the glue for all other modules.  

To see the module, please go to:

<a href="modules/lib/StandardTraining.py" >StandardTraining</a>

Please go to the supporting notebook, please go to: 

<a href="notebooks/Support%20Notebooks%20for%20Modules/StandardTraining_NB.ipynb" >Standard Training Notebook</a>

There are 2 classes in this module, StandardTraining and ModelLoop.

### StandardTraining Class

![alt text](Diagrams/StandardTraining.png "Standard Training Overview") 

Since this is the gateway class, the constructor takes in a lot of optional parameters:

        def __init__(self,   number_images, 
                         batch_size, 
                         learning_rate, 
                         num_epochs,
                         device, 
                         net,
                         epoch_args='standard',
                         use_positivity_weights=False, 
                         image_width = 320,
                         image_height = 320,
                         affineDegrees=5, 
                         translatePrecent=0.05, 
                         shearDegrees=5, 
                         brightnessJitter=0.2, 
                         contrastJitter=0.1, 
                         augPercent=0.2,
                         observation_min_count=None,
                         l2_reg=0,
                         loss_reduction='mean',
                         target_columns=None,
                         target_thresholds=None,
                         save_path=None,
                         net_name=None,
                         net_kwargs=None):
                         
These parameters include options for the data loaders, metrics and training loop classes as well as the model itself.  There are also options needed for the loss function, optimizer and persistence. 

The main purpose of the StandardTraining class is to allow repeatable model runs using the exact same default values.  So if you want to compare 2 runs with and without L2 regularization, you only need to override one parameter.  All the other parameters will be the same since the default values are used.

The basic steps involved are shown below:

![alt text](Diagrams/StandardTrainingSteps.png "Standard Training Steps") 

## Criterion (Loss Function) <a class="anchor" id="Criterion_Optimizer"></a>

**BCEWithLogitsLoss** was chosen for the loss function.  The output of the models are the raw nn.Linear() values.  Some loss functions assume an input that has already been converted to a probability (usually sigmoid or softmax).  Some accept 1 or two values for binary classification and some accept multiple probabilities based on multi-class problems (softmax).  

Here we have a multi-label problem, with each target configured to be binary.  So Binary Cross Entropy would be a good fit, but this needs to be done separately for each target.  So we can't softmax the output of the model since that would predict which target is the best fit and not how well the match was for each target independently.

BCEWithLogitsLoss solves this problem for us.  It takes each value for the targets, applies the sigmoid function to each of the targets and then runs the BCE on each.  So the calculation will be done independently for our 12 labels.

BCEWithLogitsLoss has the ability to reduce these 12 values into a single value (mean or sum) or to output the 12 loss values independently.  

Since we are training all labels concurrently, we will reduce the 12 values into a single loss scalar.  The StandardTraining class allows you to pass in which reduction you want.  Mean is the current default value, but a very good case can be made for sum.  Performing a mean on the 12 values causes some canceling, a label with a large loss can cancel the label with a small loss if both are evenly distanced from the mean.  Sum makes all the labels contribute more to the learning since there is not any cancelling.  But using the sum will make the loss value very large and might require adjustments to other parameters like the learning rate.


## Optimizer

There has been a lot of success with the Adam approach to optimization.  This is mainly achieved by looking at the Hessian matrix and determining the curvature of the hyperplane at the current position of the training parameters.  If the curvature is concave down, we are heading to a trough and we need to slow the learning rate down so we don't jump a critical valley.  If the curvature is flat, we keep the learning rate constant.  If we are near a local maxima, we can increase the lr to quicken learning since we are not near any troughs.  

PyTorch offers L2 regularization directly in it's Adam optimizer.  This is done by the weight_decay parameter.  Though we could manually do regularization after the loss value is obtained, i.e. Matrix norm of various flavors, L1 Lasso, etc., the current framework only offers the L2 λ option which is passed into the Adam weight_decay parameter.


### ModelLoop

This class is simular to the StandardTraining class, but you pass in a collection of different models and parameter and all items are run in the same notebook.  This turned out to be of limited value due to the inability to fully do gargage collection between runs.  But there some a few loop notebooks in the following folder:

<a href="notebooks/ModelLoop" >Model Loop Folder</a>

There is also a support notebook: 

<a href="notebooks/Support%20Notebooks%20for%20Modules/ModelLoop_NB.ipynb" >Model Loop Notebook</a>

# Models  <a class="anchor" id="Models"></a>

## Custom <a class="anchor" id="Custom"></a>

<a href="modules/models/CustomPneumonia.py" >CustomPneumonia.py</a>

#### Note:  The name of the class and module were poorly chosen.  There is no dependency on the Pneumonia label and this model can be used with all 12 labels, any single label, or any combination between.

This is a custom built network done mostly for the experience of manually building a CNN from scratch.

It was first used with a binary classification dataset from Kaggle using pediatric x-rays detecting a single finding of pneumonia.

All the convolution layers follow a simple pattern of:
- 2d Convolution
- Batch Normalization
- Relu
- 2x2 Max Pooling

All the kernel sizes are 3x3 except for the first layer which is 5x5.

The 3 layers are:
1. 1 in / 512 out
2. 512 in / 256 out
3. 256 in / 64 out

After all the pooling, the output "image" size is not 40x40, so the first FC layer takes in 64x40x40.

The first FC layers follow the patterns of:
- Linear
- Relu
- DropOut

The last FC layer outputs the raw scores from nn.Linear

The Fully Connected layers are:
1. 64x40x40 in / 1024 out
2. 1024 in / 512 out
3. 512 in / n out

n is the number of targets used which can be any number from 1 to 12 with our dataset.




#### Summary of CustomPneumonia


```python
net = CustomPneumoniaNN() 
net = nn.DataParallel(net)
net.cuda()
summary(net, (1, 320, 320)) 
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1        [-1, 512, 320, 320]          13,312
           BatchNorm2d-2        [-1, 512, 320, 320]           1,024
             MaxPool2d-3        [-1, 512, 160, 160]               0
                Conv2d-4        [-1, 256, 160, 160]       1,179,904
           BatchNorm2d-5        [-1, 256, 160, 160]             512
             MaxPool2d-6          [-1, 256, 80, 80]               0
                Conv2d-7           [-1, 64, 80, 80]         147,520
           BatchNorm2d-8           [-1, 64, 80, 80]             128
             MaxPool2d-9           [-1, 64, 40, 40]               0
               Linear-10                 [-1, 1024]     104,858,624
              Dropout-11                 [-1, 1024]               0
               Linear-12                  [-1, 512]         524,800
              Dropout-13                  [-1, 512]               0
               Linear-14                    [-1, 1]             513
    CustomPneumoniaNN-15                    [-1, 1]               0
    ================================================================
    Total params: 106,726,337
    Trainable params: 106,726,337
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.39
    Forward/backward pass size (MB): 1019.55
    Params size (MB): 407.13
    Estimated Total Size (MB): 1427.07
    ----------------------------------------------------------------
    

## ResNet <a class="anchor" id="ResNet"></a>

<a href="modules/models/ResNet.py" >ResNet.py</a>

This is a class that uses one of the 5 ResNet models that come with PyTorch.   The 5 layer options offered are [18, 34, 50, 101, 152].  

In the init method, the model is set to one of the pre-built models based on the layers parameter.  Since we are using grayscale images and the models were built for RBG, the first layer is overwritten to take in 1 input channel instead of the hard coded 3 channels.

There is also an option to pass in a drop_out_precent parameter.  If this is passed into the constructor, a hook is registered to the fully connected layers that adds a dropout call to the sequence.

The default number of output channels of the last FC layer is set to 12, but this can be changed to any value between 1 and 12 for this dataset.

There is also a Pre-trained class in this module.  It is almost identical to the above class, but keeps the hard coded 3 input channels.  In the forward method, the grayscale image is run through:

    x = torch.repeat_interleave(input=x, repeats=3, dim=1)
    
This duplicates the grayscale into 3 identical channels to mimic the RGB images the model was trained on.  

Though not required, the image size should be adjusted to 224x224 to match ImageNet.  This can be done with 2 parameters in the StandardTraining class constructor.



## DenseNet <a class="anchor" id="DenseNet"></a>

<a href="modules/models/DenseNet.py" >DenseNet.py</a>

This is a copy of the DenseNet with 5 layers in each DenseBlock using 3 of these blocks.  

The source was obtained from:
<a href="https://towardsdatascience.com/simple-implementation-of-densely-connected-convolutional-networks-in-pytorch-3846978f2f36" >Simple Implementation of Densely Connected Convolutional Networks in PyTorch</a>

Instead of skipping connections like ResNet does, DenseNet concatenates all the previous layers in the block.  Both approaches allow earlier layers to continue to have influence



# Results <a class="anchor" id="Results"></a>

## General Remarks

### Difficulty Increases as Number of Multi-Labels Targets Increases

Picking a single target to train on produces much better results.  Even adding a second target, especially if codependency is low, greatly reduces the predictive value.  If we do all 12 targets, we see a "toggling" of targets:
- Epoch n
    * Good results for target A
    * Poor results for target B
- Epoch n+1
    * Poor results for target A
    * Good results for target B
    
This effect is so dominate that if we were taking this to production, strong consideration should be given to training 12 different models, each with only one target.  When a new image comes in to get a prediction, it would get evaluated 12 times.  A lot of factors would come into play here to determine if this kind of approach would be feasible in a production environment, but one can imagine having the 12 models deployed to the hospital/clinic location so the increased time to do 12 predictions might not be a problem.

### Very Few Epochs were Needed
At the beginning of this project, a lot of work was put into tracking progress across multiple epochs.  There are plots for various scores that show how the metrics progressed from epoch to epoch.

But with over 130,000 images, it looks like most of the training occurred in the first epoch or two.  Most of the models had no problem increasing the training accuracy to the point where Recall, Precision, ROC AUD and Average Precision all showed values in the upper 90% range.  In fact, this was very much a textbook case of overfitting where as training got better, validation got worse.

So most of the model runs used very low epoch counts.  The epoch progression plots do not look very impressive with only a few values on the x-axis, but these still proved valuable since it sometimes gave a good indication of the best number of epochs to use.

### Never Enough Regularization / Fighting Imbalance
This is a combination of the two findings above.  With the "toggle" effect and the overfitting, different ways to add regularization and imbalance compensation to the training was a major effort.  

These included:
- Increased image augmentations on the training data
- Add oversampling to underrepresented target vectors
- Add dropout layers
- Large lambda for L2 regularization
- Positive weights for underrepresented targets

Though these training features were a significant help, it was hard to determine which where needed and which were not that effective.  Thought was given to doing multiple runs to determine this, but due to time constraints, this was not done.  So, most models either included all or none of these features.

Also, even with all of these training features, chances are having 12 different models with a single target each would be more effective.

## Display Metrics of Saved Model Runs
There will be 3 things we look at for results of a model run:  
1. Network name and all run parameters
2. Final results
3. Progression of the metrics at each epoch (might be skipped with low epoch counts)




## Single Target Runs <a class="anchor" id="Single_Target_Runs"></a>

Though the dataset has 12 targets, let's see what things look like with when we are only trying to predict a single target.


### ResNet34_Pleural_Effusion

<a href="notebooks/ModelRuns/ResNet34_Pleural_Effusion.ipynb" >ResNet34_Pleural_Effusion</a>

This run was just against a single target and only used one training feature: drop_out_precent=50%.

- Number of Training Images: 105,338
- Number of Validation Images: 26,410

- Number of Postivies in Training Images: 35,812 (33.9%)
- Number of Postivies in Validation Images: 8,922 (33.7%)


```python
save_name = 'ResNet34_Pleural_Effusion'
path= f'notebooks/ModelRuns/saved/{save_name}/'
StandardTraining.displayRunParameters(path)
```

    Network Name:  ResNet_GrayScale
    Nework Arguments:  layers:34,drop_out_precent:0.5,out_channels:1
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Paramter</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>number_images</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>batch_size</td>
      <td>64</td>
    </tr>
    <tr>
      <td>2</td>
      <td>learning_rate</td>
      <td>1e-05</td>
    </tr>
    <tr>
      <td>3</td>
      <td>num_epochs</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>epoch_args</td>
      <td>standard</td>
    </tr>
    <tr>
      <td>5</td>
      <td>use_positivity_weights</td>
      <td>False</td>
    </tr>
    <tr>
      <td>6</td>
      <td>image_width</td>
      <td>320</td>
    </tr>
    <tr>
      <td>7</td>
      <td>image_height</td>
      <td>320</td>
    </tr>
    <tr>
      <td>8</td>
      <td>affineDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>9</td>
      <td>translatePrecent</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>10</td>
      <td>shearDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>11</td>
      <td>brightnessJitter</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>12</td>
      <td>contrastJitter</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>augPercent</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>14</td>
      <td>observation_min_count</td>
      <td>None</td>
    </tr>
    <tr>
      <td>15</td>
      <td>l2_reg</td>
      <td>0</td>
    </tr>
    <tr>
      <td>16</td>
      <td>loss_reduction</td>
      <td>mean</td>
    </tr>
    <tr>
      <td>17</td>
      <td>target_thresholds</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    Targets:  Pleural_Effusion
    


```python
metrics = StandardTraining.loadMetrics(path) 
metrics.displayMetrics()
```

    
    TRAINING
    
    ▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.756630</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.243370</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.756630</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.858228</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.791262</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.823386</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>True Positives</th>
      <th>Predicted Positives</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1</th>
      <th>ROC AUC</th>
      <th>Avg Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Pleural_Effusion</td>
      <td>35702</td>
      <td>29810</td>
      <td>0.858228</td>
      <td>0.791262</td>
      <td>0.823386</td>
      <td>0.807503</td>
      <td>0.668752</td>
    </tr>
  </tbody>
</table>
</div>


    ***** ROC *****
    


![png](Readme%20Images/output_45_4.png)


    ***** Precision / Recall *****
    


![png](Readme%20Images/output_45_6.png)


    
    VALIDATION
    
    ▔▔▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.737125</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.262875</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.737125</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.954125</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.729667</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.826935</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>True Positives</th>
      <th>Predicted Positives</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1</th>
      <th>ROC AUC</th>
      <th>Avg Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Pleural_Effusion</td>
      <td>9032</td>
      <td>3681</td>
      <td>0.954125</td>
      <td>0.729667</td>
      <td>0.826935</td>
      <td>0.821982</td>
      <td>0.698579</td>
    </tr>
  </tbody>
</table>
</div>


    ***** ROC *****
    


![png](Readme%20Images/output_45_11.png)


    ***** Precision / Recall *****
    


![png](Readme%20Images/output_45_13.png)


#### As you can see, the Recall after the 2 epochs was 95.4% with a Precision of 73%.  The AUC was 81%.
These are pretty good numbers if Recall is the primary goal.  If all we were interested in was predicting Pleural Effusions, we would take this as a very good starting point and concentrate our hyperparameter tuning here.

If we look at the F1 and F2 score, we get 82.7% and 86.5% respectively.  F2 is the weighted harmonic mean giving bias to the Recall score.  Weight Recall with 2 and Weight Precision with 1 (F1 weighs both as 1):

$\frac{2+1}{\frac{2}{0.95412} + \frac{1}{0.72966}} = 0.86538$

## Let's look at DenseNet and the Custom model to see how they compare:

*Note: Both the Custom and DenseNet models run images at 224x244 due to memory constraints.  The ResNet runs at the default image size of 320x320.*

## Custom

<a href="notebooks/ModelRuns/CustomPneumonia_Pleural_Effusion.ipynb" >CustomPneumonia_Pleural_Effusion</a>

*Note:  Model was poorly named "CustomPneumoniaNN".  Pneumonia was named when used with a Kaggle dataset, but does NOT necessarily mean the target is Pneumonia.*


```python
save_name = 'CustomPneumonia_Pleural_Effusion'
path= f'notebooks/ModelRuns/saved/{save_name}/'
StandardTraining.displayRunParameters(path)
metrics = StandardTraining.loadMetrics(path) 

display_args = {'metricDataSource':MetricDataSource.ValidationOnly, 
                  'showCombinedMetrics':True, 
                  'showMetricDataFrame':False, 
                  'showROCCurves':False, 
                  'showPrecisionRecallCurves':False}

metrics.displayMetrics(**display_args)
```

    Network Name:  CustomPneumoniaNN
    Nework Arguments:  out_channels:1,image_size:(224, 224)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Paramter</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>number_images</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>batch_size</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>learning_rate</td>
      <td>1e-05</td>
    </tr>
    <tr>
      <td>3</td>
      <td>num_epochs</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>epoch_args</td>
      <td>standard</td>
    </tr>
    <tr>
      <td>5</td>
      <td>use_positivity_weights</td>
      <td>False</td>
    </tr>
    <tr>
      <td>6</td>
      <td>image_width</td>
      <td>224</td>
    </tr>
    <tr>
      <td>7</td>
      <td>image_height</td>
      <td>224</td>
    </tr>
    <tr>
      <td>8</td>
      <td>affineDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>9</td>
      <td>translatePrecent</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>10</td>
      <td>shearDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>11</td>
      <td>brightnessJitter</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>12</td>
      <td>contrastJitter</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>augPercent</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>14</td>
      <td>observation_min_count</td>
      <td>None</td>
    </tr>
    <tr>
      <td>15</td>
      <td>l2_reg</td>
      <td>0</td>
    </tr>
    <tr>
      <td>16</td>
      <td>loss_reduction</td>
      <td>mean</td>
    </tr>
    <tr>
      <td>17</td>
      <td>target_thresholds</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    Targets:  Pleural_Effusion
    
    VALIDATION
    
    ▔▔▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.733027</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.266973</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.733027</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.861472</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.764220</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.809937</td>
    </tr>
  </tbody>
</table>
</div>


## DenseNet_Pleural_Effusion

<a href="notebooks/ModelRuns/DenseNet_Pleural_Effusion.ipynb" >DenseNet_Pleural_Effusion</a>

The saved folder for this notebook was lost, but you can see the results directly in the notebook above.

Both the Custom and DenseNet produced similar values for Recall, around the 85% range.  It could also be that the reduced image size has an influence.  

But with this and other runs along with the increased time and resources needed for DenseNet and the Custom models, ResNet was mostly used for the rest of the runs!

# 5 Targets <a class="anchor" id="5_Targets"></a>

## ResNet34_Oversample_L2_Sum_PosWeight_5_Targets

<a href="notebooks/ModelRuns/ResNet34_Oversample_L2_Sum_PosWeight_5_Targets.ipynb" >ResNet34_Oversample_L2_Sum_PosWeight_5_Targets</a>

#### Targets:
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion

This run is similar to the Pleural Effusion run from above, but now with 5 targets.  The 5 targets were chosen from the CheXpert competition list of targets.

This run also has most of the training featured implemented:
- **use_positivity_weights** - Parameter of BCEWithLogitsLoss class to increase loss with underrepresented targets
- **affineDegrees** - Degrees of rotation
- **translatePrecent**- Percent the image is moved off center
- **shearDegrees** - Degrees of shearing
- **brightnessJitter**- How much to vary the brightness of the image
- **contrastJitter** - How much to adjust the the contrast level of the image
- **augPercent** - What percent of the training images will get the above image augmentations
- **observation_min_count** - Min count for each target vector
- **l2_reg** - λ multiplication factor applied to L2 Regularization in optimizer
- **loss_reduction** - Parameter of BCEWithLogitsLoss on how to reduce n target losses to a scalar value


```python
save_name = 'ResNet34_Oversample_L2_Sum_PosWeight_5_Targets'
path= f'notebooks/ModelRuns/saved/{save_name}/'
StandardTraining.displayRunParameters(path)
metrics = StandardTraining.loadMetrics(path) 

metrics.displayMetrics()
metrics.displayEpochProgression()
```

    Network Name:  ResNet_GrayScale
    Nework Arguments:  layers:34,drop_out_precent:0.5,out_channels:5
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Paramter</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>number_images</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>batch_size</td>
      <td>64</td>
    </tr>
    <tr>
      <td>2</td>
      <td>learning_rate</td>
      <td>1e-05</td>
    </tr>
    <tr>
      <td>3</td>
      <td>num_epochs</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>epoch_args</td>
      <td>standard</td>
    </tr>
    <tr>
      <td>5</td>
      <td>use_positivity_weights</td>
      <td>True</td>
    </tr>
    <tr>
      <td>6</td>
      <td>image_width</td>
      <td>320</td>
    </tr>
    <tr>
      <td>7</td>
      <td>image_height</td>
      <td>320</td>
    </tr>
    <tr>
      <td>8</td>
      <td>affineDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>9</td>
      <td>translatePrecent</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>10</td>
      <td>shearDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>11</td>
      <td>brightnessJitter</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>12</td>
      <td>contrastJitter</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>augPercent</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>14</td>
      <td>observation_min_count</td>
      <td>150</td>
    </tr>
    <tr>
      <td>15</td>
      <td>l2_reg</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>16</td>
      <td>loss_reduction</td>
      <td>sum</td>
    </tr>
  </tbody>
</table>
</div>


    Targets:  Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural_Effusion
    
    TRAINING
    
    ▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.208979</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.364463</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.635537</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.304625</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.172284</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.206766</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>True Positives</th>
      <th>Predicted Positives</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1</th>
      <th>ROC AUC</th>
      <th>Avg Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Atelectasis</td>
      <td>17402</td>
      <td>24427</td>
      <td>0.242328</td>
      <td>0.172637</td>
      <td>0.201630</td>
      <td>0.522524</td>
      <td>0.171640</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Cardiomegaly</td>
      <td>12805</td>
      <td>67105</td>
      <td>0.865834</td>
      <td>0.165219</td>
      <td>0.277487</td>
      <td>0.707593</td>
      <td>0.228392</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Consolidation</td>
      <td>7637</td>
      <td>43540</td>
      <td>0.705382</td>
      <td>0.123725</td>
      <td>0.210524</td>
      <td>0.720979</td>
      <td>0.186566</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Edema</td>
      <td>25177</td>
      <td>18418</td>
      <td>0.187354</td>
      <td>0.256108</td>
      <td>0.216401</td>
      <td>0.582845</td>
      <td>0.265360</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Pleural_Effusion</td>
      <td>36519</td>
      <td>47585</td>
      <td>0.760207</td>
      <td>0.583419</td>
      <td>0.660183</td>
      <td>0.810004</td>
      <td>0.672999</td>
    </tr>
  </tbody>
</table>
</div>


    ***** ROC *****
    


![png](Readme%20Images/output_51_6.png)


    ***** Precision / Recall *****
    


![png](Readme%20Images/output_51_8.png)


    
    VALIDATION
    
    ▔▔▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.169521</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.426366</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.573634</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.355300</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.169241</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.217143</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>True Positives</th>
      <th>Predicted Positives</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1</th>
      <th>ROC AUC</th>
      <th>Avg Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Atelectasis</td>
      <td>4121</td>
      <td>5562</td>
      <td>0.202863</td>
      <td>0.150306</td>
      <td>0.172674</td>
      <td>0.508370</td>
      <td>0.153481</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Cardiomegaly</td>
      <td>3019</td>
      <td>17700</td>
      <td>0.875124</td>
      <td>0.149266</td>
      <td>0.255032</td>
      <td>0.686749</td>
      <td>0.199851</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Consolidation</td>
      <td>1571</td>
      <td>14899</td>
      <td>0.813495</td>
      <td>0.085778</td>
      <td>0.155191</td>
      <td>0.678652</td>
      <td>0.101998</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Edema</td>
      <td>6109</td>
      <td>8691</td>
      <td>0.363398</td>
      <td>0.255437</td>
      <td>0.300000</td>
      <td>0.585033</td>
      <td>0.255967</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Pleural_Effusion</td>
      <td>8866</td>
      <td>15329</td>
      <td>0.875592</td>
      <td>0.506426</td>
      <td>0.641703</td>
      <td>0.818579</td>
      <td>0.686954</td>
    </tr>
  </tbody>
</table>
</div>


    ***** ROC *****
    


![png](Readme%20Images/output_51_13.png)


    ***** Precision / Recall *****
    


![png](Readme%20Images/output_51_15.png)


    
    ACCURACY
    
    ▔▔▔▔
    


![png](Readme%20Images/output_51_17.png)


    
    RECALL
    
    ▔▔▔
    


![png](Readme%20Images/output_51_19.png)


    
    PRECISION
    
    ▔▔▔▔▔▔
    


![png](Readme%20Images/output_51_21.png)


    
    F1
    
    ▔
    


![png](Readme%20Images/output_51_23.png)


    
    ROC AUC
    
    ▔▔▔▔
    


![png](Readme%20Images/output_51_25.png)


    
    AVERAGE PRECISION
    
    ▔▔▔▔▔▔▔▔
    


![png](Readme%20Images/output_51_27.png)


### As you can, the numbers were not very good

Only 3 of the 5 targets had some progress in training (ROC AUC).  The other 2 targets (Atelectasis,Edema) did not do much better than a 50/50 chance.  You can also see from the epoch progression, neither train of val was able to increase feature performance with more epoch (with the exception of Pleural Effusion).

# All 12 Targets <a class="anchor" id="All_12_Targets"></a>

Let's look at 2 runs.  Both use ResNet34 and have all the training features on.  The first is the normal ResNet and the second uses the Pretrained version.  Let's also show all the metrics available.  Both modes were run with just 3 epochs.

### ResNet34_Oversample_L2_Sum_PosWeight_12_Targets
<a href="notebooks/ModelRuns/ResNet34_Oversample_L2_Sum_PosWeight_12_Targets.ipynb" >ResNet34_Oversample_L2_Sum_PosWeight_12_Targets</a>


```python
save_name = 'ResNet34_Oversample_L2_Sum_PosWeight_12_Targets'
path= f'notebooks/ModelRuns/saved/{save_name}/'
StandardTraining.displayRunParameters(path)
metrics = StandardTraining.loadMetrics(path) 

metrics.displayMetrics()
metrics.displayEpochProgression()
```

    Network Name:  ResNet_GrayScale
    Nework Arguments:  layers:34,drop_out_precent:0.5,out_channels:12
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Paramter</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>number_images</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>batch_size</td>
      <td>64</td>
    </tr>
    <tr>
      <td>2</td>
      <td>learning_rate</td>
      <td>1e-05</td>
    </tr>
    <tr>
      <td>3</td>
      <td>num_epochs</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>epoch_args</td>
      <td>standard</td>
    </tr>
    <tr>
      <td>5</td>
      <td>use_positivity_weights</td>
      <td>True</td>
    </tr>
    <tr>
      <td>6</td>
      <td>image_width</td>
      <td>320</td>
    </tr>
    <tr>
      <td>7</td>
      <td>image_height</td>
      <td>320</td>
    </tr>
    <tr>
      <td>8</td>
      <td>affineDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>9</td>
      <td>translatePrecent</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>10</td>
      <td>shearDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>11</td>
      <td>brightnessJitter</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>12</td>
      <td>contrastJitter</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>augPercent</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>14</td>
      <td>observation_min_count</td>
      <td>150</td>
    </tr>
    <tr>
      <td>15</td>
      <td>l2_reg</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>16</td>
      <td>loss_reduction</td>
      <td>sum</td>
    </tr>
    <tr>
      <td>17</td>
      <td>target_thresholds</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    Targets:  
    
    TRAINING
    
    ▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.070698</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.256169</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.743831</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.559423</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.391042</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.431125</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>True Positives</th>
      <th>Predicted Positives</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1</th>
      <th>ROC AUC</th>
      <th>Avg Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Enlarged_Cardiomediastinum</td>
      <td>28389</td>
      <td>50925</td>
      <td>0.684807</td>
      <td>0.381757</td>
      <td>0.490229</td>
      <td>0.823958</td>
      <td>0.635104</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Cardiomegaly</td>
      <td>39628</td>
      <td>62113</td>
      <td>0.728147</td>
      <td>0.464557</td>
      <td>0.567225</td>
      <td>0.830272</td>
      <td>0.654103</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Lung_Opacity</td>
      <td>88448</td>
      <td>93903</td>
      <td>0.609183</td>
      <td>0.573794</td>
      <td>0.590959</td>
      <td>0.626146</td>
      <td>0.597991</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Lung_Lesion</td>
      <td>27198</td>
      <td>49051</td>
      <td>0.726487</td>
      <td>0.402826</td>
      <td>0.518276</td>
      <td>0.855167</td>
      <td>0.671039</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Edema</td>
      <td>49114</td>
      <td>76852</td>
      <td>0.758725</td>
      <td>0.484880</td>
      <td>0.591652</td>
      <td>0.806052</td>
      <td>0.597335</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Consolidation</td>
      <td>31121</td>
      <td>54526</td>
      <td>0.692555</td>
      <td>0.395279</td>
      <td>0.503298</td>
      <td>0.818559</td>
      <td>0.628081</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Pneumonia</td>
      <td>19396</td>
      <td>42130</td>
      <td>0.755517</td>
      <td>0.347828</td>
      <td>0.476351</td>
      <td>0.879258</td>
      <td>0.667951</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Atelectasis</td>
      <td>43380</td>
      <td>76874</td>
      <td>0.622430</td>
      <td>0.351237</td>
      <td>0.449066</td>
      <td>0.694126</td>
      <td>0.478691</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Pneumothorax</td>
      <td>28357</td>
      <td>56417</td>
      <td>0.712699</td>
      <td>0.358225</td>
      <td>0.476797</td>
      <td>0.824873</td>
      <td>0.606382</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Pleural_Effusion</td>
      <td>71673</td>
      <td>85548</td>
      <td>0.678414</td>
      <td>0.568383</td>
      <td>0.618543</td>
      <td>0.735183</td>
      <td>0.636767</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Pleural_Other</td>
      <td>16013</td>
      <td>34720</td>
      <td>0.832511</td>
      <td>0.383957</td>
      <td>0.525536</td>
      <td>0.930100</td>
      <td>0.741483</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Fracture</td>
      <td>24630</td>
      <td>50700</td>
      <td>0.717702</td>
      <td>0.348659</td>
      <td>0.469322</td>
      <td>0.843582</td>
      <td>0.635057</td>
    </tr>
  </tbody>
</table>
</div>


    ***** ROC *****
    


![png](Readme%20Images/output_54_6.png)


    ***** Precision / Recall *****
    


![png](Readme%20Images/output_54_8.png)


    
    VALIDATION
    
    ▔▔▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.073240</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.210583</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.789417</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.436456</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.263757</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.305644</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>True Positives</th>
      <th>Predicted Positives</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1</th>
      <th>ROC AUC</th>
      <th>Avg Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Enlarged_Cardiomediastinum</td>
      <td>1360</td>
      <td>715</td>
      <td>0.069853</td>
      <td>0.132867</td>
      <td>0.091566</td>
      <td>0.578662</td>
      <td>0.076578</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Cardiomegaly</td>
      <td>2968</td>
      <td>3437</td>
      <td>0.375674</td>
      <td>0.324411</td>
      <td>0.348165</td>
      <td>0.756275</td>
      <td>0.298674</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Lung_Opacity</td>
      <td>12046</td>
      <td>14266</td>
      <td>0.670430</td>
      <td>0.566101</td>
      <td>0.613864</td>
      <td>0.663134</td>
      <td>0.584248</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Lung_Lesion</td>
      <td>1145</td>
      <td>2020</td>
      <td>0.149345</td>
      <td>0.084653</td>
      <td>0.108057</td>
      <td>0.637403</td>
      <td>0.069609</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Edema</td>
      <td>6166</td>
      <td>13171</td>
      <td>0.802952</td>
      <td>0.375902</td>
      <td>0.512075</td>
      <td>0.768149</td>
      <td>0.471081</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Consolidation</td>
      <td>1589</td>
      <td>3638</td>
      <td>0.234739</td>
      <td>0.102529</td>
      <td>0.142720</td>
      <td>0.621928</td>
      <td>0.092922</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Pneumonia</td>
      <td>728</td>
      <td>3257</td>
      <td>0.214286</td>
      <td>0.047897</td>
      <td>0.078294</td>
      <td>0.597343</td>
      <td>0.040977</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Atelectasis</td>
      <td>4168</td>
      <td>8483</td>
      <td>0.410029</td>
      <td>0.201462</td>
      <td>0.270176</td>
      <td>0.594270</td>
      <td>0.196096</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Pneumothorax</td>
      <td>2167</td>
      <td>5071</td>
      <td>0.371943</td>
      <td>0.158943</td>
      <td>0.222713</td>
      <td>0.656757</td>
      <td>0.152650</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Pleural_Effusion</td>
      <td>9032</td>
      <td>12275</td>
      <td>0.723428</td>
      <td>0.532301</td>
      <td>0.613320</td>
      <td>0.760042</td>
      <td>0.587754</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Pleural_Other</td>
      <td>362</td>
      <td>2632</td>
      <td>0.279006</td>
      <td>0.038374</td>
      <td>0.067468</td>
      <td>0.681808</td>
      <td>0.031597</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Fracture</td>
      <td>1184</td>
      <td>4050</td>
      <td>0.288007</td>
      <td>0.084198</td>
      <td>0.130302</td>
      <td>0.645708</td>
      <td>0.082119</td>
    </tr>
  </tbody>
</table>
</div>


    ***** ROC *****
    


![png](Readme%20Images/output_54_13.png)


    ***** Precision / Recall *****
    


![png](Readme%20Images/output_54_15.png)


    
    ACCURACY
    
    ▔▔▔▔
    


![png](Readme%20Images/output_54_17.png)


    
    RECALL
    
    ▔▔▔
    


![png](Readme%20Images/output_54_19.png)


    
    PRECISION
    
    ▔▔▔▔▔▔
    


![png](Readme%20Images/output_54_21.png)


    
    F1
    
    ▔
    


![png](Readme%20Images/output_54_23.png)


    
    ROC AUC
    
    ▔▔▔▔
    


![png](Readme%20Images/output_54_25.png)


    
    AVERAGE PRECISION
    
    ▔▔▔▔▔▔▔▔
    


![png](Readme%20Images/output_54_27.png)


### ResNet34_Pretrained_Oversample_L2_Sum_PosWeight_12_Targets
<a href="notebooks/ModelRuns/ResNet34_Pretrained_Oversample_L2_Sum_PosWeight_12_Targets.ipynb" >ResNet34_Pretrained_Oversample_L2_Sum_PosWeight_12_Targets</a>


```python
save_name = 'ResNet34_Pretrained_Oversample_L2_Sum_PosWeight_12_Targets'
path= f'notebooks/ModelRuns/saved/{save_name}/'
StandardTraining.displayRunParameters(path)
metrics = StandardTraining.loadMetrics(path) 

metrics.displayMetrics()
metrics.displayEpochProgression()
```

    Network Name:  ResNet_PreTrained
    Nework Arguments:  layers:34,drop_out_precent:0.5,out_channels:12
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Paramter</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>number_images</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>batch_size</td>
      <td>64</td>
    </tr>
    <tr>
      <td>2</td>
      <td>learning_rate</td>
      <td>1e-05</td>
    </tr>
    <tr>
      <td>3</td>
      <td>num_epochs</td>
      <td>2</td>
    </tr>
    <tr>
      <td>4</td>
      <td>epoch_args</td>
      <td>standard</td>
    </tr>
    <tr>
      <td>5</td>
      <td>use_positivity_weights</td>
      <td>True</td>
    </tr>
    <tr>
      <td>6</td>
      <td>image_width</td>
      <td>224</td>
    </tr>
    <tr>
      <td>7</td>
      <td>image_height</td>
      <td>224</td>
    </tr>
    <tr>
      <td>8</td>
      <td>affineDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>9</td>
      <td>translatePrecent</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>10</td>
      <td>shearDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>11</td>
      <td>brightnessJitter</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>12</td>
      <td>contrastJitter</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>augPercent</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>14</td>
      <td>observation_min_count</td>
      <td>150</td>
    </tr>
    <tr>
      <td>15</td>
      <td>l2_reg</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>16</td>
      <td>loss_reduction</td>
      <td>sum</td>
    </tr>
    <tr>
      <td>17</td>
      <td>target_thresholds</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    Targets:  
    
    TRAINING
    
    ▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.255856</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.154669</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.845331</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.634481</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.523190</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.549183</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>True Positives</th>
      <th>Predicted Positives</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1</th>
      <th>ROC AUC</th>
      <th>Avg Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Enlarged_Cardiomediastinum</td>
      <td>27792</td>
      <td>40897</td>
      <td>0.814047</td>
      <td>0.553195</td>
      <td>0.658737</td>
      <td>0.921149</td>
      <td>0.819371</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Cardiomegaly</td>
      <td>38874</td>
      <td>51297</td>
      <td>0.819108</td>
      <td>0.620738</td>
      <td>0.706258</td>
      <td>0.920638</td>
      <td>0.825041</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Lung_Opacity</td>
      <td>87780</td>
      <td>89895</td>
      <td>0.690203</td>
      <td>0.673964</td>
      <td>0.681987</td>
      <td>0.766230</td>
      <td>0.760296</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Lung_Lesion</td>
      <td>27539</td>
      <td>37874</td>
      <td>0.842224</td>
      <td>0.612399</td>
      <td>0.709156</td>
      <td>0.941498</td>
      <td>0.859813</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Edema</td>
      <td>49042</td>
      <td>68855</td>
      <td>0.813058</td>
      <td>0.579101</td>
      <td>0.676421</td>
      <td>0.882525</td>
      <td>0.750748</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Consolidation</td>
      <td>30989</td>
      <td>43864</td>
      <td>0.805802</td>
      <td>0.569282</td>
      <td>0.667201</td>
      <td>0.919201</td>
      <td>0.822073</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Pneumonia</td>
      <td>18842</td>
      <td>29252</td>
      <td>0.870608</td>
      <td>0.560782</td>
      <td>0.682164</td>
      <td>0.955870</td>
      <td>0.869501</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Atelectasis</td>
      <td>42634</td>
      <td>63096</td>
      <td>0.727870</td>
      <td>0.491822</td>
      <td>0.587005</td>
      <td>0.842196</td>
      <td>0.712100</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Pneumothorax</td>
      <td>28353</td>
      <td>43242</td>
      <td>0.831235</td>
      <td>0.545026</td>
      <td>0.658370</td>
      <td>0.928227</td>
      <td>0.808028</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Pleural_Effusion</td>
      <td>71427</td>
      <td>79216</td>
      <td>0.793887</td>
      <td>0.715828</td>
      <td>0.752839</td>
      <td>0.879275</td>
      <td>0.834793</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Pleural_Other</td>
      <td>16147</td>
      <td>22079</td>
      <td>0.933362</td>
      <td>0.682594</td>
      <td>0.788521</td>
      <td>0.982883</td>
      <td>0.935948</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Fracture</td>
      <td>25547</td>
      <td>36862</td>
      <td>0.847418</td>
      <td>0.587299</td>
      <td>0.693778</td>
      <td>0.944734</td>
      <td>0.858038</td>
    </tr>
  </tbody>
</table>
</div>


    ***** ROC *****
    


![png](Readme%20Images/output_56_6.png)


    ***** Precision / Recall *****
    


![png](Readme%20Images/output_56_8.png)


    
    VALIDATION
    
    ▔▔▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.109693</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.181582</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.818418</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.444164</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.307509</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.337678</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Target</th>
      <th>True Positives</th>
      <th>Predicted Positives</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>F1</th>
      <th>ROC AUC</th>
      <th>Avg Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Enlarged_Cardiomediastinum</td>
      <td>1364</td>
      <td>2073</td>
      <td>0.188416</td>
      <td>0.123975</td>
      <td>0.149549</td>
      <td>0.619422</td>
      <td>0.093925</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Cardiomegaly</td>
      <td>2956</td>
      <td>4167</td>
      <td>0.503045</td>
      <td>0.356851</td>
      <td>0.417521</td>
      <td>0.797455</td>
      <td>0.373997</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Lung_Opacity</td>
      <td>11989</td>
      <td>12571</td>
      <td>0.617066</td>
      <td>0.588497</td>
      <td>0.602443</td>
      <td>0.679493</td>
      <td>0.601616</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Lung_Lesion</td>
      <td>1087</td>
      <td>2647</td>
      <td>0.263109</td>
      <td>0.108047</td>
      <td>0.153187</td>
      <td>0.680805</td>
      <td>0.091803</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Edema</td>
      <td>6098</td>
      <td>9647</td>
      <td>0.707773</td>
      <td>0.447393</td>
      <td>0.548238</td>
      <td>0.794989</td>
      <td>0.498590</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Consolidation</td>
      <td>1573</td>
      <td>3433</td>
      <td>0.265734</td>
      <td>0.121759</td>
      <td>0.167000</td>
      <td>0.662971</td>
      <td>0.104132</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Pneumonia</td>
      <td>680</td>
      <td>2221</td>
      <td>0.238235</td>
      <td>0.072940</td>
      <td>0.111686</td>
      <td>0.675856</td>
      <td>0.061291</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Atelectasis</td>
      <td>4186</td>
      <td>8653</td>
      <td>0.468705</td>
      <td>0.226742</td>
      <td>0.305631</td>
      <td>0.630723</td>
      <td>0.224188</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Pneumothorax</td>
      <td>2153</td>
      <td>4568</td>
      <td>0.537854</td>
      <td>0.253503</td>
      <td>0.344592</td>
      <td>0.783401</td>
      <td>0.288115</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Pleural_Effusion</td>
      <td>8859</td>
      <td>10972</td>
      <td>0.745457</td>
      <td>0.601896</td>
      <td>0.666028</td>
      <td>0.823539</td>
      <td>0.694592</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Pleural_Other</td>
      <td>377</td>
      <td>857</td>
      <td>0.185676</td>
      <td>0.081680</td>
      <td>0.113452</td>
      <td>0.716468</td>
      <td>0.063173</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Fracture</td>
      <td>1224</td>
      <td>1960</td>
      <td>0.217320</td>
      <td>0.135714</td>
      <td>0.167085</td>
      <td>0.682903</td>
      <td>0.117059</td>
    </tr>
  </tbody>
</table>
</div>


    ***** ROC *****
    


![png](Readme%20Images/output_56_13.png)


    ***** Precision / Recall *****
    


![png](Readme%20Images/output_56_15.png)


    
    ACCURACY
    
    ▔▔▔▔
    


![png](Readme%20Images/output_56_17.png)


    
    RECALL
    
    ▔▔▔
    


![png](Readme%20Images/output_56_19.png)


    
    PRECISION
    
    ▔▔▔▔▔▔
    


![png](Readme%20Images/output_56_21.png)


    
    F1
    
    ▔
    


![png](Readme%20Images/output_56_23.png)


    
    ROC AUC
    
    ▔▔▔▔
    


![png](Readme%20Images/output_56_25.png)


    
    AVERAGE PRECISION
    
    ▔▔▔▔▔▔▔▔
    


![png](Readme%20Images/output_56_27.png)


## Both show pretty good numbers for training, but poor number for validation

The combined recall for all 12 targets was around 44%.  You can also see in the epoch progression that training got better with the second epoch for most targets, but validation's recall has some targets get better and some get worse.

### It did not look like pretrained added any value.  The training time was about the same and the results did not improve.  This was expected since transfer training from ImageNet to grayscale chest x-rays is probably not a good fit.

# 20 Epochs <a class="anchor" id="20_Epochs"></a>

### ResNet with all 12 Targets with 20 Epochs:

<a href="notebooks/ModelRuns/ResNet34_Oversample_L2_Sum_PosWeight_12_Targets_20_Epochs.ipynb" >ResNet34_Oversample_L2_Sum_PosWeight_12_Targets_20_Epochs</a>

This is basically the same ResNet run as above, but with a full 20 epochs.


```python
save_name = 'ResNet34_Oversample_L2_Sum_PosWeight_12_Targets_20_Epochs'
path= f'notebooks/ModelRuns/saved/{save_name}/'
StandardTraining.displayRunParameters(path)
metrics = StandardTraining.loadMetrics(path) 

display_args = {'metricDataSource':MetricDataSource.Both, 
                  'showCombinedMetrics':True, 
                  'showMetricDataFrame':False, 
                  'showROCCurves':False, 
                  'showPrecisionRecallCurves':False}

metrics.displayMetrics(**display_args)

progression_args = {'showResultDataFrames':False, 
                  'showAccuracyProgression':False, 
                  'showRecallProgression':True, 
                  'showPrecisionProgression':True, 
                  'showROCAUCProgression':False, 
                  'showAvgPrecisionProgression':False}

metrics.displayEpochProgression(**progression_args)
```

    Network Name:  ResNet_GrayScale
    Nework Arguments:  layers:34,drop_out_precent:0.5,out_channels:12
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Paramter</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>number_images</td>
      <td>25000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>batch_size</td>
      <td>64</td>
    </tr>
    <tr>
      <td>2</td>
      <td>learning_rate</td>
      <td>1e-05</td>
    </tr>
    <tr>
      <td>3</td>
      <td>num_epochs</td>
      <td>20</td>
    </tr>
    <tr>
      <td>4</td>
      <td>epoch_args</td>
      <td>standard</td>
    </tr>
    <tr>
      <td>5</td>
      <td>use_positivity_weights</td>
      <td>True</td>
    </tr>
    <tr>
      <td>6</td>
      <td>image_width</td>
      <td>320</td>
    </tr>
    <tr>
      <td>7</td>
      <td>image_height</td>
      <td>320</td>
    </tr>
    <tr>
      <td>8</td>
      <td>affineDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>9</td>
      <td>translatePrecent</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>10</td>
      <td>shearDegrees</td>
      <td>5</td>
    </tr>
    <tr>
      <td>11</td>
      <td>brightnessJitter</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>12</td>
      <td>contrastJitter</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>augPercent</td>
      <td>0.4</td>
    </tr>
    <tr>
      <td>14</td>
      <td>observation_min_count</td>
      <td>150</td>
    </tr>
    <tr>
      <td>15</td>
      <td>l2_reg</td>
      <td>0.1</td>
    </tr>
    <tr>
      <td>16</td>
      <td>loss_reduction</td>
      <td>sum</td>
    </tr>
    <tr>
      <td>17</td>
      <td>target_thresholds</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


    Targets:  
    
    TRAINING
    
    ▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.526673</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.082643</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.917357</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.768183</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.697907</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.713543</td>
    </tr>
  </tbody>
</table>
</div>


    
    VALIDATION
    
    ▔▔▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.073999</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.193756</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.806244</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.415355</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.305286</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.326690</td>
    </tr>
  </tbody>
</table>
</div>


    
    RECALL
    
    ▔▔▔
    


![png](Readme%20Images/output_59_7.png)


    
    PRECISION
    
    ▔▔▔▔▔▔
    


![png](Readme%20Images/output_59_9.png)


    
    F1
    
    ▔
    


![png](Readme%20Images/output_59_11.png)


# Target Competition

The final scores were not very good.  With 20 epochs, you see the classical overfitting.  Even with over 130,000 images, it looks like the ResNet was able to memorize the dataset.  Almost all of the training Recall and ROC AUC values were in the 90% range.  

## Epoch Progression

The plots of the Epoch Progression are **extremely** telling here!  You can easily see how the training metric improves with more epochs.  But the validation sets very nicely shows the "toggling" effect.

## In order to get gains from one target, another target must sacrifice!

This makes sense.  We are training all 12 labels with the same set of weights.  Some of the diagnoses look for very different image features.  Several targets like  Lung_Opacity, Consolidation, Pneumonia and Atelectasis look for patterns in the air spaces of the lungs.  Targets like Cardiomegaly, Pneumothorax and Fracture look for silhouette shapes.

The Recall based metrics show a much strong "toggling" effect whereas Precision based metrics tend to be more stable.


# 5 Independent Models  <a class="anchor" id="5_Independent_Models"></a>

### Below are the Metrics for 5 Model runs, each with single target


```python
display_args = {'metricDataSource':MetricDataSource.ValidationOnly, 
                  'showCombinedMetrics':True, 
                  'showMetricDataFrame':False, 
                  'showROCCurves':False, 
                  'showPrecisionRecallCurves':False}

for save_name in ['ResNet34_Atelectasis', 
                  'ResNet34_Cardiomegaly', 
                  'ResNet34_Consolidation', 
                  'ResNet34_Edema', 
                  'ResNet34_Pleural_Effusion']:
    target_name = save_name.replace('ResNet34_', '')
    print(u"\u2583" * 20 + f'\n{target_name}\n' + u"\u2594" * 20)
    path= f'notebooks/ModelRuns/saved/{save_name}/'
    metrics = StandardTraining.loadMetrics(path) 
    metrics.displayMetrics(**display_args)
```

    ▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃
    Atelectasis
    ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
    
    VALIDATION
    
    ▔▔▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.844359</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.155641</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.844359</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.844353</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.915609</td>
    </tr>
  </tbody>
</table>
</div>


    ▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃
    Cardiomegaly
    ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
    
    VALIDATION
    
    ▔▔▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.879555</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.120445</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.879555</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.964325</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.905884</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.934191</td>
    </tr>
  </tbody>
</table>
</div>


    ▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃
    Consolidation
    ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
    
    VALIDATION
    
    ▔▔▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.940857</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.059143</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.940857</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.940857</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.969527</td>
    </tr>
  </tbody>
</table>
</div>


    ▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃
    Edema
    ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
    
    VALIDATION
    
    ▔▔▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.777065</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.222935</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.777065</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.995874</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.777842</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.873457</td>
    </tr>
  </tbody>
</table>
</div>


    ▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃
    Pleural_Effusion
    ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
    
    VALIDATION
    
    ▔▔▔▔▔▔▔
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score for all Targets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy Score</td>
      <td>0.737125</td>
    </tr>
    <tr>
      <td>Hamming Loss</td>
      <td>0.262875</td>
    </tr>
    <tr>
      <td>Hamming Accuracy</td>
      <td>0.737125</td>
    </tr>
    <tr>
      <td>Combined Recall</td>
      <td>0.954125</td>
    </tr>
    <tr>
      <td>Combined Precision</td>
      <td>0.729667</td>
    </tr>
    <tr>
      <td>Combined F1</td>
      <td>0.826935</td>
    </tr>
  </tbody>
</table>
</div>


### As we can see from the results, all the scores of these independant model runs are much better then when we run a single model predicting all 5 targets


# Conclusions <a class="anchor" id="Conclusions"></a>

## The Model was not Super Critical

ResNet looked to be somewhat better than DenseNet and the custom model, but not overwhelmingly so.  This is not to say if more time was spent building different short circuits or dense blocks, we could get better performance.  But the main issue was not the ability to train.  In fact, overtraining was one of the major stumbling blocks.

##  Regularization, Image Augmentation, Oversampling Critical, but NOT Enough

The results of adding these training features was obviously very helpful, but it was not enough to overcome the competition between targets.

## Model's were able to Quickly Achieve an Overtrained State
This was one of the main reasons for only using 1 or 2 epochs.

## 12 Targets, 1 Set of Weights

It looks like training 12 separate models, one for each target might be the best way to go.  But there might be other options within a Model architecture that prevents sharing weights between targets.  Perhaps not using reduction with our BCEWithLogitsLoss loss function might be a start in the right direction.


# Future Areas of Exploration <a class="anchor" id="Future_Areas_of_Exploration"></a>

### Model Architectures
Because the emphasis was placed on understanding and building a framework, not as much time was spent exploring different model architectures.  A very good understanding was obtained from working with ResNet and DenseNet, but due to time constraints, no attempts were made to modify or enhance these two architectures.  

### TensorBoard
Initially, the plan was to use TensorBoard in this project.  But it was quickly found that a deeper background into the metrics associated with CNNs was needed.  In fact, this was the reason for making the emphasis on building a framework.  But there are features available in TensorBorad that would have been nice to see such as the visualization of filters and feature maps.

### Productionization
Due to time constraints, exploring the productionization of models was not done.  It would be nice to have set up an endpoint on AWS that you could post an x-ray to and return the model predictions.  Exploring how 12 separate models, one for each target, could be melded into a single API call and how updates to the end point could be made would have been a good task to take on.

### A more systematic approach to parameter tuning
An attempt was made to do this more programmatically with the ModelLoop class.  But this was of limited help due to resource contention running multiple models in the same Python kernel.  Perhaps some kind of external code, i.e. C#, could be used to start and tear down Python kernels so a set of models and/or parameters could be run unattended.

### Post Training EDA
If time permitted, examples of False Positives and False Negatives should have been explored.  By sampling some of these images, patterns might be seen as to why some of these misclassifications occured.  i.e. False Positives might have been due to in-patient leaks from ECG leads or preexisting Cardiomegalia.

# Should Haves
As will all projects, looking back as what went good and what was not so good is always important.  Some of the things that were missed, that could be cleaned up with a refactor could be:

- Docstrings
- Unit Tests
- Refactoring of Classes
- Better organization of folders
- Better comments in the supporting notebooks
- Better visualizations of the EDA
- Add Radiology Terminology and examples of each diagnosis
- More efficient Readme (sorry, tl:dr)


Also missing were better explanations of ResNet and DenseNet.  Some educational notebooks would have been great to add.


# Final Thoughts <a class="anchor" id="Final_Thoughts"></a>

This journey was extremely helpful.  At the beginning of the project, a huge amount of time was spent trying to figure out things like:
- Why shapes between layers or in metrics did not match up
- Differences between Binary Classification, multi-class and multi-label problems
- What loss functions where used and why the loss function didn't mate with the model outputs
- etc, etc, etc

But by the end of the project, most if not all of these types of things became second nature.  Deciding to extend the framework to allow only a single target was a relatively late addition to the project.  But with the experience gained, this was a quick change with only minimal time spent figuring out any bugs.  

In hindsight, this became the goal.  To gain the understanding and experience so that tackling these types of problems in the real world will be a much more efficient and effective process.

Also, the two educational notebooks proved to be invaluable.  Understanding some of the internals of PyTorch and the fundamentals of Neural Networks gave the needed background to approach this type of problem solving.

