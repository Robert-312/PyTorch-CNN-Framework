import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

class CustomPneumoniaNN(nn.Module):
    def __init__(self, out_channels=1):
        super(CustomPneumoniaNN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.out_channels = out_channels

        #Input image size = 320x320
 
        # 1st Convolution 512 filters each with 1 kernel (channel) with each kernel 5x5=25:  512*1*25 + 512(biases) = 13,312 trainable parameters
        # Padding: With strid=1, P = (Filter - 1)/2 = (5-1)/2 = 2.  
        # If the 5x5 moves across the image 320 times, the last position will nohang off by 4 pixels, so pad 2 on each side to make it even
        # This keeps the output image 320x320
        #Note: Each batch norm will have 2 trainable parameters per input channel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(512) # 1024 trainable parameters
        #relu
        #max pool 2x2 This will make image size 160x160

        # 2nd Convolution 256 filters each with 512 kernels (input channels) with each kernel 3x3=9:  256*512*9 + 256 = 1,179,904 trainable parameters
        # Padding: With strid=1, P = (Filter - 1)/2 = (3-1)/2 = 1
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(256) # 512 trainable parameters
        #relu
        #max pool 2x2 This will make image size 80/80

        # 3rd Convolution 64 filters each with 256 kernels (input channels) with each kernel 3x3=9:  64*256*9 + 64 = 147,520 trainable parameters
        # Padding: With strid=1, P = (Filter - 1)/2 = (3-1)/2 = 1
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64) # 128 trainable parameters
        #relu
        #max pool 2x2 This will make image size 40/40

        # Fully Connected Vector: unraveling this last 40x40 image gives use 1680 values for our vector
        #  But we need to do this for each output channel so 64x40x40 = 107,520 values for our flattened vector
        self.flattened_length_ = 64*40*40

        # FC1 This is where we can really accumulate trainable parameters
        # Linear: for each filter (output channel), will will have over 100k slopes and 1 y-intercept
        # So, ~ 50k params for each output channel
        # Lets target around 100 million params, so ~1,000 output channels.  We will use 1024 to keep with our powers of 2 convention
        # m=1024x(64*40*40), b=1024 = 1024x64x40x40 + 1024 = 104,858,624 trainable parameters
        self.fc1 = nn.Linear(self.flattened_length_, 1024)

        # FC2 2 Let's shoot for a reuction to 1/2 the size of the input vector (512 out)
        # m=512x(1024), b=512 = 524,800 trainable parameters
        self.fc2 = nn.Linear(1024, 512)
        
        # FC3 2 output channels, this need to match the number of labels we have (binary in this case)
        # 2x1014 slopes and 2 y-intercepts = 2050 trainable parameters
        # The softmax (or log softmax for Cross Entrypy) is done in the loss function to detmine the probability for each class
        self.fc3 = nn.Linear(512, self.out_channels)

        # Regularization:  Dropout is kind of like ensemble component of Random Forest
        #  RF will randomly pick some features for each tree in the ensemble
        #  This makes sure that some of the less impactful features get a chance to contribute
        #  Here, we will randomly zero out 20% of the flattened vector's weights
        #  This will be done with each mini-batch
        #  The goal is to prevent overfitting or memorizing the trainging data
        self.dropout = nn.Dropout(0.5) 

         
    def forward(self, x):

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.pool(x)
       
        x = x.view(-1, self.flattened_length_)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
      
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x