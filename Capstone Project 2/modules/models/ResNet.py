import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNet_GrayScale_12_Out(nn.Module):

    def __init__(self,  layers= 18, in_channels=1):
        super(ResNet_GrayScale_12_Out, self).__init__()

        # bring resnet
        if layers==18:
            self.model = torchvision.models.resnet18()
        if layers==34:
            self.model = torchvision.models.resnet34()
        if layers==50:
            self.model = torchvision.models.resnet50()
        if layers==101:
            self.model = torchvision.models.resnet101()
        if layers==152:
            self.model = torchvision.models.resnet152()

        # original definition of the first layer on the renset class (same for all versions)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # your case
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.final_fc = nn.Linear(1000, 12)

    def forward(self, x):
        x= self.model(x)
        return self.final_fc(x)