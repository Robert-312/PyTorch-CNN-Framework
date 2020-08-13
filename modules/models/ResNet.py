import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNet_GrayScale(nn.Module):
    
    """
    docstring
    """

    def __init__(self,  layers= 18, in_channels=1, out_channels=12, drop_out_precent=None):
        super(ResNet_GrayScale, self).__init__()

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
        
        if drop_out_precent is not None:
            self.dropout2d = nn.Dropout2d(p=drop_out_precent)
            self.model.fc.register_forward_hook(lambda m, inp, out: self.dropout2d(out))

        self.final_fc = nn.Linear(1000, out_channels)

    def forward(self, x):
        x = self.model(x)
        return self.final_fc(x)
    
class ResNet_PreTrained(nn.Module):
    
    """
    In order to use pretrained, we have to have 3 input channels.
    
    Since our data loaders produce a grayscale batch of shape (channels, height, width),
    we need to duplicate the 1 channel into 3 channels.
    
    We do this on the forward using the repeat function.
    """

    def __init__(self,  layers= 18, out_channels=12, drop_out_precent=None):
        super(ResNet_PreTrained, self).__init__()

        # bring resnet
        if layers==18:
            self.model = torchvision.models.resnet18(pretrained=True)
        if layers==34:
            self.model = torchvision.models.resnet34(pretrained=True)
        if layers==50:
            self.model = torchvision.models.resnet50(pretrained=True)
        if layers==101:
            self.model = torchvision.models.resnet101(pretrained=True)
        if layers==152:
            self.model = torchvision.models.resnet152(pretrained=True)

       
        if drop_out_precent is not None:
            self.model.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=drop_out_precent, training=m.training))

        self.final_fc = nn.Linear(1000, out_channels)

    def forward(self, x):
        # Repeat the single channel 3 times to mimic RBG
        # shape is (channels, height, width), so the dim we want to repeat is 1
        x = torch.repeat_interleave(input=x, repeats=3, dim=1)
        
        
        x = self.model(x)
        return self.final_fc(x)