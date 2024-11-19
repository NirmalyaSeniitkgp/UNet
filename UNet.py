# Implementation of UNet
# This Program is written by:
# Dr. Nirmalya Sen
# PhD (IIT Kharagpur)

import torch
import torch.nn as nn
import torchvision.transforms as transforms


# Implementation of Double Convolution Operation
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x
    

# Implementation of Down Sampling Operation
class DownSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        conv_output = self.conv(x)
        pool_output = self.pool(conv_output)
        return (conv_output, pool_output)
    

# Implementation of Up Sampling Operation
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2,bias=False)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, y):
        x = self.up(x)
        if x.shape!=y.shape:
            p = (x.shape[-2],x.shape[-1])
            transform1 = transforms.CenterCrop(p)
            y = transform1(y)
        z = torch.cat([x,y], dim=1)
        z = self.conv(z)
        return z


# Implementation of UNet
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels,64)
        self.down_convolution_2 = DownSample(64,128)
        self.down_convolution_3 = DownSample(128,256)
        self.down_convolution_4 = DownSample(256,512)

        self.bottle_neck = DoubleConv(512,1024)

        self.up_convolution_1 = UpSample(1024,512)
        self.up_convolution_2 = UpSample(512,256)
        self.up_convolution_3 = UpSample(256,128)
        self.up_convolution_4 = UpSample(128,64)

        self.out = nn.Conv2d(64,out_channels=num_classes,kernel_size=1)
    
    def forward(self, x):
        conv_output_1, pool_output_1 = self.down_convolution_1(x)
        conv_output_2, pool_output_2 = self.down_convolution_2(pool_output_1)
        conv_output_3, pool_output_3 = self.down_convolution_3(pool_output_2)
        conv_output_4, pool_output_4 = self.down_convolution_4(pool_output_3)

        bottle_neck_output = self.bottle_neck(pool_output_4)

        expand_output_1 = self.up_convolution_1(bottle_neck_output, conv_output_4)
        expand_output_2 = self.up_convolution_2(expand_output_1, conv_output_3)
        expand_output_3 = self.up_convolution_3(expand_output_2, conv_output_2)
        expand_output_4 = self.up_convolution_4(expand_output_3, conv_output_1)

        output_segmentation_map = self.out(expand_output_4)
        return output_segmentation_map



# Testing the UNet model
# Here, we have taken input channels = 1 and
# number of classes = 2 as given in the Original Paper.

model = UNet(1,2)

# When we use padding=0 in Double Convolution Operation
# As given in the Original Paper
# Then for input size of (5,1,572,572)
# The output size will be (5,2,388,388)

# However, When we use padding=1 in Double Convolution Operation
# Then for input size of (5,1,572,572)
# The output size will be (5,2,560,560)

input = torch.randn(5,1,572,572)
print(input.shape)
print('*'*25)
output = model(input)
print(output.shape)

