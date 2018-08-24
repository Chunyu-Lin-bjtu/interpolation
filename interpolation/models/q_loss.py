### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np



# grid network
class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

	self.loss = nn.L1Loss()

    def forward(self,x,y):

	loss=self.loss(x, y.detach())

        return loss


class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()

	self.vgg = Vgg19().cuda()
	self.loss = nn.MSELoss()

    def forward(self,x,y):

	x_vgg, y_vgg = self.vgg(x), self.vgg(y)
	loss=self.loss(x_vgg, y_vgg.detach())

        return loss

class LaplacianLoss(nn.Module):
    def __init__(self,requires_grad=False):
        super(LaplacianLoss, self).__init__()

	self.criterion = nn.L1Loss()
	self.lap=LaplacianPyramid()
	self.weights = [1, 2, 4, 8, 16]
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self,x,y):


	x_lap, y_lap = self.lap(x), self.lap(y)
	for i in range(len(x_lap)):
            loss = self.weights[i] * self.criterion(x_lap[i], y_lap[i].detach())    

        return loss


def makeGaussian(size, fwhm=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    gaussiankernel = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

    return gaussiankernel


def gaussianConv(inputValue,batchSize=1, channel=3,kernel_size=5):
    g = makeGaussian(kernel_size)
    kernel = torch.FloatTensor(g)
    kernel = torch.stack([kernel for i in range(channel)])
    batched_gaussian = Variable(torch.stack([kernel for i in range(batchSize)])).cuda()
    out = torch.nn.functional.conv2d(inputValue, batched_gaussian,padding=2)

    return out


def upsample2(input):
    inputValue=input.data
    inputValue_size = inputValue.size()
    row_upsample = torch.zeros(inputValue_size[0], inputValue_size[1],2*inputValue_size[2],inputValue_size[3]).cuda()
    col_upsample = torch.zeros(inputValue_size[0], inputValue_size[1],2*inputValue_size[2],2*inputValue_size[3]).cuda()
    even1 = []
    for i in range(2*inputValue_size[2]):
    	if i%2 ==0:
	    even1.append(i)
    index1=torch.LongTensor(even1).cuda()
    for i in  range(inputValue_size[0]):
	for j in range(inputValue_size[1]):
    		row_upsample[i][j].index_copy_(0, index1, inputValue[i][j])

    even2 = []
    for i in range(2*inputValue_size[3]):
    	if i%2 ==0:
	    even2.append(i)
    index2=torch.LongTensor(even2).cuda()
    for i in  range(inputValue_size[0]):
	for j in range(inputValue_size[1]):
    		col_upsample[i][j].index_copy_(1, index2, row_upsample[i][j])

    result=Variable(col_upsample)
    return result


def downsample2(input):
    inputValue=input.data
    inputValue_size = inputValue.size()
    if inputValue_size[2]%2 == 0:
	width=inputValue_size[2]/2
	height=inputValue_size[3]/2
    else:
	width=inputValue_size[2]/2 + 1
	height=inputValue_size[3]/2 + 1
    row_downsample = torch.zeros(inputValue_size[0], inputValue_size[1],width,inputValue_size[3])
    col_downsample = torch.zeros(inputValue_size[0], inputValue_size[1],width,height)
    even1 = []
    for i in range(inputValue_size[2]):
    	if i%2 ==0:
	    even1.append(i)
    index1=torch.LongTensor(even1).cuda()
    for i in  range(inputValue_size[0]):
	for j in range(inputValue_size[1]):
    		row_downsample[i][j]=torch.index_select(inputValue[i][j], 0, index1)

    even2 = []
    for i in range(inputValue_size[3]):
    	if i%2 ==0:
	    even2.append(i)
    index2=torch.LongTensor(even2)
    for i in  range(inputValue_size[0]):
	for j in range(inputValue_size[1]):
    		col_downsample[i][j]=torch.index_select(row_downsample[i][j], 1, index2)

    result=Variable(col_downsample).cuda()
    return result

class LaplacianPyramid(torch.nn.Module):
    def __init__(self):
        super(LaplacianPyramid, self).__init__()

    def forward(self, X):
        gau1=gaussianConv(X,batchSize=3, channel=3,kernel_size=5)
	gau_pyramid1=downsample2(gau1)

	gau2=gaussianConv(gau_pyramid1,batchSize=3, channel=3,kernel_size=5)
	gau_pyramid2=downsample2(gau2)

	gau3=gaussianConv(gau_pyramid2,batchSize=3, channel=3,kernel_size=5)
	gau_pyramid3=downsample2(gau3)

	gau4=gaussianConv(gau_pyramid3,batchSize=3, channel=3,kernel_size=5)
	gau_pyramid4=downsample2(gau4)

	gau5=gaussianConv(gau_pyramid4,batchSize=3, channel=3,kernel_size=5)
	gau_pyramid5=downsample2(gau5)


	###

        laplacian_up1=upsample2(gau_pyramid1)
	laplacian_conv1=gaussianConv(laplacian_up1,batchSize=1, channel=3,kernel_size=5)
	laplacian_pyramid1=X-laplacian_conv1

        laplacian_up2=upsample2(gau_pyramid2)
	laplacian_conv2=gaussianConv(laplacian_up2,batchSize=1, channel=3,kernel_size=5)
	laplacian_pyramid2=gau_pyramid1-laplacian_conv2

        laplacian_up3=upsample2(gau_pyramid3)
	laplacian_conv3=gaussianConv(laplacian_up3,batchSize=1, channel=3,kernel_size=5)
	laplacian_pyramid3=gau_pyramid2-laplacian_conv3

        laplacian_up4=upsample2(gau_pyramid4)
	laplacian_conv4=gaussianConv(laplacian_up4,batchSize=1, channel=3,kernel_size=5)
	laplacian_pyramid4=gau_pyramid3-laplacian_conv4

        laplacian_up5=upsample2(gau_pyramid5)
	laplacian_conv5=gaussianConv(laplacian_up5,batchSize=1, channel=3,kernel_size=5)
	laplacian_pyramid5=gau_pyramid4-laplacian_conv5

	
        return [laplacian_pyramid1,laplacian_pyramid2,laplacian_pyramid3,laplacian_pyramid4,laplacian_pyramid5]


from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)                      
        out = h_relu4
        return out





