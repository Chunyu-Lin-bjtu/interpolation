### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torchvision.models as models

##############################################################################
# context extractor
##############################################################################
class ContextGenerator(nn.Module):
    def __init__(self,requires_grad=False):
        super(ContextGenerator, self).__init__()
	
	res_model = models.resnet18(pretrained=True)
	res_model.conv1=nn.Conv2d (3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
	self.model = nn.Sequential(*list(res_model.children())[0:1])
	'''
	if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
	'''
	
    def forward(self, input):
	feature_conv1=self.model(input)

        return feature_conv1



