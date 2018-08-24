import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable

model = torch.load('see_model.pth')
params=model.state_dict()
for k,v in params.items():
	print(k)

print(params['synthesis_network.upsample_model_03.2.bias'])

    


