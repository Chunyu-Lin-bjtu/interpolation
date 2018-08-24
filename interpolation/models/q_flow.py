### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

gpu_mod = True

if gpu_mod:
    from resample2d_package.modules.resample2d import Resample2d
else:
    from resample2d_package_cpu.modules.resample2d import Resample2d


##############################################################################
# flow calculate
##############################################################################
class FlowGenerator(nn.Module):
    def __init__(self):
        super(FlowGenerator, self).__init__()
        activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
	self.resample1 = Resample2d()     
	self.upsample_2_factor = nn.Upsample(scale_factor=2, mode='bilinear')
	
	
	### feature pyramid extractor network
	pyramid2 = [nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),activation]
	pyramid2 += [nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),activation]
	self.pyramid2 = nn.Sequential(*pyramid2)
	pyramid3 = [nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),activation]
	pyramid3 += [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),activation]
	self.pyramid3 = nn.Sequential(*pyramid3)
	pyramid4 = [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),activation]
	pyramid4 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),activation]
	self.pyramid4 = nn.Sequential(*pyramid4)
	pyramid5 = [nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),activation]
	pyramid5 += [nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),activation]
	self.pyramid5 = nn.Sequential(*pyramid5)
	pyramid6 = [nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),activation]
	pyramid6 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),activation]
	self.pyramid6 = nn.Sequential(*pyramid6)
	pyramid7 = [nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),activation]
	pyramid7 += [nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),activation]
	self.pyramid7 = nn.Sequential(*pyramid7)

	### optical flow estimator network
	flowestimator7 = [nn.Conv2d(386, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator7 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator7 += [nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),activation]
	flowestimator7 += [nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),activation]
	flowestimator7 += [nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),activation]
	flowestimator7 += [nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)]
	self.flowestimator7 = nn.Sequential(*flowestimator7)

	flowestimator6 = [nn.Conv2d(258, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator6 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator6 += [nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),activation]
	flowestimator6 += [nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),activation]
	flowestimator6 += [nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),activation]
	flowestimator6 += [nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)]
	self.flowestimator6 = nn.Sequential(*flowestimator6)

	flowestimator5 = [nn.Conv2d(194, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator5 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator5 += [nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),activation]
	flowestimator5 += [nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),activation]
	flowestimator5 += [nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),activation]
	flowestimator5 += [nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)]
	self.flowestimator5 = nn.Sequential(*flowestimator5)

	flowestimator4 = [nn.Conv2d(130, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator4 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator4 += [nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),activation]
	flowestimator4 += [nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),activation]
	flowestimator4 += [nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),activation]
	flowestimator4 += [nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)]
	self.flowestimator4 = nn.Sequential(*flowestimator4)

	flowestimator3 = [nn.Conv2d(66, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator3 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator3 += [nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),activation]
	flowestimator3 += [nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),activation]
	flowestimator3 += [nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),activation]
	flowestimator3 += [nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)]
	self.flowestimator3 = nn.Sequential(*flowestimator3)

	flowestimator2_f2 = [nn.Conv2d(34, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator2_f2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),activation]
	flowestimator2_f2 += [nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),activation]
	flowestimator2_f2 += [nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),activation]
	flowestimator2_f2 += [nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),activation]
	self.flowestimator2 = nn.Sequential(nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1))
	self.flowestimator2_f2 = nn.Sequential(*flowestimator2_f2)

	### context network
	context = [nn.Conv2d(34, 128, kernel_size=3, stride=1, padding=1,dilation=1,bias=False),activation]
	context += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2,dilation=2,bias=False),activation]
	context += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4,dilation=4,bias=False),activation]
	context += [nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=8,dilation=8,bias=False),activation]
	context += [nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=16,dilation=16,bias=False),activation]
	context += [nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1,dilation=1,bias=False),activation]
	context += [nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1,dilation=1,bias=False)]
	self.context = nn.Sequential(*context)
		
    def forward(self, input1,input2):
	feature1_pyramid2=self.pyramid2(input1)
	feature2_pyramid2=self.pyramid2(input2)
	feature1_pyramid3=self.pyramid3(feature1_pyramid2)
	feature2_pyramid3=self.pyramid3(feature2_pyramid2)

	feature1_pyramid4=self.pyramid4(feature1_pyramid3)
	feature2_pyramid4=self.pyramid4(feature2_pyramid3)

	feature1_pyramid5=self.pyramid5(feature1_pyramid4)
	feature2_pyramid5=self.pyramid5(feature2_pyramid4)

	feature1_pyramid6=self.pyramid6(feature1_pyramid5)
	feature2_pyramid6=self.pyramid6(feature2_pyramid5)

	feature1_pyramid7=self.pyramid7(feature1_pyramid6)
	feature2_pyramid7=self.pyramid7(feature2_pyramid6)


	####
	#inputflow_shape=feature2_pyramid7.size()
	#inputflow = Variable(torch.zeros(inputflow_shape[0],2,inputflow_shape[2],inputflow_shape[3]),requires_grad=False).cuda()
        #inputflow_shape=feature2_pyramid7.size()  ###1*192*8*8
        inputflow = feature2_pyramid7[:,0:2,:,:]
        if gpu_mod:
            inputflow.detach()
        inputflow = inputflow - inputflow
	
	feature7_warp = self.resample1(feature2_pyramid7, inputflow)
	feature7_cat=torch.cat((feature1_pyramid7, feature7_warp,inputflow), dim=1)
	flow7=self.flowestimator7(feature7_cat)


	flow7_upsample=self.upsample_2_factor(flow7)
	feature6_warp = self.resample1(feature2_pyramid6, flow7_upsample)
	feature6_cat=torch.cat((feature1_pyramid6, feature6_warp,flow7_upsample), dim=1)
	flow6=self.flowestimator6(feature6_cat)  ###diff: min: 0.0, max: 2.20537185669e-06, mean: 2.6004636311e-07, median: 7.82310962677e-08

	flow6_upsample=self.upsample_2_factor(flow6) ###diff: min: 0.0, max: 0.0156976226717, mean: 0.001012046705, median: 2.14204192162e-07
	feature5_warp = self.resample1(feature2_pyramid5, flow6_upsample)
	feature5_cat=torch.cat((feature1_pyramid5, feature5_warp,flow6_upsample), dim=1)
	flow5=self.flowestimator5(feature5_cat) ###diff: min: 0.0, max: 0.000339537858963, mean: 2.658415724e-05, median: 2.77673825622e-06

	flow5_upsample=self.upsample_2_factor(flow5)
	feature4_warp = self.resample1(feature2_pyramid4, flow5_upsample)
	feature4_cat=torch.cat((feature1_pyramid4, feature4_warp,flow5_upsample), dim=1)
	flow4=self.flowestimator4(feature4_cat)

	flow4_upsample=self.upsample_2_factor(flow4)
	feature3_warp = self.resample1(feature2_pyramid3, flow4_upsample)
	feature3_cat=torch.cat((feature1_pyramid3, feature3_warp,flow4_upsample), dim=1)
	flow3=self.flowestimator3(feature3_cat)

	flow3_upsample=self.upsample_2_factor(flow3)
	feature2_warp = self.resample1(feature2_pyramid2, flow3_upsample)
	feature2_cat=torch.cat((feature1_pyramid2, feature2_warp,flow3_upsample), dim=1)
	f2=self.flowestimator2_f2(feature2_cat)
	flow2=self.flowestimator2(f2)



	###
	f2_flow2_cat=torch.cat((f2, flow2), dim=1)
	flow_delta=self.context(f2_flow2_cat)
	flow_quarter=torch.add(flow_delta,flow2)
	flow=self.upsample_2_factor(flow_quarter)
        
        return flow


