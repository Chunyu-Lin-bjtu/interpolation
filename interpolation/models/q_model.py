### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from util.image_pool import ImagePool
from q_context import ContextGenerator
from q_flow import FlowGenerator
from q_synthesis import GridNetwork
save_mod = False
train_mod = False
gpu_mod = True



if gpu_mod:
    from .resample2d_package_gpu.modules.resample2d import Resample2d
else:
    from .resample2d_package_cpu.modules.resample2d import Resample2d


if train_mod:
    from .q_loss import FeatureLoss
    from .q_loss import ColorLoss
    from .q_loss import LaplacianLoss

'''
def add_to_64(input,inputsize):
    
    h=inputsize[2]
    w=inputsize[3]
    
    padw=0
    padh=0
    intPaddingLeft = 0
    intPaddingRight = 0
    intPaddingTop = 0
    intPaddingBottom = 0

    if w%64 != 0:
    	padw=(w/64+1)*64 - w

    if h%64 != 0:
    	padh=(h/64+1)*64 - h

    
    if padw != 0:
    	intPaddingLeft = padw/2
    	intPaddingRight = padw - intPaddingLeft

    if padh != 0:
    	intPaddingTop = padh/2
    	intPaddingBottom = padh - intPaddingTop

    if padw != 0 or padh != 0:
    	modulePad = nn.ReplicationPad2d([ intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])
    	modulePad = modulePad.cuda()
    	output=modulePad(input)
    else:
	output = input
    
    
    

    return output

def delete_to_64(input,inputsize):
    
    h=inputsize[2]
    w=inputsize[3]
    
    padw=0
    padh=0
    intPaddingLeft = 0
    intPaddingRight = 0
    intPaddingTop = 0
    intPaddingBottom = 0

    if w%64 != 0:
    	padw=(w/64+1)*64 - w

    if h%64 != 0:
    	padh=(h/64+1)*64 - h

    
    if padw != 0:
    	intPaddingLeft = padw/2
    	intPaddingRight = padw - intPaddingLeft

    if padh != 0:
    	intPaddingTop = padh/2
    	intPaddingBottom = padh - intPaddingTop

    if padw != 0 or padh != 0:
    	modulePad = nn.ReplicationPad2d([ 0-intPaddingLeft, 0-intPaddingRight, 0-intPaddingTop, 0-intPaddingBottom])
    	modulePad = modulePad.cuda()
    	output=modulePad(input)
    else:
	output = input
    
    
    

    return output
'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class InterpolationNetwork(nn.Module):
    def __init__(self):
        super(InterpolationNetwork, self).__init__()

	self.alph = 0.5
	self.belta = 0.5

	###context extractor
	self.context_extractor=ContextGenerator()
	###flow estimator
	self.flow_estimator=FlowGenerator()
	self.flow_estimator.apply(weights_init)#initialize weights
	###synthesis network
	self.synthesis_network=GridNetwork(134, 3)
	self.synthesis_network.apply(weights_init)#initialize weights
        
        ###flow warp
        self.resample2 = Resample2d()
        


    def forward(self, input1,input2):
	
	context1 = self.context_extractor(input1)
	context2 = self.context_extractor(input2)

	flow1 = self.flow_estimator(input1,input2)*self.alph
	#flow2 = self.flow_estimator(input2,input1)*self.belta
        flow2 = -1*flow1
        
	###forward warping
	
        if train_mod:
	    input1_warp=self.resample2(input1,flow1)
	    context1_warp=self.resample2(context1.detach(),flow1)
	    input2_warp=self.resample2(input2,flow2.detach())
	    context2_warp=self.resample2(context2.detach(),flow2.detach())
        else:
	    input1_warp=self.resample2(input1,flow1)
	    context1_warp=self.resample2(context1,flow1)
	    input2_warp=self.resample2(input2,flow2)
	    context2_warp=self.resample2(context2,flow2)


	###stack
	stack_=torch.cat((input1_warp, context1_warp,input2_warp,context2_warp), dim=1)
	
	###synthesis network
	result=self.synthesis_network(stack_)

        return result

class InterpolationModel(BaseModel):
    def name(self):
        return 'InterpolationModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
	self.isTest = opt.isTest
	self.continue_train = opt.continue_train

        ##### define networks
        if gpu_mod:     
            self.netG = InterpolationNetwork().cuda()
        else:
            self.netG = InterpolationNetwork()

        print('---------- Networks initialized -------------')

        # load networks
        if train_mod:
            # load networks
            if not self.isTest:
                if opt.continue_train:
            	    pretrained_path = opt.load_pretrain
            	    self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            else:
                pretrained_path = opt.load_pretrain
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
        else:
            pretrained_path = opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)

        if train_mod:
            # set loss functions and optimizers
            if self.isTrain or opt.continue_train:
                if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                    raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
                self.fake_pool = ImagePool(opt.pool_size)
                self.old_lr = opt.lr


                # define loss functions
                if not opt.no_vgg_loss:             
                    self.criterionVGG = FeatureLoss()

	        if not opt.no_color_loss: 
		    self.criterionCOLOR = nn.L1Loss()

	        if not opt.no_lap_loss: 
		    self.criterionLAP = LaplacianLoss()
        
                # Names so we can breakout loss
                self.loss_names = ['G_LAP', 'G_COLOR', 'G_VGG']

                # optimizer 
                params = list(self.netG.parameters())    
                self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999)) 

    #def forward(self, input1, groundtruth,input2, infer=False):
    def forward(self, input1,input2):
	'''
	if self.isTest:
	    inputsize = in1.size()
	    print('************')
	    print(inputsize)
	    print('************')
	    input1=add_to_64(in1,inputsize)
	    input2=add_to_64(in2,inputsize)	
	    groundtruth=add_to_64(ground,inputsize)
	'''

        fake_image = self.netG.forward(input1,input2)
	if save_mod:
            torch.save(self.netG.cpu(), 'see_model.pth')
	
        if train_mod:
            # VGG feature matching loss
            loss_VGG = 0
            if not self.opt.no_vgg_loss:
                loss_VGG = self.criterionVGG(fake_image, groundtruth.detach())

	    # color matching loss
	    loss_COLOR = 0
	    if not self.opt.no_color_loss: 
	        loss_COLOR = self.criterionCOLOR(fake_image, groundtruth.detach())*10

	    # laplacian feature matching loss
	    loss_LAP = 0
	    if not self.opt.no_lap_loss: 
	        loss_LAP = self.criterionLAP(fake_image, groundtruth.detach())*0.000000001*50
	    '''
	    fake_ = delete_to_64(fake_image,inputsize)
	    '''

            # Only return the fake_B image if necessary to save BW
            return [ [loss_LAP, loss_COLOR, loss_VGG], None if not infer else fake_image ]
        else:
            return fake_image


    def reference(self, input1,input2):
        fake_image = self.netG.forward(input1,input2)
	return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)

    '''
    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning global generator -----------')
    '''

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
