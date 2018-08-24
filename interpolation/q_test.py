### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import time
from collections import OrderedDict
from options.test_options import TestOptions
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import numpy
import torch
import PIL
import PIL.Image
from torch.autograd import Variable
import torchvision.transforms as transforms

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# model
model = create_model(opt)
visualizer = Visualizer(opt)
# test
transform_list = [transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
transform_img=transforms.Compose(transform_list)
image_number=1001
for i in range(image_number):
    if i <= image_number-2:
	if i==1:
	    epoch_start_time = time.time()
	if i==101:
	   epoch_end_time = time.time()
	   timecost=epoch_end_time-epoch_start_time
	   print(timecost)
    	j=i+1
    	trpre="%(image)s-%(j)07d.jpg"%{'image':'image','j':j}
    	jj=j+1
    	trnext="%(image)s-%(jj)07d.jpg"%{'image':'image','jj':jj}
	datastrpre = os.path.join('/home/george/project/pix2pixHD_3.0/datasets/size/40/',trpre)
	datastrnext = os.path.join('/home/george/project/pix2pixHD_3.0/datasets/size/40/',trnext)
    	print(datastrpre)
    	print(datastrnext)

	tensorInputFirst = torch.FloatTensor(numpy.rollaxis(numpy.asarray(PIL.Image.open(datastrpre)), 2, 0).astype(numpy.float) / 255.0).contiguous()
	tensorInputFirst=transform_img(tensorInputFirst)
	variablePaddingFirst = torch.autograd.Variable(data=tensorInputFirst.view(1, 3, 576, 576), volatile=True)
	tensorInputFirs = variablePaddingFirst.cuda()

	tensorInputSecond = torch.FloatTensor(numpy.rollaxis(numpy.asarray(PIL.Image.open(datastrnext)), 2, 0).astype(numpy.float) / 255.0).contiguous()
	tensorInputSecond=transform_img(tensorInputSecond)
	variablePaddingSecond = torch.autograd.Variable(data=tensorInputSecond.view(1, 3, 576, 576), volatile=True)
	tensorInputSecon = variablePaddingSecond.cuda()

    	generated = model.reference(tensorInputFirs, tensorInputSecon)
        #print generated
    	visuals = OrderedDict([('b_synthesized', util.tensor2im(generated.data[0]))])
    	visualizer.my_save_images(visuals, i, opt.how_many)


