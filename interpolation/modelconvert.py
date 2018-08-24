# -*- coding: utf-8
from pytorch2caffe import plot_graph, pytorch2caffe
import sys
sys.path.append('/data/build_caffe/caffe_rtpose/python')
import caffe
import numpy as np
import os

import torch
from torch.autograd import Variable
import torchvision

#######george###############---------
import time
from collections import OrderedDict
from options.test_options import TestOptions
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import numpy
import PIL
import PIL.Image
import torchvision.transforms as transforms
from PIL import Image



opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

'''
transform_list = [transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
transform_img=transforms.Compose(transform_list)
image_number=2
for i in range(image_number):
    if i <= image_number-2:
        if i==0:
            epoch_start_time = time.time()
        if i==100:
           epoch_end_time = time.time()
           timecost=epoch_end_time-epoch_start_time
           print(timecost)
        j=i+1
        trpre="%(image)s-%(j)07d.jpg"%{'image':'image','j':j}
        jj=j+1
        trnext="%(image)s-%(jj)07d.jpg"%{'image':'image','jj':jj}
        datastrpre = os.path.join('/home/george/project/pix2pixHD_v3/datasets/size/',trpre)
        datastrnext = os.path.join('/home/george/project/pix2pixHD_v3/datasets/size/',trnext)
        print(datastrpre)
        print(datastrnext)

        tensorInputFirst = torch.FloatTensor(numpy.rollaxis(numpy.asarray(PIL.Image.open(datastrpre)), 2, 0).astype(numpy.float) / 255.0).contiguous()
        tensorInputFirst=transform_img(tensorInputFirst)
        variablePaddingFirst = torch.autograd.Variable(data=tensorInputFirst.view(1, 3, 1024, 1984), volatile=True)
        tensorInputFirs = variablePaddingFirst.cuda()

        tensorInputSecond = torch.FloatTensor(numpy.rollaxis(numpy.asarray(PIL.Image.open(datastrnext)), 2, 0).astype(numpy.float) / 255.0).contiguous()
        tensorInputSecond=transform_img(tensorInputSecond)
        variablePaddingSecond = torch.autograd.Variable(data=tensorInputSecond.view(1, 3, 1024, 1984), volatile=True)
        tensorInputSecon = variablePaddingSecond.cuda()

        #generated = model.reference(tensorInputFirs, tensorInputSecon)
        input_tuple=(tensorInputFirs,tensorInputSecon,tensorInputSecon)
        torch_out = torch.onnx._export(model,input_tuple,"resolution.onnx")
'''
#####george###################>>>>>>>>>>>

# test the model or generate model
test_mod = False

caffemodel_dir = 'modelconvert'
input_size = (1, 3, 256, 256)

model_def = os.path.join(caffemodel_dir, 'model.prototxt')
model_weights = os.path.join(caffemodel_dir, 'model.caffemodel')
input_name = 'ConvNdBackward1'
output_name = 'ConvNdBackward521'

# pytorch net
model = create_model(opt)
#model.eval()
#set gpu
torch.cuda.set_device(1)


transform_list = [transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
transform_img=transforms.Compose(transform_list)
datastrpre = '/home/george/deepstructure/caffe/build/examples/cpp_classification/1.jpg'
datastrnext = '/home/george/deepstructure/caffe/build/examples/cpp_classification/3.jpg'

tensorInputFirst = torch.FloatTensor(numpy.rollaxis(numpy.asarray(PIL.Image.open(datastrpre)), 2, 0).astype(numpy.float) / 255.0).contiguous()
tensorInputFirst=transform_img(tensorInputFirst)
variablePaddingFirst = torch.autograd.Variable(data=tensorInputFirst.view(1, 3, 256, 256), volatile=True)
variablePaddingFir = variablePaddingFirst.cuda()
input_data1 = variablePaddingFirst.data.numpy()

tensorInputSecond = torch.FloatTensor(numpy.rollaxis(numpy.asarray(PIL.Image.open(datastrnext)), 2, 0).astype(numpy.float) / 255.0).contiguous()
tensorInputSecond=transform_img(tensorInputSecond)
variablePaddingSecond = torch.autograd.Variable(data=tensorInputSecond.view(1, 3, 256, 256), volatile=True)
variablePaddingSeco = variablePaddingSecond.cuda()
input_data2 = variablePaddingSecond.data.numpy()



# random input
'''
image1 = np.random.random(input_size)
image2 = np.random.random(input_size)
input_data1 = image1.astype(np.float32)
input_data2 = image2.astype(np.float32)
'''
'''
image1 = np.zeros(input_size)
image2 = np.zeros(input_size)
input_data1 = image1.astype(np.float32)
input_data2 = image2.astype(np.float32)
'''

# pytorch forward
input_var1 = Variable(torch.from_numpy(input_data1))
input_var2 = Variable(torch.from_numpy(input_data2))
if not test_mod:
    # generate caffe model
    output_var = model(input_var1,input_var2)
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<o ,fuck you!>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    plot_graph(output_var,os.path.join(caffemodel_dir, 'pytorch_graph.dot'))
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<o ,fuck you again!>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    pytorch2caffe(input_var1,output_var, model_def, model_weights)
    exit(0)

# test caffemodel
caffe.set_device(1)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()
net = caffe.Net(model_def, model_weights, caffe.TEST)

print('hi baby...1')
net.blobs['data1'].data[...] = input_data1
net.blobs['data2'].data[...] = input_data2
print('hi baby...2')
hioutput=net.forward(start=input_name)
print('hi baby...3')
caffe_output = net.blobs[output_name].data
print(caffe_output)
print('hi baby...4')
#filters = net.params['data1_input_0_split_1'][0].data  
#print(filters) 

for layer_name, blob in net.blobs.iteritems():  
    print layer_name + '\t' + str(blob.data.shape) 




model=model.cuda()
input_var1=input_var1.cuda()
input_var2=input_var2.cuda()
output_var = model(input_var1,input_var2)

oriimage=util.tensor2im(output_var.data[0])
image_pil = Image.fromarray(oriimage)
image_pil.save("fuck.jpg")


###+++
#visuals = OrderedDict([('b_synthesized', util.tensor2im(output_var.data[0]))])
#visualizer.my_save_images(visuals, i, opt.how_many)
###---
pytorch_output = output_var.data.cpu().numpy()

print(input_size, pytorch_output.shape, caffe_output.shape)
print('pytorch: min: {}, max: {}, mean: {}'.format(pytorch_output.min(), pytorch_output.max(), pytorch_output.mean()))
print('  caffe: min: {}, max: {}, mean: {}'.format(caffe_output.min(), caffe_output.max(), caffe_output.mean()))

diff = np.abs(pytorch_output - caffe_output)
print('   diff: min: {}, max: {}, mean: {}, median: {}'.format(diff.min(), diff.max(), diff.mean(), np.median(diff)))


