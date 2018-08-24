import sys

sys.path.append('/extra/caffe/build_caffe/caffe_rtpose/python')
import caffe
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from prototxt import *
import pydot
import re

layer_dict = {'ConvNdBackward': 'Convolution',
              'ThresholdBackward': 'ReLU',
              'MaxPool2dBackward': 'Pooling',
              'AvgPool2dBackward': 'Pooling',
              'DropoutBackward': 'Dropout',
              'AddmmBackward': 'InnerProduct',
              'BatchNormBackward': 'BatchNorm',
              'AddBackward': 'Eltwise',
              'AddBackward1':'Eltwise',
              'ViewBackward': 'Reshape',
              'ConcatBackward': 'Concat',
              'CatBackward':'Concat',
              'UpsamplingNearest2d': 'Deconvolution',
              'UpsamplingBilinear2d': 'Deconvolution',
              'UpsamplingBilinear2dBackward': 'Deconvolution',
              'SigmoidBackward': 'Sigmoid',
              'LeakyReLUBackward': 'ReLU',
              'LeakyReluBackward': 'ReLU',
              'NegateBackward': 'Power',
              'MulBackward': 'Eltwise',
              'MulBackward0': 'Eltwise',   #  A * value
              'SubBackward1': 'Eltwise',
              'SpatialCrossMapLRNFunc': 'LRN',
              'Resample2dFunctionBackward':'FlowWarp',
              'IndexBackward':'Slice',
              'PReLUBackward':'PReLU'}

layer_id = 0

def find_bottom_name(layers,key):
    for index in range((len(layers))):
        layer = layers[index]
        if(layer['name'] == key):
            top = layer['top']
            if (len(top) > 1):
                result = top
            else:
                result = layer['top'][0]
            return result

def flow_shared_layer(layers,flow1_layer,flow2_layer):
    for index_flow2 in range(len(flow2_layer)):
        flow2 = flow2_layer[index_flow2]
        for index_layer2 in range(len(layers)):
            if layers[index_layer2]['name'] == flow2:
                index_layer1 = index_layer2 - 153
                #if index_flow2 == 0 or index_flow2 == 4:
                #    layers[index_layer2]['top'] = layers[index_layer1]['top']
                #if index_flow2 == len(flow2_layer) - 1:
                #    layers[index_layer2]['bottom'] = layers[index_layer1]['bottom']
                #else:
                #    layers[index_layer2]['top'] = layers[index_layer1]['top']
                #    layers[index_layer2]['bottom'] = layers[index_layer1]['bottom']

def pytorch2caffe(input_var,output_var, protofile, caffemodel):
    global layer_id
    net_info = pytorch2prototxt(input_var,output_var)
    #print_prototxt(net_info)
    save_prototxt(net_info, protofile)
    
    if caffemodel is None:
        return
    net = caffe.Net(protofile, caffe.TEST)
    params = net.params
    layer_id = 1
    seen = set()
    
    def convert_layer(func):
        if True:
            global layer_id
            parent_type = str(type(func).__name__)
            if hasattr(func, 'next_functions'):
                for u in func.next_functions:
                    if u[0] is not None:
                        child_type = str(type(u[0]).__name__)
                        child_name = child_type + str(layer_id)
                        if child_type != 'AccumulateGrad' and (
                                parent_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                            if u[0] not in seen:
                                convert_layer(u[0])
                                seen.add(u[0])
                            if child_type != 'ViewBackward':
                                layer_id = layer_id + 1

            parent_name = parent_type + str(layer_id)
            print('converting %s' % parent_name)
            if parent_type == 'ConvNdBackward':
                if func.next_functions[1][0] is not None:
                    weights = func.next_functions[1][0].variable.data
                    if func.next_functions[2][0]:
                        biases = func.next_functions[2][0].variable.data
                    else:
                        biases = None
                    save_conv2caffe(weights, biases, params[parent_name])
            elif parent_type == 'BatchNormBackward':
                running_mean = func.running_mean
                running_var = func.running_var
                bn_name = parent_name + "_bn"
                save_bn2caffe(running_mean, running_var, params[bn_name])

                affine = func.next_functions[1][0] is not None
                if affine:
                    scale_weights = func.next_functions[1][0].variable.data
                    scale_biases = func.next_functions[2][0].variable.data
                    scale_name = parent_name + "_scale"
                    save_scale2caffe(scale_weights, scale_biases, params[scale_name])
            elif parent_type == 'AddmmBackward':
                biases = func.next_functions[0][0].variable.data
                weights = func.next_functions[2][0].next_functions[0][0].variable.data
                save_fc2caffe(weights, biases, params[parent_name])
            elif parent_type == 'UpsamplingNearest2d':
                print('UpsamplingNearest2d')
    
    convert_layer(output_var.grad_fn)
    print('save caffemodel to %s' % caffemodel)
    net.save(caffemodel)

def save_conv2caffe(weights, biases, conv_param):
    if biases is not None:
        conv_param[1].data[...] = biases.numpy()
    conv_param[0].data[...] = weights.numpy()


def save_fc2caffe(weights, biases, fc_param):
    print(biases.size(), weights.size())
    print(fc_param[1].data.shape)
    print(fc_param[0].data.shape)
    fc_param[1].data[...] = biases.numpy()
    fc_param[0].data[...] = weights.numpy()


def save_bn2caffe(running_mean, running_var, bn_param):
    bn_param[0].data[...] = running_mean.numpy()
    bn_param[1].data[...] = running_var.numpy()
    bn_param[2].data[...] = np.array([1.0])


def save_scale2caffe(weights, biases, scale_param):
    scale_param[1].data[...] = biases.numpy()
    scale_param[0].data[...] = weights.numpy()

def pytorch2prototxt(input_var, output_var):
    global layer_id
    net_info = OrderedDict()
    props1 = OrderedDict()
    props1['name'] = 'pytorch'
    props1['input'] = 'data1'
    props1['input_dim'] = input_var.size()              ###for two inputs
    props2 = OrderedDict()
    props2['input'] = 'data2'
    props2['input_dim'] = input_var.size()  

    layers = []

    layer_id = 1
    seen = set()
    top_names = dict()
    attention_layid = [186,208,214,274,190,201,194,219,248,224,230,255,281]  ###for synthesis network,which can't use in-place
    flow1_layer = []
    flow2_layer = []

    def add_layer(func):
        global layer_id
        parent_type = str(type(func).__name__)          ###parent_type is from pytorch_graph.dot
        parent_bottoms = []
        
        if hasattr(func, 'next_functions'):
            for u in func.next_functions:
                if u[0] is not None:
                    child_type = str(type(u[0]).__name__)
                    child_name = child_type + str(layer_id)
                    if child_type != 'AccumulateGrad' and (
                            parent_type != 'AddmmBackward' or child_type != 'TransposeBackward'):
                        if u[0] not in seen:
                            top_name = add_layer(u[0])
                            parent_bottoms.append(top_name)
                            seen.add(u[0])
                        else:
                            top_name = top_names[u[0]]
                            parent_bottoms.append(top_name)
                        if child_type != 'ViewBackward':
                            layer_id = layer_id + 1
        
        parent_name = parent_type + str(layer_id)
        layer = OrderedDict()
        layer['name'] = parent_name
        layer['type'] = layer_dict[parent_type]
        parent_top = parent_name
        #if layer_id > 0 and layer_id < 171:
        #    flow1_layer.append(parent_name)
        #if layer_id > 174 and layer_id < 345:
        #    flow2_layer.append(parent_name)
        if len(parent_bottoms) > 0:
            #layer['bottom'] = parent_bottoms
            ### for relu use in-place
            z_parent_bottoms = []   ###for layer whoes bottom include relu,cause in-place,place bottom
            for index in range(len(parent_bottoms)):
	        if(re.search('^LeakyReLUBackward',parent_bottoms[index]) or re.search('^LeakyReluBackward',parent_bottoms[index]) or re.search('^PReLUBackward',parent_bottoms[index])):
                    z_top = find_bottom_name(layers,parent_bottoms[index])
                    z_parent_bottoms.append(z_top)
                else:
                    z_parent_bottoms.append(parent_bottoms[index])
            layer['bottom'] = z_parent_bottoms
            parent_bottoms = z_parent_bottoms
            ###
        else:                                           ###I found that just for the layer whoes bottom only has 'input' has len(parent_bottoms)=0
            if parent_name == 'ConvNdBackward1':    ###input1 used to get flow(input1,input2) 1 data1
                layer['bottom'] = ['data1']   
            elif parent_name == 'ConvNdBackward5':  ###input2  used to get flow(input1,input2) 5 data2
                layer['bottom'] = ['data2']
            elif parent_name == 'ConvNdBackward172':###context 172  data1
                layer['bottom'] = ['data1']
            elif parent_name == 'ConvNdBackward178':###context  347 data2
                layer['bottom'] = ['data2']
        layer['top'] = parent_top

        if parent_type == 'MulBackward':
            eltwise_param = {
                'operation': 'PROD',
            }
            layer['eltwise_param'] = eltwise_param
        if parent_type == 'MulBackward0':
            eltwise_param = OrderedDict()
            eltwise_param['operation'] = 'MUL'
            mul_param = []
            if parent_name == 'MulBackward0176':
                mul_param.append(1.0)
            else:
                mul_param.append(0.5)
            eltwise_param['mulvalue'] = mul_param
            layer['eltwise_param'] = eltwise_param
        elif parent_type == 'NegateBackward':
            power_param = {
                'power': 1,
                'scale': -1.,
                'shift': 0
            }
            layer['power_param'] = power_param
        elif parent_type == 'LeakyReLUBackward':
            negative_slope = func.additional_args[0]
            #layer['relu_param'] = {'negative_slope': negative_slope,'engine': 'CUDNN'}
            layer['relu_param'] = {'negative_slope': negative_slope}
        elif parent_type == 'LeakyReluBackward':
            negative_slope = 0.01    ########################### 0.01
            #layer['relu_param'] = {'negative_slope': negative_slope,'engine': 'CUDNN'}
            layer['relu_param'] = {'negative_slope': negative_slope}
        elif parent_type == 'PReLUBackward':
	    if parent_name == 'PReLUBackward1':         ###########################...............................
                parent_bottoms.append('data1')
                layer['bottom'] = parent_bottoms
            layer['param'] = {'lr_mult': 1, 'decay_mult': 0}
            prelu_param = OrderedDict()
            slope_a = func.saved_variables[1].data.numpy()[0]
            prelu_param['filler'] = {'value':slope_a}
            prelu_param['channel_shared'] = 'false'
            layer['prelu_param'] = prelu_param
        elif parent_type == 'UpsamplingNearest2d':
            conv_param = OrderedDict()
            factor = func.scale_factor
            conv_param['num_output'] = func.saved_tensors[0].size(1)
            conv_param['group'] = conv_param['num_output']
            conv_param['kernel_size'] = (2 * factor - factor % 2)
            conv_param['stride'] = factor
            conv_param['pad'] = int(np.ceil((factor - 1) / 2.))
            conv_param['weight_filler'] = {'type': 'bilinear'}
            conv_param['bias_term'] = 'false'
            layer['convolution_param'] = conv_param
            layer['param'] = {'lr_mult': 0, 'decay_mult': 0}
        elif parent_type == 'UpsamplingBilinear2d':
            conv_param = OrderedDict()
            factor = func.scale_factor[0]
            conv_param['num_output'] = func.input_size[1]
            conv_param['group'] = conv_param['num_output']
            conv_param['kernel_size'] = (2 * factor - factor % 2)
            conv_param['stride'] = factor
            conv_param['pad'] = int(np.ceil((factor - 1) / 2.))
            conv_param['weight_filler'] = {'type': 'bilinear'}
            conv_param['bias_term'] = 'false'
            layer['convolution_param'] = conv_param
            layer['param'] = {'lr_mult': 0, 'decay_mult': 0}
        elif parent_type == 'UpsamplingBilinear2dBackward':
            conv_param = OrderedDict()
            factor = func.scale_factor[0]
            conv_param['num_output'] = func.input_size[1]
            conv_param['group'] = conv_param['num_output']
            conv_param['kernel_size'] = (2 * factor - factor % 2)
            conv_param['stride'] = factor
            conv_param['pad'] = int(np.ceil((factor - 1) / 2.))
            conv_param['weight_filler'] = {'type': 'bilinear'}
            conv_param['bias_term'] = 'false'
            layer['convolution_param'] = conv_param
            layer['param'] = {'lr_mult': 0, 'decay_mult': 0}
        elif parent_type == 'ConcatBackward':
            concat_param = OrderedDict()
            concat_param['axis'] = func.dim
            layer['concat_param'] = concat_param
        elif parent_type == 'CatBackward':
            concat_param = OrderedDict()
            #concat_param['axis'] = func.dim
            concat_param['axis'] = 1
            layer['concat_param'] = concat_param

            #parent_bottoms.append('data1')#-------------------------------------fuck-------------------------------experiment should cancel
            #parent_bottoms.append('data1')#-------------------------------------fuck-------------------------------
            #layer['bottom'] = parent_bottoms#-------------------------------------fuck-------------------------------


        elif parent_type == 'IndexBackward':     ### for IndexBackward,it has a special 'top'
            index_param = OrderedDict()
            index_top = []
            index_top.append(parent_name)
            #top_one = parent_name + '_zcam1'
            top_two = parent_name + '_zcam2'
            #index_top.append(top_one)
            index_top.append(top_two)
            index_param['axis'] = 1
            index_param['slice_point'] = 2
            layer['slice_param'] = index_param
            layer['top'] = index_top
        elif parent_type == 'Resample2dFunctionBackward': 
            flowwarp_layer = OrderedDict()
            if len(parent_bottoms) == 1:
                if parent_name == 'Resample2dFunctionBackward171':  ###flow warp,input1*flow
                    parent_bottoms.insert(0,'data1')
                elif parent_name == 'Resample2dFunctionBackward177':###flow warp,input2*flow
                    parent_bottoms.insert(0,'data2')
            flowwarp_layer['bottom'] = parent_bottoms
            layer['bottom'] = flowwarp_layer['bottom']
        elif parent_type == 'ConvNdBackward':
            # Only for UpsamplingCaffe
            if func.transposed is True and func.next_functions[1][0] is None:
                layer['type'] = layer_dict['UpsamplingBilinear2d']
                conv_param = OrderedDict()
                factor = func.stride[0]
                conv_param['num_output'] = func.next_functions[0][0].saved_tensors[0].size(1)
                conv_param['group'] = conv_param['num_output']
                conv_param['kernel_size'] = (2 * factor - factor % 2)
                conv_param['stride'] = factor
                conv_param['pad'] = int(np.ceil((factor - 1) / 2.))
                conv_param['weight_filler'] = {'type': 'bilinear'}
                conv_param['bias_term'] = 'false'
                #conv_param['engine'] = 'CUDNN'
                layer['convolution_param'] = conv_param
                layer['param'] = {'lr_mult': 0, 'decay_mult': 0}
            else:
                weights = func.next_functions[1][0].variable
                conv_param = OrderedDict()
                conv_param['num_output'] = weights.size(0)
                conv_param['pad_h'] = func.padding[0]
                conv_param['pad_w'] = func.padding[1]
                conv_param['kernel_h'] = weights.size(2)
                conv_param['kernel_w'] = weights.size(3)
                conv_param['stride'] = func.stride[0]
                conv_param['dilation'] = func.dilation[0]
                #conv_param['engine'] = 'CUDNN'
                if func.next_functions[2][0] == None:
                    conv_param['bias_term'] = 'false'
                else:
                    conv_param['bias_term'] = 'true'     ###########################...............................
                layer['convolution_param'] = conv_param

        elif parent_type == 'BatchNormBackward':
            bn_layer = OrderedDict()
            bn_layer['name'] = parent_name + "_bn"
            bn_layer['type'] = 'BatchNorm'
            bn_layer['bottom'] = parent_bottoms
            bn_layer['top'] = parent_top

            batch_norm_param = OrderedDict()
            batch_norm_param['use_global_stats'] = 'true'
            batch_norm_param['eps'] = func.eps
            bn_layer['batch_norm_param'] = batch_norm_param

            affine = func.next_functions[1][0] is not None
            # func.next_functions[1][0].variable.data
            if affine:
                scale_layer = OrderedDict()
                scale_layer['name'] = parent_name + "_scale"
                scale_layer['type'] = 'Scale'
                scale_layer['bottom'] = parent_top
                scale_layer['top'] = parent_top
                scale_param = OrderedDict()
                scale_param['bias_term'] = 'true'
                scale_layer['scale_param'] = scale_param
            else:
                scale_layer = None

        elif parent_type == 'ThresholdBackward':
            parent_top = parent_bottoms[0]
        elif parent_type == 'MaxPool2dBackward':
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'MAX'
            pooling_param['kernel_size'] = func.kernel_size[0]
            pooling_param['stride'] = func.stride[0]
            # http://netaz.blogspot.com/2016/08/confused-about-caffes-pooling-layer.html
            padding = func.padding[0]
            # padding = 0 if func.padding[0] in {0, 1} else func.padding[0]
            pooling_param['pad'] = padding
            layer['pooling_param'] = pooling_param
        elif parent_type == 'AvgPool2dBackward':
            pooling_param = OrderedDict()
            pooling_param['pool'] = 'AVE'
            pooling_param['kernel_size'] = func.kernel_size[0]
            pooling_param['stride'] = func.stride[0]
            pooling_param['pad'] = func.padding[0]
            layer['pooling_param'] = pooling_param
        elif parent_type == 'DropoutBackward':
            parent_top = parent_bottoms[0]
            dropout_param = OrderedDict()
            dropout_param['dropout_ratio'] = func.p
            layer['dropout_param'] = dropout_param
        elif parent_type == 'AddmmBackward':
            inner_product_param = OrderedDict()
            inner_product_param['num_output'] = func.next_functions[0][0].variable.size(0)
            layer['inner_product_param'] = inner_product_param
        elif parent_type == 'ViewBackward':
            parent_top = parent_bottoms[0]
        elif parent_type == 'AddBackward':
            eltwise_param = OrderedDict()
            eltwise_param['operation'] = 'SUM'
            layer['eltwise_param'] = eltwise_param
        elif parent_type == 'AddBackward1':
            eltwise_param = OrderedDict()
            eltwise_param['operation'] = 'SUM'
            layer['eltwise_param'] = eltwise_param
        elif parent_type == 'SubBackward1':
            eltwise_param = OrderedDict()
            eltwise_param['operation'] = 'SUM'
            coeff_param = []
            coeff_param.append(1)
            coeff_param.append(-1)
            eltwise_param['coeff'] = coeff_param
            layer['eltwise_param'] = eltwise_param
        elif parent_type == 'SpatialCrossMapLRNFunc':
            layer['lrn_param'] = {
                'local_size': func.size,
                'alpha': func.alpha,
                'beta': func.beta,
            }

        if parent_type != 'IndexBackward':   ###for IndexBackward layer,it has a special 'top'
            layer['top'] = parent_top  # reset layer['top'] as parent_top may change
	if parent_type == 'LeakyReLUBackward' or parent_type == 'LeakyReluBackward' or parent_type == 'PReLUBackward':###for relu,fill top in-place
	    layer['top'] = parent_bottoms
        for index in range(len(attention_layid)):###for synthesis network,you can't use in-place
            if layer_id == attention_layid[index]:
                layer['top'] = parent_top
        if parent_type != 'ViewBackward':
            if parent_type == "BatchNormBackward":
                layers.append(bn_layer)
                if scale_layer is not None:
                    layers.append(scale_layer)
            else:
                layers.append(layer)
                # layer_id = layer_id + 1
        top_names[func] = parent_top
        return parent_top

    add_layer(output_var.grad_fn)
    net_info['props1'] = props1
    net_info['props2'] = props2
    #flow_shared_layer(layers,flow1_layer,flow2_layer)
    net_info['layers'] = layers
    return net_info
    

def plot_graph(top_var, fname,params=None):
    """
    This method don't support release v0.1.12 caused by a bug fixed in: https://github.com/pytorch/pytorch/pull/1016
    So if you want to use `plot_graph`, you have to build from master branch or wait for next release.

    Plot the graph. Make sure that require_grad=True and volatile=False
    :param top_var: network output Varibale
    :param fname: file name
    :param params: dict of (name, Variable) to add names to node that
    :return: png filename
    """
    from graphviz import Digraph
    import pydot
    dot = Digraph(comment='LRP',
                  node_attr={'style': 'filled', 'shape': 'box'})
    # , 'fillcolor': 'lightblue'})

    seen = set()

    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    ###var has three types:tensor(data)/variable/layer
    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):         ###"var" is tensor or data
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):   ###"var" is "<AccumulateGrad object at 0x7f36d0413090>"
                u = var.variable
                name = '{}\n '.format(param_map[id(u)]) if params is not None else ''
                node_name = '{}{}'.format(name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:                            ###"var" is layer which include the whole layer  (450 layers)
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):###tensor(data) has no next_functuions (which is 728,include 450 layers and 278 AccumulateGrad)
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):###it's strange,i just find three layers here:upsampling/prelu/warp,totally 92 layers(which include 16 warp,18 upsample,58 prelu)
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(top_var.grad_fn)   ###"top_var" is output of pytorch,it has data which looks like parameters or tensor and node name which looks like layer name
    dot.save(fname)
    (graph,) = pydot.graph_from_dot_file(fname)
    im_name = '{}.png'.format(fname)
    graph.write_png(im_name)

    return im_name


if __name__ == '__main__':
    import torchvision
    import os

    m = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    m.eval()

    input_var = Variable(torch.rand(1, 3, 299, 299))
    output_var = m(input_var)

    # plot graph to png
    output_dir = 'demo'
    plot_graph(output_var, os.path.join(output_dir, 'inception_v3.dot'))

    pytorch2caffe(input_var, output_var, os.path.join(output_dir, 'inception_v3-pytorch2caffe.prototxt'),
                  os.path.join(output_dir, 'inception_v3-pytorch2caffe.caffemodel'))
