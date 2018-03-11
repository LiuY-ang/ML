# -*- coding:utf-8 -*-
import numpy as np
import random
from FullConnectedLayer import *
from datetime import datetime

#为数组增加Zero padding
def zero_padding(input_array,zero_padding):
    '''
    为数组增加zero padding，自动适配输入为2D和3D的情况
    '''
    if zero_padding==0:
        return input_array
    dimensions=input_array.ndim#获取input_array纬度
    if dimensions==3:
        input_depth,input_height,input_width=input_array.shape#分别获取深度，行，列数
        #建立扩展后的3维数组，该数组中的元素全是0
        padded_array=np.zeros(( input_depth,input_height+2*zero_padding,input_width+2*zero_padding ))
        #将input_array的值赋值给原来相应的位置
        padded_array[:,zero_padding:zero_padding+input_height,zero_padding:zero_padding+input_width]=input_array
        return padded_array
    elif dimensions==2:
        input_height,input_width=input_array.shape#获得矩阵的行和列
        padded_array=np.zeros( (input_height+2*zero_padding,input_width+2*zero_padding) )
        padded_array[zero_padding:zero_padding+input_height,zero_padding:zero_padding+input_width]=input_array
        return padded_array

def get_max_index(array):
    '''
    获取一个2维区域的最大值所在的索引
    '''
    pos=array.argmax()
    width=array.shape[1]
    return pos/width,pos%width
    # max_i=0
    # max_j=0
    # max_value=array[0,0]
    # for i in range(array.shape[0]):
    #     for j in range(array.shape[1]):
    #         if array[i,j]>max_value:
    #             max_i=i
    #             max_j=j
    #             max_value=array[i,j]
    # return max_i,max_j

#计算卷积
def convolution(input_array,kernel_array,output_array,stride,bias):
    '''
    计算卷积,自动适配输入为2D和3D的情况
    '''
    channel_number=kernel_array.ndim#获得卷积矩阵的纬度
    output_height,output_width=output_array.shape
    kernel_width=kernel_array.shape[-1]
    kernel_height=kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (\
                    get_patch(input_array,i,j,kernel_width,\
                            kernel_height,stride)*kernel_array\
                            ).sum()+bias
def get_patch(input_array,i,j,filter_width,filter_height,stride):
    '''
    从输入数组中获取本次卷积的区域，自动适配输入为2D和3D的情况
    返回input_array相应的部分，自己的理解
    '''
    start_i,start_j=i*stride,j*stride #分别计算在input_array起始的行和列
    if input_array.ndim==2:  #input_array为2维矩阵
        return input_array[start_i:start_i+filter_height,\
                            start_j:start_j+filter_width]
    elif input_array.ndim==3: #input_array为3维矩阵
        return input_array[:,\
                        start_i:start_i+filter_height,\
                        start_j:start_j+filter_width ]

def element_wise_op(array,operator):
    '''
    对numpy数组进行element wise操作
    在默认情况下，nditer将输入数组视为只读对象。要修改数组元素。必须指定读写(readwrite)
    或只写(writeonly)模式。这是由每个操作数标志控制的。一般而言，python中的赋值只需要
    更改本地或全局变量字典中的引用，而不是改变修改现有变量
    '''
    for i in np.nditer(array,op_flags=['readwrite']):
        i[...]=operator(i) #将激活函(operator)作用在output_array中的每个元素

class ReluActivator(object):
    def forward(self,weighted_input):
        return max(0,weighted_input)

    def backward(self,output):
        return 1 if output>0 else 0

class IdentityActivator(object):
    def forward(self,weighted_input):
        return weighted_input

    def backward(self,output):
        return 1

class Filter(object):
    def __init__(self,width,height,depth):
        '''
        初始化filter
        '''
        self.weights=np.random.uniform(-1,1,(depth,height,width))
        # self.bias=random.uniform(-1,1)
        self.bias=0
        #self.weights.shape 获得矩阵的形状，返回(深度，行数，列数)
        self.weights_gradient=np.zeros(self.weights.shape)
        self.bias_gradient=0
    def __repr__(self):
        '''
        打印信息
        '''
        return "filter weights:\n%s\nbias:\n%s" % (repr(self.weights),repr(self.bias))
    def get_weights(self):
        '''
        返回权值信息
        '''
        return self.weights
    def get_bias(self):
        '''
        返回偏置项
        '''
        return self.bias
    def update(self,learning_rate):
        '''
        更新权重矩阵和偏置。但是为什么是“-”号？
        # '''
        self.weights-=learning_rate*self.weights_gradient
        self.bias-=learning_rate*self.bias_gradient
        # self.weights+=learning_rate*self.weights_gradient
        # self.bias+=learning_rate*self.bias_gradient
class ConvolutionLayer(object):
    def __init__(self,input_width,input_height,channel_number,filter_width,\
            filter_height,filter_number,zero_padding,stride,activator,learning_rate):
        '''
        input_width:输入矩阵的宽度，即矩阵列
        input_height:输入矩阵的高度，即矩阵行
        channel_number:Feature Map也称作channel（通道）
        filter_width:卷积矩阵的宽度
        filter_height:卷积矩阵的高度
        filter_number:卷积的个数
        zero_padding:在矩阵需要补外围的o
        stride:步长
        activator:激活函数
        learning_rate:学习率
        '''
        self.input_width=input_width
        self.input_height=input_height
        #把本层的深度看作是feature map的个数
        self.channel_number=channel_number
        self.filter_width=filter_width
        self.filter_height=filter_height
        self.filter_number=filter_number#filter数量，超参数
        self.zero_padding=zero_padding#需要在矩阵外围添加zero_padding圈0
        self.stride=stride#步长
        #计算输出矩阵的列数
        self.output_width=ConvolutionLayer.calculate_output_size(\
            self.input_width,filter_width,zero_padding,stride
            )
        #计算输出矩阵的行数
        self.output_height=ConvolutionLayer.calculate_output_size(\
            self.input_height,filter_height,zero_padding,stride
            )
        #建立输出矩阵
        self.output_array=np.zeros( (self.filter_number,self.output_height,self.output_width) )
        self.filters=[]
        for i in range(self.filter_number):
            self.filters.append( Filter(self.filter_width,self.filter_height,self.channel_number) )
        self.activator=activator
        self.learning_rate=learning_rate
    def forward(self,input_array):
        '''
        计算卷积层的输出
        输出结果保存在self.output_array
        '''
        self.input_array=input_array#卷积层的输入
        #将self.input_array添加self.zero_padding圈0
        self.padded_input_array=zero_padding(input_array,self.zero_padding)
        for f in range(self.filter_number):
            filter_f=self.filters[f]
            convolution(self.padded_input_array,filter_f.get_weights(),\
                        self.output_array[f],self.stride,filter_f.get_bias())
        #将激活函数(op)作用在self.output_array中的每个元素
        #self.output_array=self.activator.forward(self.output_array)
        element_wise_op(self.output_array,self.activator.forward)
    def backward(self,input_array,sensitivity_array,activator):
        '''
        计算传递给前一层的误差项
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        '''
        #self.forward(input_array)#这句话有必要吗？
        self.bp_sensitivity_map(sensitivity_array,activator)
        self.bp_gradient(sensitivity_array)
    def bp_sensitivity_map(self,sensitivity_array,activator):
        '''
        计算传递到上一层的sensitivity map
        sensitivity_array:本层的sensitivity map,我感觉应该是前一层的sensitivity map
        activator:上一层的激活函数
        '''
        #处理卷积步长，对原始sensitivity map进行扩展,还原为步长为1的sensitivity map
        expanded_sensitivity_array=self.expand_sensitivity_map(sensitivity_array)
        #full卷积，对sensitivity map进行zero padding
        #虽然原始输入的zero padding单元也会获得残差
        #但这个残差不需要继续向上传递，因此就不计算了
        #获得扩展后sensitivity map的列数
        expanded_sensitivity_array_width=expanded_sensitivity_array.shape[2]
        #计算应该补的zero_padding的圈数
        zp=(self.input_width+self.filter_width-1-expanded_sensitivity_array_width)/2
        #将stride为1的sensitivity map在外圈进行补0,补zp圈0
        padded_sensitivity_array=zero_padding(expanded_sensitivity_array,zp)
        #初始化delta_array,用于保存传递到上一层的sensitivity map。也就是建立一个输入矩阵大小的矩阵
        self.delta_array=self.create_delta_array()
        #对于有多个filter的卷积层来说，最终传递到上一层的sensitivity map相当于
        #所有的filter的sensitivity map之和
        for f in range(self.filter_number):
            filter_f=self.filters[f]
            #将filter的权重翻转180度
            filpped_weights=np.array( map(lambda i:np.rot90(i,2),filter_f.get_weights()) )
            #计算与一个filter对应的delta_array
            delta_array=self.create_delta_array()
            for d in range(delta_array.shape[0]):
                convolution(padded_sensitivity_array[f],filpped_weights[d],delta_array[d],1,filter_f.get_bias())
            self.delta_array+=delta_array
        #将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array=np.array(self.input_array)
        element_wise_op(derivative_array,activator.backward)
        self.delta_array*=derivative_array

    def expand_sensitivity_map(self,sensitivity_array):
        '''
        将步长为S的sensitivity map还原为步长为1的sensitivity map
        '''
        depth=sensitivity_array.shape[0]
        #确定扩展后的sensitivity map的大小
        #计算stride为1时sensitivity map的大小。
        #因为步长为1，所以没有除以stride
        expanded_width=(self.input_width-self.filter_width+2*self.zero_padding+1)
        expanded_height=(self.input_height-self.filter_height+2*self.zero_padding+1)
        #构建新的sensitivity_map
        expand_array=np.zeros( (depth,expanded_height,expanded_width) )
        #从原始sensitivity map拷贝误差值
        # print "sensitivity_array.shape is  ",sensitivity_array.shape
        # print "expand_array.shape is  ",expand_array.shape
        # print "output_height is ",self.output_height
        # print "output_width is ",self.output_width
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos=i*self.stride
                j_pos=j*self.stride
                # print "expand_array[:,i_pos,j_pos] is ",expand_array[:,i_pos,j_pos]
                # print sensitivity_array[:,i,j]
                #print i,j
                expand_array[:,i_pos,j_pos]=sensitivity_array[:,i,j]
        return expand_array

    def create_delta_array(self):
        '''
        创建用来保存传递到上一层的sensitivity map的数组
        '''
        return np.zeros( (self.channel_number,self.input_height,self.input_width) )

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        for f in self.filters:
            f.update(self.learning_rate)

    def bp_gradient(self,sensitivity_array):
        '''
        '''
        #处理卷积步长，对原始sensitivity map进行扩展
        expanded_sensitivity_array=self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            filter_f=self.filters[f]
            for d in range(filter_f.weights.shape[0]):
                convolution(self.padded_input_array[d],expanded_sensitivity_array[f],\
                        filter_f.weights_gradient[d],1,filter_f.get_bias() )
            #计算偏置项的梯度
            filter_f.bias_gradient=expanded_sensitivity_array[f].sum()

    @staticmethod
    def calculate_output_size(input_size,filter_size,zero_padding,stride):
        '''
        计算输入矩阵的高度和宽度
        计算公式为:(origin_size-filter_size+2*zero_padding)/stride+1
        '''
        return (input_size - filter_size + 2*zero_padding)/stride +1

class MaxPoolingLayer(object):
    def __init__(self,input_width,input_height,channel_number,filter_width,filter_height,stride):
        '''
        '''
        self.input_width=input_width
        self.input_height=input_height
        self.channel_number=channel_number
        self.filter_width=filter_width
        self.filter_height=filter_height
        self.stride=stride
        self.output_width=(input_width - filter_width)/stride + 1
        self.output_height=(input_height - filter_height)/stride + 1
        self.output_array=np.zeros( (self.channel_number,self.output_height,self.output_width) )
    def forward(self,input_array):
        '''
        '''
        self.input_array=input_array
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d,i,j]=( get_patch(input_array[d],i,j,\
                            self.filter_width,self.filter_height,self.stride).max() )
    def backward(self,input_array,sensitivity_array):
        '''
        '''
        #print "INFO: MaxPoolingLayer_backward is ",input_array.shape
        self.delta_array=np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array=get_patch(input_array[d],i,j,\
                        self.filter_width,self.filter_height,self.stride)
                    k,l=get_max_index(patch_array)
                    self.delta_array[d,i*self.stride+k,j*self.stride+l]=sensitivity_array[d,i,j]

class ConvolutionNeuralNetwork(object):
    def __init__(self):
        '''
        建立处理MNIST数据集的卷积神经神经网络
        先建立卷积网络在建立全连接神经网络
        '''
        #建立卷积神经网络
        # convolution_1=ConvolutionLayer(28,28,1,5,5,16,0,1,ReluActivator(),0.001)
        # pooling_1=MaxPoolingLayer(24,24,16,2,2,2)
        # convolution_2=ConvolutionLayer(12,12,16,5,5,32,0,1,ReluActivator(),0.001)
        # pooling_2=MaxPoolingLayer(8,8,32,2,2,2)#最后输出的图像为(64,4,4),改完之后变为[32,4,4]
        # self.cnn_layers=[convolution_1,pooling_1,convolution_2,pooling_2]
        # #建立全连接神经网络
        # #self.fc_layers=FC_Network([1024,10])
        # self.fc_layers=FC_Network([512,300,10])
        #建立卷积神经网络
        convolution_1=ConvolutionLayer(28,28,1,5,5,8,0,1,ReluActivator(),0.001)
        pooling_1=MaxPoolingLayer(24,24,8,2,2,2)
        convolution_2=ConvolutionLayer(12,12,8,5,5,16,0,1,ReluActivator(),0.001)
        pooling_2=MaxPoolingLayer(8,8,16,2,2,2)#最后输出的图像为(20,4,4)
        self.cnn_layers=[convolution_1,pooling_1,convolution_2,pooling_2]
        #建立全连接神经网络
        #self.fc_layers=FC_Network([1024,10])
        self.fc_layers=FC_Network([256,150,10])
    def predict(self,sample):
        '''
        使用卷积神经网络实现预测
        sample：输入样本
        '''
        output=sample
        #先用卷积神经网络提取特征
        for layer in self.cnn_layers:
            layer.forward(output)
            output=layer.output_array
        #将多维数组变为一维
        #cnn_output=self.cnn_layers[-1].output_array#取得卷积层最后的输出
        output=output.reshape( output.size,1 )#转化为numpy的一维数组
        #print "cnn_output is ",output
        # print "output.ndim is ",output.ndim
        # print "output.shape is ",output.shape
        # print "len(output) is ",len(output)
        #再用全连接神经网络对一维数组进行预测
        for layer in self.fc_layers.layers:
            # print "output is ",output
            layer.forward(output)
            output=layer.output
        # print "last output is ",output
        return output
    def train(self,labels,data_set,rate,epoch):
        '''
        训练函数
        lables:样本标签
        data_set：输入样本
        rate：学习速率
        epoch：训练轮数
        '''
        for i in range(epoch):#训练的轮数
            for d in range(len(data_set)):#依次取得相应的训练特征和训练标记
                print "INFO: Time:%s,data_set is %d starting....." % (datetime.now(),d)
                # print "labels is ",labels[d]
                # print "true label is ",labels[d].argmax()
                self.train_one_sample(data_set[d],labels[d],rate)
    def train_one_sample(self,sample,label,rate):
        '''
        用一个样本训练神经网络
        '''
        #print "INFO: Start predict data_set..."
        self.predict(sample)
        #print "INFO: Calc_gradient data_set & label..."
        self.calc_gradient(label)
        #print "INFO: Update_weight rate..."
        self.update_weight(rate)
    def calc_gradient(self,label):
        '''
        逆向计算敏感度
        '''
        #print "self.fc_layers.layers[-1].output is ",self.fc_layers.layers[-1].output
        #print "label is ",label
        delta=self.fc_layers.layers[-1].activator.backward(self.fc_layers.layers[-1].output)*\
            (label-self.fc_layers.layers[-1].output)#计算最后一层的delta
        #在全连接神经网络，逆向计算敏感度
        #print "len(delta) is ",len(delta)
        #print "delta.shape is ",delta.shape
        #print "delta is ",delta
        for layer in self.fc_layers.layers[::-1]:
            layer.backward(delta)
            delta=layer.delta
        #把一维数组转化为三维
        #重塑矩阵的形状
        delta=delta.reshape(self.cnn_layers[-1].output_array.shape)
        #在卷积神经网络，逆向计算敏感度
        cnn_layers_num=len(self.cnn_layers)
        for layer in range(cnn_layers_num)[::-1]:
            #print "layer is ",layer
            if layer%2!=0:#池化层往后计算敏感度
                self.cnn_layers[layer].backward(self.cnn_layers[layer].input_array,delta)
            else:#卷积层往后计算敏感度
                self.cnn_layers[layer].backward(self.cnn_layers[layer].input_array,delta,self.cnn_layers[layer].activator)
            delta=self.cnn_layers[layer].delta_array
        return delta
    def update_weight(self,rate):
        '''
        更新权重
        '''
        cnn_layers_num=len(self.cnn_layers)
        for layer in range(cnn_layers_num)[::2]:
            # print "before cnn weights , self.cnn_layers[layer].weights is ",self.cnn_layers[layer].filters
            self.cnn_layers[layer].update()#学习率已经在建立类对象的时候已经给出类
            # print "after cnn weights , self.cnn_layers[layer].weights is ",self.cnn_layers[layer].filters
        for layer in self.fc_layers.layers:
            # print "before fc weights, elf.fc_layers.layers is ",layer.W
            layer.update(rate)
            # print "after fc weights, elf.fc_layers.layers is ",layer.W
    def dump(self):
        '''
        '''
        for layer in self.layers:
            layer.dump()
    def loss(self,output,label):
        '''
        计算神经网络的损失
        '''
        return 0.5*((label-output)*(label-output)).sum()



















# def init_test():
#     #a为输入
#     a = np.array( \
#         [[[0,1,1,0,2],\
#           [2,2,2,2,1],\
#           [1,0,0,2,0],\
#           [0,1,1,0,0],\
#           [1,2,0,0,2]],\
#          [[1,0,2,2,0],\
#           [0,0,0,2,0],\
#           [1,2,1,2,1],\
#           [1,0,0,0,0],\
#           [1,2,1,1,1]],\
#          [[2,1,2,0,0],\
#           [1,0,0,1,0],\
#           [0,2,1,0,1],\
#           [0,1,2,2,2],\
#           [2,1,0,0,1]]])
#     #b为权重,错了❌
#     b = np.array( \
#         [[[0,1,1],\
#           [2,2,2],\
#           [1,0,0]],\
#          [[1,0,2],\
#           [0,0,0],\
#           [1,2,1]]])
#     cl=ConvolutionLayer(5,5,3,3,3,2,1,2,IdentityActivator(),0.001)
#     cl.filters[0].weights=np.array(\
#         [[[-1,1,0],\
#           [0,1,0],\
#           [0,1,1]],\
#          [[-1,-1,0],\
#           [0,0,0],\
#           [0,-1,0]],\
#          [[0,0,-1],\
#           [0,1,0],\
#           [1,-1,-1]]], dtype=np.float64)
#     cl.filters[0].bias=1
#     cl.filters[1].weights = np.array(\
#         [[[1,1,-1],\
#           [-1,-1,1],\
#           [0,-1,1]],\
#          [[0,1,0],\
#          [-1,0,-1],\
#           [-1,1,0]],\
#          [[-1,0,0],\
#           [-1,0,1],\
#           [-1,0,0]]], dtype=np.float64)
#     return a,b,cl
#
# def test():
#     a,b,cl=init_test()
#     cl.forward(a)
#     print cl.output_array
#
# def test_bp():
#     a,b,cl=init_test()
#     cl.backward(a,b,IdentityActivator())
#     cl.update()
#     print cl.filters[0]
#     print cl.filters[1]
#
# def gradient_check():
#     '''
#     梯度检查
#     '''
#     #涉及一个误差函数，取所有节点输出项之和
#     error_function=lambda o:o.sum()
#     #计算forward值
#     a,b,cl=init_test()
#     cl.forward(a)
#     #求取sensitivity map,是一个全1数组
#     sensitivity_array=np.ones(cl.output_array.shape,dtype=np.float64)
#     #计算梯度
#     cl.backward(a,sensitivity_array,IdentityActivator())
#     #计算梯度
#     epsilon=10e-4
#     for d in range(cl.filters[0].weights_gradient.shape[0]):
#         for i in range(cl.filters[0].weights_gradient.shape[1]):
#             for j in range(cl.filters[0].weights_gradient.shape[2]):
#                 cl.filters[0].weights[d,i,j]+=epsilon
#                 cl.forward(a)
#                 err1=error_function(cl.output_array)
#                 cl.filters[0].weights[d,i,j]-=2*epsilon
#                 cl.forward(a)
#                 err2=error_function(cl.output_array)
#                 expect_grad=(err1-err2)/(2*epsilon)
#                 cl.filters[0].weights[d,i,j]+=epsilon
#                 print 'weights(%d,%d,%d): expected - actural %f - %f' % (\
#                     d, i, j, expect_grad, cl.filters[0].weights_gradient[d,i,j])
# gradient_check()
# def init_pool_test():
#     a = np.array(
#         [[[1,1,2,4],
#           [5,6,7,8],
#           [3,2,1,0],
#           [1,2,3,4]],
#          [[0,1,2,3],
#           [4,5,6,7],
#           [8,9,0,1],
#           [3,4,5,6]]], dtype=np.float64)
#
#     b = np.array(
#         [[[1,2],
#           [2,4]],
#          [[3,5],
#           [8,2]]], dtype=np.float64)
#
#     mpl=MaxPoolingLayer(4,4,2,2,2,2)
#
#     return a,b,mpl
# def test_pool():
#     '''
#     '''
#     a,b,mpl=init_pool_test()
#     mpl.forward(a)
#     print 'input array:\n%s\noutput array:\n%s' % (a,\
#         mpl.output_array)
# def test_pool_bp():
#     '''
#     '''
#     a,b,mpl=init_pool_test()
#     mpl.backward(a,b)
#     print 'input array:\n%s\nsensitivity array:\n%s\ndelta array:\n%s' % (
#         a, b, mpl.delta_array)
