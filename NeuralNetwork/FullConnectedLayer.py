# -*- coding:UTF-8 -*-
import random
import numpy as np
from datetime import datetime
#Relu激活函数
class ReluActivator(object):
    def forward(self,weighted_input):
        return max(0,weighted_input)

    def backward(self,output):
        return 1 if output>0 else 0

#Sigmoid激活函数类
class SigmoidActivator(object):
    def forward(self,weight_input):
        '''
        '''
        return 1.0/(1.0+np.exp(-weight_input))
    def backward(self,output):
        '''
        '''
        return output*(1-output)
class FullConnectedLayer(object):
    def __init__(self,input_size,output_size,activator):
        '''
        构造函数
        input_size:本层输入向量的纬度
        output_size:本层输出向量的纬度
        activator:激活函数
        '''
        self.input_size=input_size
        self.output_size=output_size
        self.activator=activator
        #权重数组W
        self.W=np.random.uniform(-0.1,0.1,(output_size,input_size))
        #偏置项
        self.b=np.zeros((output_size,1))
        #输出向量
        self.output=np.zeros( (output_size,1) )
    def forward(self,input_array):
        '''
        前向计算
        input_array:输入向量，纬度必须等于input_size
        '''
        #
        self.input=input_array
        #print "fc_layers input_array is ",input_array
        #print "np.dot(self.W,input_array)+self.b is ",np.dot(self.W,input_array)+self.b
        # print "self.W.shape is ",self.W.shape
        # print "input_array.shape is ",input_array.shape
        # print "self.b is ",self.b
        self.output=self.activator.forward( np.dot(self.W,input_array)+self.b )
        #print "test is ",np.dot(self.W,input_array)
    def backward(self,delta_array):
        '''
        反向计算W和b
        delta_array:从下一层传递过来的误差项
        '''
        #self.W.T为权重举矩阵的转置
        #self.W_grad:为权重的梯度。grad为gradient
        #计算本节点的敏感度
        #print "delta_array is ",delta_array
        self.delta=self.activator.backward(self.input)*np.dot(self.W.T,delta_array)
        self.W_grad=np.dot(delta_array,self.input.T)#二维矩阵
        self.b_grad=delta_array

    def update(self,learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W+=learning_rate*self.W_grad
        self.b+=learning_rate*self.b_grad
    def dump(self):
        print "W: %s\nb: %s" % (self.W,self.b)
class FC_Network(object):
    def __init__(self,layers):
        '''
        构造函数
        '''
        self.layers=[]
        for i in range( len(layers)-1 ):
            self.layers.append( \
             FullConnectedLayer(layers[i],layers[i+1],SigmoidActivator())
            )
        # for i in range( len(layers)-1 ):
        #     self.layers.append( \
        #      FullConnectedLayer(layers[i],layers[i+1],ReluActivator())
        #     )
    def predict(self,sample):
        '''
        使用神经网络实现预测
        sample：输入样本
        '''
        output=sample
        for layer in self.layers:
            #print "output is ",output
            layer.forward(output)
            output=layer.output
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
                #print "INFO: Time:%s,data_set is %d starting....." % (datetime.now(),d)
                self.train_one_sample(labels[d],data_set[d],rate)
    def train_one_sample(self,label,sample,rate):
        '''
        用一个样本训练神经网络
        '''
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)
    def calc_gradient(self,label):
        '''
        计算梯度
        '''
        delta=self.layers[-1].activator.backward(self.layers[-1].output)*\
            (label-self.layers[-1].output)#最后一层的delta
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta=layer.delta
        return delta
    def update_weight(self,rate):
        '''
        更新权重
        '''
        for layer in self.layers:
            layer.update(rate)
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
    def gradient_check(self,sample_feature,sample_label):
        '''
        梯度检查
        network：神经网络对象
        sample_feature：样本的特征
        sample_label:样本的标签
        '''
        #获取网络的当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)
        #检查梯度
        epsilon=10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):#得到矩阵W的行数
                for j in range(fc.W.shape[1]):#得到矩阵W的列数
                    fc.W[i,j]+=epsilon
                    output=self.predict(sample_feature)
                    err1=self.loss(sample_label,output)
                    fc.W[i,j]-=2*epsilon
                    output=self.predict(sample_feature)
                    err2=self.loss(sample_label,output)
                    expect_gradient=(err1-err2)/(2*epsilon)
                    fc.W[i,j]+=epsilon
                    print "weight(%d,%d):expected-actural %.4e-%.4e" % \
                            (i,j,expect_gradient,fc.W_grad[i,j])
def transpose(args):
    '''
    '''
    # print "args is ",args
    return map(lambda arg:\
            map( lambda line:np.array(line).reshape(len(line),1),arg),\
            args)
class Normalizer(object):
    def __init__(self):
        '''
        '''
        self.mask=[0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80]
    def norm(self,number):
        '''
        '''
        data=map( lambda m:0.9 if number & m else 0.1,self.mask )
        # print data
        return np.array(data).reshape(8,1)#变成一个二维数组
    def denorm(self,vec):
        '''
        '''
        binary=map( lambda i:1 if i>0.5 else 0,vec[:,0])#变成0，1序列
        for i in range(len(self.mask)):#将对应位的权重相乘
            binary[i]=binary[i]*self.mask[i]
        return reduce(lambda x,y:x+y,binary)#最后将各个位上数相加即为原来的数字
def train_data_set():
    '''
    '''
    normalizer=Normalizer()
    data_set=[]
    labels=[]
    for i in range(0,256):
        n=normalizer.norm(i)
        data_set.append(n)
        labels.append(n)
    return labels,data_set
def correct_ratio(network):
    '''
    计算正确率
    '''
    normalizer=Normalizer()
    correct=0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i)))==i:
            correct+=1.0
    print "correct_ratio: %.2f%%" % (correct/256*100)
def test():
    '''
    '''
    labels,data_set=transpose(train_data_set())
    net=Network([8,3,8])
    rate=0.5
    mini_batch=20
    epoch=10
    for i in range(epoch):
        net.train(labels,data_set,rate,mini_batch)
        print 'after epoch %d loss: %f' % ( \
            (i + 1),\
            net.loss(labels[-1], net.predict(data_set[-1])) \
        )
        rate/=2
        correct_ratio(net)
def gradient_check():
    '''
    梯度检查
    '''
    labels,data_set=transpose(train_data_set())
    #print "labels is ",labels
    #print "data_set is ",data_set
    net=FC_Network([8,3,8])
    net.gradient_check(data_set[0],labels[0])
    return net
if __name__=="__main__":
    gradient_check()
