# -*- coding:utf-8 -*-
from FullConnectedLayer import *
import numpy as np
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
