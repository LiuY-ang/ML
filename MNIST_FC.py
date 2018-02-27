#!/usr/bin/env python
#-*- coding:UTF-8 -*-
import struct
#from ANN import *
from FullConnectedLayer import *
from datetime import datetime
def transpose(args):
    '''
    '''
    return map(lambda arg:\
            map( lambda line:np.array(line).reshape(len(line),1),arg),\
            args)
#数据加载基类
class Loader(object):
    def __init__(self,path,count):
        '''
        初始化加载起
        path：数据文件路径
        count：文件中的样本个数
        '''
        self.path=path
        self.count=count
    def get_file_content(self):
        '''
        读取文件内容
        '''
        input_file=open(self.path,"rb")
        content=input_file.read()
        return content
    def to_int(self,byte):
        '''
        将unsigned byte字符转换为整数
        '''
        return struct.unpack('B',byte)[0]
#图像数据加载器
class ImageLoader(Loader):
    def get_picture(self,content,index):
        '''
        内部函数，从文件中获取图像。返回值是二维数组
        '''
        start=index*28*28+16
        picture=[]
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(self.to_int(content[start+28*i+j])  )
        return picture
    def get_one_sample(self,picture):
        '''
        内部函数，将图像按照行优先转化为样本的输入向量。将二维数组转化为一位数组
        '''
        sample=[]
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample
    def load(self):
        '''
        记载数据文件，获得全部样本的输入向量
        '''
        content=self.get_file_content()
        data_set=[]
        for index in range(self.count):
            data_set.append( self.get_one_sample( self.get_picture(content,index) ) )
        return data_set
#标签数据加载器
class LabelLoader(Loader):
    def load(self):
        '''
        加载数据文件，获得全部样本的标签向量
        '''
        content=self.get_file_content()
        labels=[]
        for index in range(self.count):
            labels.append(self.norm(content[index+8]))
        return labels
    def norm(self,label):
        '''
        内部函数，将一个值转换为10维标签向量
        '''
        label_vec=[]
        label_value=self.to_int(label)
        for i in range(10):
            if i==label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec
def get_training_data_set():
    '''
    获得训练数据集
    '''
    image_loader=ImageLoader("MNIST/train-images.idx3-ubyte",60000)
    label_loader=LabelLoader("MNIST/train-labels.idx1-ubyte",60000)
    return image_loader.load(),label_loader.load()
def get_test_data_set():
    '''
    获得测试数据集
    '''
    image_loader=ImageLoader("MNIST/t10k-images.idx3-ubyte",10000)
    label_loader=LabelLoader("MNIST/t10k-labels.idx1-ubyte",10000)
    return image_loader.load(),label_loader.load()
def get_result(vec):
    '''
    从输出向量中寻找结果
    '''
    max_value_index=0
    max_value=0
    for i in range(len(vec)):
        if vec[i]>max_value:
            max_value=vec[i]
            max_value_index=i
    return max_value_index
def evaluate(network,test_data_set,test_labels):
    '''
    用测试集来测试神经网络。
    '''
    error=0
    total=len(test_data_set)
    for i in range(total):
        label=get_result(test_labels[i])#取的正确结果
        predict=get_result(network.predict(test_data_set[i]))#预测结果
        if label!=predict:
            error+=1#统计错误的个数
    return float(error)/float(total)#计算错误率
def save_connection_weights(path,network):
    '''
    保存神经网络的权重参数
    '''
    save_file=open(path)
    layer_length=len(network.layers)
    for index in range(layer_length):
        for node in network.layers[index]:
            for conn in node.downstream:
                save_file.write(conn.weight)

def train_and_evaluate():
    '''
    '''
    last_error_ratio=1.0
    epoch=0
    train_data_set,train_labels=transpose(get_training_data_set() )
    # print train_data_set[0]
    # print train_labels[0]
    print "INFO: Train data set read finished-------------------"
    test_data_set,test_labels=transpose( get_test_data_set() )
    print "INFO: Test data set read finished********************"
    network=FC_Network([784,200,10])
    print "INFO: Network has been built+++++++++++++++++++++++++"
    while True:
         epoch+=1
         #只用所有的数据训练一轮
         network.train(train_labels,train_data_set,0.001,1)
         print "%s epoch=%d finished, loss %f" % (datetime.now(),epoch,\
                network.loss(train_labels[-1],network.predict(train_data_set[-1])))
        #  if epoch%2==0:
         error_ratio=evaluate(network,test_data_set,test_labels)
         print "%s after epoch=%d,error_ratio is %f" % (datetime.now(),epoch,error_ratio)
         if error_ratio>last_error_ratio:
             break
         else:
             last_error_ratio=error_ratio
if __name__=="__main__":
    print "%s start training data_set" % (datetime.now())
    train_and_evaluate()
