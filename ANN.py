#-*- coding:utf-8 -*-
import random
import math
from numpy import *
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
class Node(object):
    def __init__(self,layer_index,node_index):
        '''
        构造节点对象
        layer_index:节点所属层的编号
        node_index:节点在该层的编号
        '''
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.upstream=[]
        self.output=0
        self.delta=0#损失函数对输入的偏导数
    def set_output(self,output):
        '''
        设置节点的输出值。如果节点属于输入层则回用到这个函数
        '''
        self.output=output
    def append_downstream_connection(self,conn):
        '''
        添加一个到下游节点的连接
        '''
        self.downstream.append(conn)
    def append_upstream_connection(self,conn):
        '''
        添加一个上游节点连接
        '''
        self.upstream.append(conn)
    def calc_output(self):
        '''
        计算节点的输出
        '''
        input_val=reduce(lambda ans,conn:ans+conn.weight*conn.upstream_node.output,self.upstream,0)
        self.output=sigmoid(input_val)
    def calc_hidden_layer_delta(self):
        '''
        节点属于隐藏层时，根据ai(1-ai)sum(Wki*DELTAk)计算delta
        '''
        downstream_delta_sum=reduce(lambda ans,conn:ans+conn.weight*conn.downstream_node.delta,self.downstream,0.0)
        self.delta=self.output*(1-self.output)*downstream_delta_sum
    def calc_output_layer_delta(self,label):
        '''
        节点属于输出层，根据Ai(1-Ai)(Ti-Ai)计算delta
        '''
        self.delta=self.output*(1-self.output)*(label-self.output)
    def __str__(self):
        '''
        打印节点的信息
        '''
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str
class ConstNode(object):
    def __init__(self,layer_index,node_index):
        '''
        在每一层设置一个输出恒为1的节点
        '''
        self.layer_index=layer_index
        self.node_index=node_index
        #存储下游节点的连接。没有上游节点的连接
        self.downstream=[]
        self.output=1
        self.delta=0.0
    def append_downstream_connection(self,conn):
        '''
        添加一个下游节点的连接。无上游节点的连接
        '''
        self.downstream.append(conn)
    def calc_hidden_layer_delta(self):
        '''
        节点属于隐藏层，计算节点的delta
        '''
        downstream_delta_sum=reduce(lambda ans,conn:ans+conn.weight*conn.downstream_node.output,self.downstream,0.0)
        self.delta=self.output*(1-self.output)*downstream_delta_sum
    def __str__(self):
        '''
        打印节点的信息
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str
class Layer(object):
    def __init__(self,layer_index,node_count):
        self.layer_index=layer_index
        self.nodes=[]#存储该层中的节点
        for i in range(node_count):#初始化该层中的节点
            self.nodes.append(Node(layer_index,i))
        #每一层添加一个输出值恒为1的节点
        self.nodes.append(ConstNode(layer_index,node_count))
    def set_output(self,data):
        '''
        设置层的输出。当层是输入层的时候会用到
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])
    def calc_output(self):
        '''
        计算该层中节点的输出
        最后一个节点的输出值恒为1，不需要计算
        '''
        for node in self.nodes[:-1]:
            node.calc_output()
    def dump(self):
        '''
        打印层的信息
        '''
        for node in self.nodes:
            print node
class Connection(object):
    def __init__(self,upstream_node,downstream_node):
        '''
        downstream_node:连接的l层节点
        upstream_node:连接的l-1层节点
        '''
        self.downstream_node=downstream_node
        self.upstream_node=upstream_node
        self.weight=random.uniform(-0.1,0.1)
        self.gradient=0.0
    def calc_gradient(self):
        '''
        计算梯度
        '''
        self.gradient=self.downstream_node.delta*self.upstream_node.output
    def get_gradient(self):
        '''
        获取当前的梯度
        '''
        return self.gradient
    def update_weight(self,rate):
        '''
        根据梯度下降，更新连接的权重（weight）
        '''
        self.calc_gradient()
        self.weight=self.weight+rate*self.gradient
    def __str__(self):
        '''
        打印连接信息
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)
class Connections(object):
    def __init__(self):
        self.connections=[]
    def add_connection(self,conn):
        '''
        添加一条连接
        '''
        self.connections.append(conn)
    def dump(self):
        for conn in self.connecions:
            print conn
class Network(object):
    def __init__(self,layers):
        '''
        初始化一个全连接的神经网络
        layers：一维数组，描述神经网络每层节点数
        '''
        self.connections=Connections()
        self.layers=[]
        layer_count=len(layers)
        node_count=0
        for i in range(layer_count):
            self.layers.append(Layer(i,layers[i]))
        for layer in range(layer_count-1):
            connections=[ \
                Connection(upstream_node,downstream_node)   \
                for upstream_node in self.layers[layer].nodes   \
                for downstream_node in self.layers[layer+1].nodes[:-1]   ]
            for conn in connections:
                self.connections.add_connection(conn)
                # print conn.upstream_node.layer_index,":",conn.upstream_node.node_index
                # print conn.downstream_node.layer_index,":",conn.downstream_node.node_index
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)
    def train(self,labels,data_set,rate,iteration):
        '''
        训练神经网络
        lables：数组，训练样本标签。每个元素是一个样本的标签
        data_set:二维数组，训练样本特征。每个元素是一个样本的特征
        '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],data_set[d],rate)
    def train_one_sample(self,label,sample,rate):
        '''
        用一个样本训练网络。随机梯度
        '''
        predict_value=self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)
    def predict(self,sample):
        '''
        根据输入的样本预测输出值
        sample：数组，样本的特征，网络的输入
        '''
        self.layers[0].set_output(sample)
        for i in range(1,len(self.layers)):
            self.layers[i].calc_output()#依次计算每一层的输出
        return map(lambda node: node.output,self.layers[-1].nodes[:-1])
    def calc_delta(self,label):
        '''
        计算每个节点的delta。从后往前计算
        '''
        output_nodes=self.layers[-1].nodes
        for i in range(len(label)-1):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()
    def update_weight(self,rate):
        '''
        更新每个连接的权重
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)
    def calc_gradient(self):
        '''
        计算每个连接的梯度
        '''
        for layer in self.layers[::-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()
    def get_gradient(self,label,sample):
        '''
        获得网络在一个样本下，每个连接上的梯度
        label：样本的标签
        sample：样本的输入
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()
    def dump(self):
        '''
        打印网络信息
        '''
        for layer in self.layers:
            layer.dump()
#梯度检查
def gradient_check(network,sample_feature,sample_label):
    '''
    检查梯度
    network:神经网络的对象
    sample_feature:样本的特征
    sample_label:样本的标记
    '''
    #计算网络的
    network_error=lambda vec1,vec2:\
                0.5*reduce(lambda ans,b:ans+b,\
                map(lambda v:(v[0]-v[1])*(v[0]-v[1]),\
                zip(vec1,vec2)))
    #获取网络的各个连接的梯度
    network.get_gradient(sample_feature,sample_label)
    for conn in network.connections.connections:
        #获取指定的真实的梯度值
        actual_gradient=conn.get_gradient()
        #增加一个很小的值，计算网络的误差
        epsilon=0.0001
        conn.weight+=epsilon
        error1=network_error( network.predict(sample_feature),sample_label )
        #减去一个很小的值，计算网络的误差
        conn.weight-=2*epsilon
        error2=network_error(network.predict(sample_feature),sample_label)
        #估计的梯度值
        expected_gradient=(error2-error1)/(2*epsilon)
        # 打印
        print 'expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient)
if __name__=="__main__":
    network=Network([3,4,2])
    gradient_check(network,[1,2,3],[4,5])
