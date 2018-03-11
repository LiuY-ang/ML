# -*- coding:utf-8
import numpy as np
'''
-生成实例：clf=DecisionTree().参数mode可选，ID3或者C4.5，默认为C4.5
-训练，调用fit方法：clf.fit(X,y)。X,y均为np.ndarray类型
-预测，调用predict方法：clf.predict(X)。X为np.ndarray类型
-可视化决策树，调用showTree方法
'''
class DecisionTree(object):
    def __init__(self,mode='C4.5'):
        '''
        '''
        self._tree=None
        if mode=='C4.5' or mode=='ID3':
            self._mode=mode
        else:
            raise Exception('mode must be C4.5 or ID3')
    def _calEntropy(self,y):
        '''
        计算熵（在某个属性的值取定之后，该集合为y，计算该集合的信息熵）
        信息熵计算公式：Entroy(D)=sum(pi*log2(pi)),i in labels。
        参数y:数据集的标签。y的数据类型
        '''
        num=y.shape[0]#获取样本个数
        labelCounts={}#统计y中不同label值的个数，并用labelCounts存储
        for label in y:
            # print "_calEntropy ",label
            if label not in labelCounts.keys():
                labelCounts[label]=0
            labelCounts[label]+=1
        #计算熵
        entropy=0.0
        for key in labelCounts:
            prob=float(labelCounts[key])/num
            entropy-=prob*np.log2(prob)
        return entropy
    def _splitDataSet(self,X,y,index,value):
        '''
        index:bestFeatureIndex,最佳分割属性的下标
        value:最佳分割属性的某个具体的值
        返回属性的下标为index的某个确定的属性值value的集合
        '''
        ret=[]
        featVec=X[:,index]
        X=X[:,[i for i in range(X.shape[1]) if i!=index ]]
        for i in range(len(featVec)):
            if featVec[i]==value:
                ret.append(i)
        return X[ret,:],y[ret]
    def _chooseBestFeatureToSplit_ID3(self,X,y):
        '''
        对输入数据，根据信息增益(ID3)选择最佳分割特征
        dataSet:数据集，最后一列为label
        numFeatures:特征个数
        oldEntropy:原始数据集的熵
        newEntropy:按某个特征分割数据集后的熵
        infoGain:信息增益
        bestInfoGain:记录最大的信息增益
        bestFeatureIndex:信息增益最大时，所选择的分割特征的下标
        '''
        numFeatures=X.shape[1]#特征个数
        oldEntropy=self._calcEntropy(y)#原始数据集的熵
        bestInfoGain=0.0
        beatFeatureIndex=-1
        #计算每个特征都计算一下infoGain，并用bestInfoGain记录最大的那个
        for i in range(numFeatures):
            featList=X[:,i]
            uniqueVals=set(featList)#去除featList中的重复元素
            newEntropy=0.0
            #对第i个特征的各个value，得到各个子数据集，计算各个子数据集的熵
            #进一步地计算得到根据第i个特征分割原始数据集后的熵newEntropy
            for value in uniqueVals:
                sub_X,sub_y=self._splitDataSet(X,y,i,value)
                prob=len(sub_y)/float(len(y))#某个属性的值为value所占的比例
                newEntropy+=prob*self._calcEntropy(sub_y)
            #计算信息增益，根据信息增益选择最佳分割特征
            infoGain=oldEntropy-newEntropy
            if infoGain>bestInfoGain:
                bestInfoGain=infoGain
                bestFeatureIndex=i
        return bestFeatureIndex
    def _chooseBestFeatureToSplit_C45(self,X,y):
        '''
        对输入数据，根据信息增益率(C4.5)选择最佳分割特征
        ID3算法计算信息增益，C4.5算法计算的是信息增益比
        '''
        numFeatures=X.shape[1]
        oldEntropy=self._calEntropy(y)
        bestGainRatio=0.0
        bestFeatureIndex=-1
        #对每个特征都计算一个gainRatio=infoGain/splitInformation
        for i in range(numFeatures):
            featList=X[:,i]
            uniqueVals=set(featList)
            newEntropy=0.0
            splitInformation=0.0
            #对第i个特征的各个value，得到各个子数据集，计算各个子数据集的熵
            #进一步地计算得到根据第i个特征分割原始数据集后的熵newEntropy
            for value in uniqueVals:
                sub_X,sub_y=self._splitDataSet(X,y,i,value)
                prob=len(sub_y)/float(len(y))
                newEntropy+=prob*self._calEntropy(sub_y)
                splitInformation-=prob*np.log2(prob)#计算分母
            #计算信息增益比，根据信息增益比选择最佳分割特征
            #splitInformation若为0，说明该特征的所有值都是相同的，显然不能作为特征
            if splitInformation==0.0:
                pass
            else:
                infoGain=oldEntropy-newEntropy#信息增益
                gainRatio=infoGain/splitInformation#信息增益率
                if(gainRatio>bestGainRatio):
                    bestGainRatio=gainRatio
                    bestFeatureIndex=i
        return bestFeatureIndex

    def _majorityCnt(self,labelList):
        '''
        返回labelList中出现次数最多的label
        '''
        labelCount={}
        for vote in labelList:
            if vote not in labelCount.keys():
                labelCount[vote]=0
            labelCount[vote]+=1
        #排序
        sortedClassCount=sorted(labelCount.iteritems(),key=lambda x:x[1],reverse=True)
        return sortedClassCount[0][0]

    def _createTree(self,X,y,featureIndex):
        '''
        建立决策树
        feature,类型是元组，它记录了X中的特征在原始数据中对应的下标
        '''
        # print "in _createTree"
        labelList=list(y)
        #下面两条语句是递归结束条件
        #所有label都相同的话，则停止分割，返回该label
        if labelList.count(labelList[0])==len(labelList):
            return labelList[0]
        #没有特征可分割时，停止分割，返回出现次数最多的label
        if len(featureIndex)==0:
            return self._majorityCnt(labelList)

        #如果可以继续分割的话说，确定最佳分割特征的下标
        if self._mode=='C4.5':
            bestFeatIndex=self._chooseBestFeatureToSplit_C45(X,y)
        elif self._mode=='ID3':
            bestFeatIndex=self._chooseBestFeatureToSplit_ID3(X,y)
        bestFeatStr=featureIndex[bestFeatIndex]#获得最佳分割特征
        featureIndex=list(featureIndex)
        featureIndex.remove(bestFeatStr)
        featureIndex=tuple(featureIndex)
        #使用字典存储决策树。最佳分割特征作为key，而对应的键值仍然是一棵树（仍然用字典存储）
        myTree={ bestFeatStr:{} }
        featValues=X[:,bestFeatIndex]
        uniqueVals=set(featValues)#去除重复属性值
        for value in uniqueVals:
            #对uniqueVals中的每个value递归地创建树
            sub_X,sub_y=self._splitDataSet(X,y,bestFeatIndex,value)
            #对该属性的属性值为value建立一个分支，该分支上的数据集为sub_X,sub_y
            myTree[bestFeatStr][value]=self._createTree(sub_X,sub_y,featureIndex)
        return myTree
    def fit(self,X,y):
        '''
        训练决策树树
        X:二维数组
        y:一维数组
        '''
        if isinstance(X,np.ndarray) and isinstance(y,np.ndarray):
            pass
        else:
            try:
                X=np.array(X)
                y=np.array(y)
            except:
                raise TypeError("numpy.ndarray required for X,y")
        #生成属性集合的索引
        featureIndex=tuple([ 'x'+str(i) for i in range(X.shape[1]) ])
        #经过处理后的featureIndex为('x0', 'x1', 'x2', 'x3', 'x4')
        # print "start create tree..."
        self._tree=self._createTree(X,y,featureIndex)
        # print "fit ",type(self._tree)
        # print self._tree
        return self
    def predict(self,X):
        '''
        '''
        if self._tree==None:
            raise NotFittedError("Estimator not fitted,call fit first" )
        #类型检查
        if isinstance(X,np.ndarray):
            pass
        else:
            try:
                X=np.array(X)
            except:
                raise  TypeError("numpy.ndarray required for X")
        def _classify(tree,sample):
            '''
            用训练好的决策树对输入数据分类
            决策树的构建是一个递归的过程，用决策树分类也是一个递归的过程
            _classify()一次只能对一个样本(sample)分类
            '''
            featureIndex=tree.keys()[0]#dict.keys()返回的是list,tree.keys[0]取到第一个元素
            secondDict=tree[featureIndex]#获得下一层的dict
            key=sample[int(featureIndex[1:])]#取到样本featureIndex的属性值
            valueOfKey=secondDict[key]#valueOfKey的类型为dict
            if isinstance(valueOfKey,dict):
                label=_classify(valueOfKey,sample)
            else:
                label=valueOfKey
            return label

        if len(X.shape)==1:
            return _classify(self._tree,X)
        else:
            results=[]
            for i in range(X.shape[0]):
                results.append(_classify(self._tree,X[i]))
            return np.array(results)
    def show(self):
        if self._tree==None:
            raise NotFittedError("Estimator not fitted,call fit first")
        #plot the tree using matplotlib
        import treePlotter
        treePlotter.createPlot(self._tree)
class NotFittedError(Exception):
    '''
    Exception class to raise if estimator is used before fitting
    '''
    pass
if __name__=="__main__":
    data_set=[ ["mid","high","high","high"],["mid","mid","high","high"],["low","high","high","high"] ]
    labels=[True,True,False]
    dt=DecisionTree()
    print "Start fit......"
    dt.fit(data_set,labels)
    print "Finish fit!"
    test_set=[["mid","high","high","high"]]
    test_labels=[True]
    if dt.predict(test_set):
        print "DecisionTree predict is right!"
    else:
        print "DecisionTree predict is wrong!"
