以识别MNIST数据集练习神经网络<br>
ANN.py:
	按照面向对象的思想实现全连接神经网络，定义Node，layer，Connection对象，并定义对外的接口类Network。并含梯度检查的代码。
FullConnectedLayer.py:
	ANN.py代码运行慢。因此进行向量化改造，以加快运行速度，并定义对外接口类FC_Network
MNIST_FC.py:
	定义全连接神经网络读取MNIST文件方式，并调用FullConnectedLayer.py文件中FC_Network类中的各方法进行训练和测试。
CNN.py:
	卷积神经网络实现。对外接口类ConvolutionNeuralNetwork。
MNIST_CNN.py:
	定义卷积神经网络读取MNIST文件方式，并调用CNN.py文件中ConvolutionNeuralNetwork类中的各方法进行训练和测试。
识别结果:
	全连接神经网络最低错误率：0.049600。
	卷积神经网络最低错误率：0.255300。卷积神经网络有大量的循环，运行时间还是较慢。通过增加Feature map的数量，应该还能使错误率进一步降低。