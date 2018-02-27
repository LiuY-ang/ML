以识别MNIST数据集练习神经网络<br>
ANN.py:<br>
&nbsp;&nbsp;&nbsp;按照面向对象的思想实现全连接神经网络，定义Node，layer，Connection对象，并定义对外的接口类Network。并含梯度检查的代码。<br>
FullConnectedLayer.py:<br>
&nbsp;&nbsp;&nbsp;ANN.py代码运行慢。因此进行向量化改造，以加快运行速度，并定义对外接口类FC_Network<br>
MNIST_FC.py:<br>
&nbsp;&nbsp;&nbsp;定义全连接神经网络读取MNIST文件方式，并调用FullConnectedLayer.py文件中FC_Network类中的各方法进行训练和测试。<br>
CNN.py:<br>
&nbsp;&nbsp;&nbsp;卷积神经网络实现。对外接口类ConvolutionNeuralNetwork。<br>
MNIST_CNN.py:<br>
&nbsp;&nbsp;&nbsp;定义卷积神经网络读取MNIST文件方式，并调用CNN.py文件中ConvolutionNeuralNetwork类中的各方法进行训练和测试。<br>
识别结果:<br>
&nbsp;&nbsp;&nbsp;全连接神经网络最低错误率：0.049600。<br>
&nbsp;&nbsp;&nbsp;卷积神经网络最低错误率：0.255300。卷积神经网络有大量的循环，运行时间还是较慢。通过增加Feature map的数量，应该还能使错误率进一步降低。<br>