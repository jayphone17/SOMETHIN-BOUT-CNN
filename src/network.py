# 一个实现随机梯度下降学习的模块
# 一种前馈神经网络的算法。梯度计算
# 使用反向传播。

import random
import numpy as np
from joblib.numpy_pickle_utils import xrange

class Network(object):
  def __init__(self,sizes):
    # sizes列表表示各层神经元的数量

    # 例如，如果sizes[2, 3, 1]，那么它将是一个三层网络，第一层有2个神经元，

    # 第二层有3个神经元，第三层1个神经元。

    # 偏差和权重被随机初始化。

    # 使用一个均值为0，方差为1的高斯分布。

    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y,1) for y in sizes [1:]]
    self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

  def feedforward(self,a):
    # 前馈神经网络
    # 如果输入是a
    # 返回……
    for b,w in zip(self.biases,self.weights):
      a = sigmoid(np.dot(w,a)+b)
    return a

  def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
    # 随即梯度下降算法

    # 使用小批量数据进行神经网络训练

    # training_data是一个元组列表(x,y)表示输入和期望输出

    # test_data是网络进行评估的测试数据，在每次迭代过程中会将部分过程打印出来。

    if test_data: n_test = len(test_data)
    # 原版是：if test_data: n_test = len(test_data)
    # 由于版本变更问题，现在使用如上代码
    n = len(training_data)
    for j in xrange (epochs):
      random.shuffle(training_data)
      mini_batches = [training_data [k:k+mini_batch_size]
                      for k in xrange(0,n,mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      if test_data:
        print("Epochs {0} : {1}/ {2}".format(j,self.evaluate(test_data),n_test))
      else:
        print("{Epoch {0} complete}".format(j))

  def update_mini_batch(self,mini_batch,eta):
    # 应用随即梯度下降，使用反向传播更新网络的权重和偏置。

    # mini_batch是一个(X,Y)的元组，eta是学习率

    # 在每个迭代期，它首先随机地将训练数据打乱，然后将它们分成多个适当大小的小批量数据。

    # 对于每一个mini_batch应用一次梯度下降

    # 仅仅使用mini_batch中的训练数据，根据单次梯度下降的迭代更新网络的权重和偏置。

    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x,y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      # 大部分工作都由这行代码工作⬆️
      # 调用了一个反向传播算法，快速计算代价函数的梯度
      # 本次函数仅仅是对mini_batch中的每一个训练养嫩计算梯度，适当更新权重和偏置。
      nabla_b = [nb +dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w-(eta/len(mini_batch))*nw
                    for w,nw,in zip(self.weights,nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb
                    for b,nb in zip(self.biases,nabla_b)]

  def backprop(self,x,y):
    # 返回元组(nabla_b, nabla_w)表示代价函数C_x的梯度。
    # nabla_b,nabla_w 是 numpy数组的逐层列表，类似self.biases和self.weights
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    activation = x
    activations = [x]
    # 存储所有激活状态的列表
    zs = []
    # 存储每一层的z向量
    for b,w in zip(self.biases,self.weights):
      z = np.dot(w,activation)+b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    delta = self.cost_derivative(activations[-1],y) * \
            sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta,activations[-2].transpose())
    # 注意，下面循环中的变量l使用了一点
    # 与本书第二章的符号不同。在这里,
    # l = 1表示最后一层神经元，l = 2表示
    # 倒数第二层，以此类推。这是对
    # scheme在书中，这里用来利用事实
    # Python可以在列表中使用负索引。
    for l in xrange(2,self.num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
    return(nabla_b,nabla_w)

  def evaluate(self,test_data):
    # 返回神经的测试输入的数量
    # 网络输出正确的结果。注意神经系统
    # 假设Network的输出是任意一个的索引
    # 最后一层神经元的激活程度最高。
    test_results = [(np.argmax(self.feedforward(x)),y)
                    for (x,y) in test_data]
    return sum(int(x==y) for (x,y) in test_results )

  def cost_derivative(self,output_activations,y):
    return (output_activations - y)

def sigmoid(z):
    # sigmoid函数
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # 计算sigma的倒数
    return sigmoid(z)*(1-sigmoid(z))
