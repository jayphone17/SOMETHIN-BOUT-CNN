import json
import random
import sys
import numpy as np
from joblib.numpy_pickle_utils import xrange

# 这是一个network.py的改进版本。
# 实现了使用随机梯度下降学习一个前馈神经网络。
# 主要改进是使用了交叉上代价函数
# 以及使用正则化技术更好初始化网络的的权值、偏置。

class QuadraticCost(object):
# 网络输出a和目标输出y的二次代价函数的直接计算结果
  @staticmethod
  def fn(a,y):
    return 0.5 * np.linalg.norm(a-y)**2
  @staticmethod
  def delta(z,a,y):
    return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):
# 交叉熵损失函数
  @staticmethod
  def fn(a,y):
    # 返回与输出a和期望输出相关的成本y。
    # 注意, np.Nan_to_num用于确保数值稳定。
    # 特别是，如果a和y都有1.0在同一个槽中，
    # 则表达式(1 - y) * np.log(1 - a)返回nan。
    # np.Nan_to_num确保将其转换到正确的值(0.0)。
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

  @staticmethod
  def delta(z,a,y):
    return (a-y)

class Network(object):

  def __init__(self,sizes,cost = CrossEntropyCost):
    self.num_layer = len(sizes)
    self.sizes = sizes
    self.default_weight_initializer()
    self.cost = cost

  def default_weight_initializer(self):
    # 使用了改进后的初始化权重方法。
    # 使用了均值为0标准差为1/root(n)初始化权重
    # 使用均值为0，标准差为1初始化偏置
    self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
    self.weights = [np.random.randn(y,x)/np.sqrt(x)
                    for x,y in zip(self.sizes[:-1], self.sizes[1:])]

  def large_weight_initializer(self):
    self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
    self.weights = [np.random.randn(y,x)
                    for x,y in zip(self.sizes[:-1], self.sizes[1:])]

  def feedforward(self,a):
    for b,w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w,a)+b)
    return a

  def SGD(self,training_data, epochs, mini_batch_size, eta, lmbda = 0.0,
          evaluation_data = None,
          monitor_evaluation_cost = False,
          monitor_evaluation_accuracy = False,
          monitor_training_cost = False,
          monitor_training_accuracy = False):
    if evaluation_data : n_data = len(evaluation_data)
    n = len(training_data)
    evaluation_cost, evaluation_accuracy = [],[]
    training_cost, trainging_accuracy = [],[]
    for j in xrange(epochs):
      random.shuffle(training_data)
      mini_batches = [
        training_data[k:k+mini_batch_size]
        for k in xrange(0,n,mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch,eta,lmbda,len(training_data))
      print("Epoch %s training complete" % j)
      if monitor_training_cost:
        cost = self.total_cost(training_data,lmbda)
        training_cost.append(cost)
        print("Cost on training data : {}".format(cost))
      if monitor_training_accuracy:
        accuracy = self.accuracy(training_data, convert=True)
        trainging_accuracy.append(accuracy)
        print("Accuracy on training data: {}/{}".format(accuracy,n))
      if monitor_evaluation_cost:
        cost = self.total_cost(evaluation_data,lmbda,convert=True)
        evaluation_cost.append(cost)
        print("Cost on evaluation data : {}".format(cost))
      if monitor_evaluation_accuracy:
        accuracy = self.accuracy(evaluation_data)
        evaluation_accuracy.append(accuracy)
        print("Accuracy on evaluation data : {}/{}".format(self.accuracy(evaluation_data),n_data))
      print
    return evaluation_cost,evaluation_accuracy,\
        training_cost,trainging_accuracy

  def update_mini_batch(self,mini_batch, eta, lmbda, n):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x,y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x,y)
      nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
      nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
    # 接下来使用L2正则化优化。
    self.weights = [(1-eta *(lmbda/n))*w - (eta/len(mini_batch))*nw
                    for w,nw in zip(self.weights,nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb
                   for b,nb in zip(self.biases,nabla_b)]

  def backprop(self,x,y):
    # 反向传播
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # 前馈传播
    activation = x
    activations = [x]
    zs = []
    for b,w in zip(self.biases,self.weights):
      z = np.dot(w,activation)+b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    # backpass
    delta = (self.cost).delta(zs[-1], activations[-1],y)
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # 注意，下面循环中的变量l使用了一点
    # 与本书第二章的符号不同。在这里,
    # l = 1表示最后一层神经元，l = 2表示
    # 倒数第二层，以此类推。这是对
    # scheme在书中，这里用来利用事实
    # Python可以在列表中使用负索引。
    for l in xrange(2,self.num_layer):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
    return (nabla_b,nabla_w)

  def accuracy(self,data,convert = False):
    # 返回data中的输入的数量
    # 网络输出正确的结果。神经网络的
    # 输出假定为中任意一个神经元的指数
    # 最后一层的活性最高。
    # 如果数据集为False，则标志' ' convert ' '应该设置为False
    # 验证或测试数据(通常情况)，如果
    # 数据集是训练数据。对这面旗帜的需求出现了
    # 由于结果' ' y ' '的方式不同
    # 表示在不同的数据集。特别是,它
    # 标记是否需要在不同的
    # 表示。用different似乎有些奇怪
    # 不同数据集的表示。为什么不用
    # 这三个数据集的表示法相同吗?是做的
    # 效率原因——程序通常评估成本
    # 训练数据和其他数据集的准确性。
    # 这些是不同类型的计算，使用不同的
    # 表示可以加快速度。有关
    # 可以在
    # mnist_loader.load_data_wrapper。
    if convert:
      results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                 for (x,y) in data]
    else:
      results = [(np.argmax(self.feedforward(x)), y)
                 for (x,y) in data]
    return sum(int (x==y) for (x,y) in results)

  def total_cost(self, data, lmbda, convert = False):
    cost = 0.0
    for x,y in data:
      a = self.feedforward(x)
      if convert:y = vectorized_result(y)
      cost += self.cost.fn(a,y)/len(data)
    cost += 0.5 * (lmbda / len(data))*sum(
      np.linalg.norm(w)**2 for w in self.weights
    )
    return cost

  def save(self,filename):
    # 保存神经网络到文件夹
    # filename是文件名
    data = {"sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__)}
    f = open("/Users/jayphone/Desktop/SOMETHIN-BOUT-CNN/fig","w")
    json.dump(data,f)
    f.close()

def load(filename):
  f = open("/Users/jayphone/Desktop/SOMETHIN-BOUT-CNN/fig","r")
  data = json.load(f)
  f.close()
  cost = getattr(sys.modules[__name__],data["cost"])
  net = Network(data["sizes"],cost = cost)
  net.weights = [np.array(w) for w in data["weigths"]]
  net.biases = [np.array(b) for b in data["biases"]]
  return net
def vectorized_result(j):
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e
def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z):
  return sigmoid(z) * (1 - sigmoid(z))
