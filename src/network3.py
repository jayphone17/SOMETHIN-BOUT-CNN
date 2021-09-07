# 一个基于theano的程序，用于训练和运行简单的神经网络。
#
# 支持几种层类型(完全连接，卷积，最大和激活函数(sigmoid, tanh，和整流线性单位，更容易添加)。
#
# 当在CPU上运行时，这个程序比network.py和network2.py。
# 但是，与network.py和network.py不同的是，它也可以在GPU上运行，这让它更快。
#
# 因为代码是基于Theano的，所以代码在很多方面都是不同的
# network.py和network2.py的方法。然而，只要可能，我就会这么做试图与早期的程序保持一致。
# 特别是，API类似于network2.py。注意我有专注于使代码简单、易读和容易修改的。
# 它没有进行优化，并且省略了许多理想的特性。

import pickle
import gzip
import numpy as np
import theano
import theano.tensor as T
from joblib.numpy_pickle_utils import xrange
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

def linear(z): return z
def ReLU(z): return T.maximum(0.0,z)
# 引入线性修正单元

from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

# GPU模块
GPU = False
if GPU:
  print("开始尝试使用GPU运行程序。如果不使用GPU则把上面GPU改成False。")
  try:theano.config.device = 'gpu'
  except: pass
  theano.config.floatX = 'float32'
else:
  print("使用CPU运行。如果有需要则修改为GPU运行。")

def load_data_shared(filename = "../data/mnist.pkl.gz"):
  f = gzip.open(filename,'rb')
  training_data, validation_data, test_data = pickle.load(f)
  f.close()
  def shared(data):
    # 将数据放入共享变量中。这使得Theano可以复制
    # 如果GPU可用，则将数据传输到GPU。
    shared_x = theano.shared(
      np.asarray(data[0],dtype=theano.config.floatX),borrow = True)
    shared_y = theano.shared(
      np.asarray(data[0],dtype=theano.config.floatX),borrow = True)
    return shared_x,T.cast(shared_y,"int32")
  return [shared(training_data),shared(validation_data),shared(test_data)]

# 主网络架构
class Network(object):
  def __init__(self,layers,mini_batch_size):
    self.layers = layers
    self.mini_batch_size = mini_batch_size
    self.params = [params for layer in self.layers for params in layer.params]
    self.x = T.matrix("x")
    self.y = T.matrix("y")
    init_layer = self.layers[0]
    init_layer.set_inpt(self.x,self.x,self.mini_batch_size)
    for j in xrange(1,len(self.layers)):
      prev_layer,layer = self.layers[j-1],self.layers[j]
      layer.set_inpt(
        prev_layer.output,prev_layer.ouput_dropout,self.mini_batch_size
      )
    self.output = self.layers[-1].output
    self.output_dropout = self.layers[-1].output_dropout

  def SGD(self,training_data, epochs, mini_batch_size, eta, validation_data,test_data, lmbda = 0.0):
    # 使用小批量数据以及随机梯度下降训练网络
    training_x,training_y = training_data
    validation_x,validation_y = validation_data
    test_x, test_y = test_data
    # 计算小批量用于训练、验证以及测试
    num_training_batches = size(training_data)/mini_batch_size
    num_validation_batches = size(validation_data)/mini_batch_size
    num_test_batches = size(test_data)/mini_batch_size

    # 定义正则化的代价函数，以及梯度，更新
    # L2正则化
    l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
    cost = self.layers[-1].cost(self)+\
           0.5*lmbda*l2_norm_squared/num_training_batches
    grads = T.grad(cost,self.params)
    updates = [(param, param-eta*grad)
               for param, grad in zip(self.params, grads)]

    # 定义一个函数训练小批量，通过验证数据以及测试小批量计算准确率
    # minibatch的索引
    i = T.lscalar()
    train_mb = theano.function(
      [i],cost,updates = updates,
      givens={
        self.x:
        training_x[i*self.mini_batch_size:(i+1)*self.mini_batch_size],
        self.y:
        training_x[i*self.mini_batch_size:(i+1)*self.mini_batch_size]
      }
    )
    validate_mb_accuracy = theano.function(
      [i], self.layers[-1].accuracy(self.y),
      givens={
        self.x:
          validation_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
        self.y:
          validation_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
      })
    test_mb_accuracy = theano.function(
      [i], self.layers[-1].accuracy(self.y),
      givens={
        self.x:
          test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
        self.y:
          test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
      })
    self.test_mb_predictions = theano.function(
      [i], self.layers[-1].y_out,
      givens={
        self.x:
          test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
      })

    # 进行真正的训练
    base_validation_accuracy = 0.0
    for epoch in xrange(epochs):
      for minibatch_index in xrange(num_training_batches):
        iteration  = num_validation_batches*epoch+minibatch_index
        if iteration % 1000 ==0:
          print("training mini-batch number {0}".format(iteration))
        cost_ij = train_mb(minibatch_index)
        if(iteration+1) % num_training_batches == 0:
          validation_accuracy = np.mean(
            [validate_mb_accuracy(j) for j in xrange(num_validation_batches)]
          )
          print("epoch {0} :validation accuracy {1:2%}".format(epoch,validation_accuracy))
          if validation_accuracy >= best_validation_accuracy:
            print("this is the best validation accuracy to data")
            best_validation_accuracy = validation_accuracy
            best_iteration = iteration
            if test_data:
              test_accuracy = np.mean(
                [test_mb_accuracy[j] for j in xrange(num_test_batches)]
              )
              print("the corresponding test accuracy is {0:,2%}".format(test_accuracy))
    print("finished training network")
    print("best validation accuracy of {0:.2%} obtained at iteration {1}".format(best_validation_accuracy,best_iteration))
    print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

# 卷基层以及池化层
class ConvPoolLayer(object):
  def __init__(self, filtter_shape, image_shape, poolsize = (2,2),activation_fn = sigmoid):
    # filtter_shape是一个长度为4的元组
    # 表示1.数字的个数、2.输入特征映射个数、3.滤波器的高度、4.滤光片的宽度。
    # image_shape是一个长度为4的元组
    # 表示1.小戳的大小、2.输入特征图数量、3.图像高度、4.图像宽度
    # poolsize是一个长度为2的元组
    # 表示x和y的卷积核大小。
    self.filtter_shape = filtter_shape
    self.image_shape = image_shape
    self.poolsize = poolsize
    self.activation_fn = activation_fn

    # 初始化权重和偏置
    n_out = (filtter_shape[0]*np.prod(filtter_shape[2:])/np.prod(poolsize))
    self.w = theano.shared(
      np.asarray(
        np.random.normal(loc = 0, scale=np.sqrt(1.0 / n_out), size = filtter_shape),
        dtype= theano.config.floatX,
      ),
      borrow=True
    )
    self.b = theano.shared(
      np.adarray(
        np.random.normal(loc = 0, scale=1.0, size = (filtter_shape[0],)),
        dtype = theano.config.floatX,
      ),
    borrow = True
    )
    self.params = [self.w,self.b]

  def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
    self.inpt = inpt.reshape(self.image_shape)
    conv_out = conv.conv2d(
      input = self.inpt, filters=self.w,filter_shape=self.filtter_shape,
      image_shape=self.image_shape
    )
    pooled_out = downsample.max_pool_2d(
      input = conv_out, ds = self.poolsize, ignore_border=True
    )
    self.output = self.activation_fn(
      pooled_out + self.b.dimshuffle('x',0,'x','x')
    )
    self.output_dropout = self.output

# 全连接层
class FullyConnectedLayer(object):
  def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
    self.n_in = n_in
    self.n_out = n_out
    self.activation_fn = activation_fn
    self.p_dropout = p_dropout
    # Initialize weights and biases
    self.w = theano.shared(
      np.asarray(
        np.random.normal(
          loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
        dtype=theano.config.floatX),
      name='w', borrow=True)
    self.b = theano.shared(
      np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                 dtype=theano.config.floatX),
      name='b', borrow=True)
    self.params = [self.w, self.b]

  def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
    self.inpt = inpt.reshape((mini_batch_size, self.n_in))
    self.output = self.activation_fn(
      (1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
    self.y_out = T.argmax(self.output, axis=1)
    self.inpt_dropout = dropout_layer(
      inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
    self.output_dropout = self.activation_fn(
      T.dot(self.inpt_dropout, self.w) + self.b)

  def accuracy(self, y):
    "Return the accuracy for the mini-batch."
    return T.mean(T.eq(y, self.y_out))

# 柔性最大层
class SoftmaxLayer(object):
  def __init__(self, n_in, n_out, p_dropout=0.0):
    self.n_in = n_in
    self.n_out = n_out
    self.p_dropout = p_dropout
    # Initialize weights and biases
    self.w = theano.shared(
      np.zeros((n_in, n_out), dtype=theano.config.floatX),
      name='w', borrow=True)
    self.b = theano.shared(
      np.zeros((n_out,), dtype=theano.config.floatX),
      name='b', borrow=True)
    self.params = [self.w, self.b]

  def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
    self.inpt = inpt.reshape((mini_batch_size, self.n_in))
    self.output = softmax((1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
    self.y_out = T.argmax(self.output, axis=1)
    self.inpt_dropout = dropout_layer(
      inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
    self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

  def cost(self, net):
    "Return the log-likelihood cost."
    return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

  def accuracy(self, y):
    "Return the accuracy for the mini-batch."
    return T.mean(T.eq(y, self.y_out))

def size(data):
  return data[0].get_value(borrow = True).shape[0]

def dropout_layer(layer, p_dropout):
  srng = shared_randomstreams.RandomStreams(
    np.random.RandomState(0).randint(999999)
  )
  mask = srng.binomial(n=1,p=1-p_dropout, size=layer.shape)
  return layer*T.cast(mask,theano.config.floatX)
