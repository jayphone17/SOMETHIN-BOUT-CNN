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
GPU = True
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
