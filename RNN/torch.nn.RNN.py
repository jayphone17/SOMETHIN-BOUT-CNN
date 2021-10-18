# 循环神经网络主要被用于序列数据问题
#
# 但是在CV领域也可以用于图片分类问题
#
# 对于一个搭建好的卷积神经网络，输入以及输出的数据维度是固定的，且层次也是结构也是固定不管的
#
# 但是循环神经网络中的循环单元可以随意控制输入数据以及输出数据的数量
#
# 有非常强大的<<<<<灵活性>>>>>！！


# 虽然RNN能有效处理序列数据，但是有一个很大的弊端就是：不能存储长期记忆

# 如果近期输入的数据产生了变化，则会对当前的输出产生较大的影响

# 因此，LSTM出现了，长短期记忆类型的循环神经网络
import torch


class RNN(torch.nn.Module):
  def __init__(self):
    super(RNN, self).__init__()
    self.rnn = torch.nn.RNN(
      input_size = 28,
      # 用于指定输入数据特征数，因为输入图片都是28 x 28的
      hidden_size = 128,
      # 隐藏层输出特征数
      num_layers = 1,
      # 循环层堆叠的数量，默认是1
      batch_first = True
      # 默认是（seq,batch,feature）
      # 如果设置为True，会设置成：
      # （batch,seq,featrue）
    )

    self.output = torch.nn.Linear(128,10)
    # output是10，有10个分类
    # 隐藏层的输出代表input，128

  def forward(self, input):
    print(input.shape)
    output,_ = self.rnn(input,None)
    # H^0一般采用0初始化，所以H^0这里设置到None
    output = self.output(output[:,-1,:])
    # 提取最后一个序列的输出结果作为当前循环神经网络模型的输出。
    return output

model = RNN()
print(model)
