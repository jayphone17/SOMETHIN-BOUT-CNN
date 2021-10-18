import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

print("downloading datasets")
data_train = datasets.MNIST(root = "../data/", transform = transform, train=True, download=True)
data_test = datasets.MNIST(root = "../data/", transform = transform, train=False)
print("finished downloading")

data_loader_train = torch.utils.data.DataLoader(dataset = data_train,
                                                batch_size = 64,
                                                shuffle = True)
data_loader_test = torch.utils.data.DataLoader(dataset = data_test,
                                               batch_size = 64,
                                               shuffle = True)

images,labels = next(iter(data_loader_train))
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean
print([labels[i] for i in range(64)])
plt.imshow(img)
plt.show()

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
    self.output = torch.nn.Linear(128, 10)
    # self.output = torch.nn.Linear(192,10)
    # output是10，有10个分类
    # 隐藏层的输出代表input，128

  def forward(self, input):

    print(input.shape)
    # torch.Size([192, 28, 28])

    output,_ = self.rnn(input,None)
    # H^0一般采用0初始化，所以H^0这里设置到None
    output = self.output(output[:,-1,:])
    # 提取最后一个序列的输出结果作为当前循环神经网络模型的输出。
    return output

model = RNN()

# 接下来进行训练，进行10个epoch
optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

epoch_n = 10
for epoch in range(epoch_n):
  running_loss = 0.0
  running_correct = 0.0
  testing_correct = 0.0
  print("Epoch {}/{}".format(epoch,epoch_n))
  print("-"*50)

  # 训练集
  for data in data_loader_train:
    X_train, y_train = data

    print(X_train.shape)
    print(y_train.shape)

    X_train = X_train.view(-1,28,28)

    print(X_train.shape)

    # 默认设置H^0为0，28x28的图片大小
    X_train, y_train = Variable(X_train), Variable(y_train)
    y_pred = model(X_train)
    loss = loss_func(y_pred, y_train)
    _, pred = torch.max(y_pred.data, 1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    running_correct += torch.sum(pred == y_train.data)

  # 测试集
  for data in data_loader_test:
    X_test, y_test = data
    X_test = X_test.view(-1, 28, 28)
    X_test, y_test = Variable(X_test), Variable(y_test)
    outputs = model(X_test)
    _, pred = torch.max(outputs.data, 1)
    testing_correct += torch.sum(pred == y_test.data)

print("Loss is: {:.4f}, Training Accuracy: {:.4f}%, Test Accuracy: {:.4f}"
      .format(running_loss/len(data_train), 100*running_correct/len(data_train),
              100*testing_correct/len(data_test)))
