# 包含torchvision、torch.transforms、dataloader、dropout
import matplotlib.pyplot as plt
import torch
import torchvision.utils
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
from torchvision import *


#标准化变换方法：标准差变换法。
# 需要原始数据的均值（Mean）和标准差（Standard Division）来进行数据的标准化
# 标准化之后符合数据全部均值为0，标准差为1的正态分布
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

#模型构建步骤
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #卷积层
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size = 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
        )
        #全连接层
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(1024,10)
        )

    def forward(self,x):
        x = self.conv1(x) #进行卷积处理
        x = x.view(-1,14*14*128) #实现扁平化，不确定是不是就是flatten
        x = self.dense(x) #定义全连接进行最后分类
        return x



#模型参数优化步骤
model = Model()
if torch.cuda.is_available():
    model.cuda()
#定义损失函数
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print(model)

#模型训练步骤
epoch_n = 10
for epoch in range(epoch_n):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, epoch_n))
    print("-"*10)
    #训练集
    for data in data_loader_train:
        X_train, y_train = data
        X_train, y_train = X_train.cuda(), y_train.cuda()
        X_train, y_train = Variable(X_train),Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data,1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        loss.backward()
        optimizer.step()
        # running_loss += loss.data[0]
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)

    #测试集
    testing_correct = 0.0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = X_test.cuda(), y_test.cuda()
        X_test, y_test = Variable(X_test),Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is :{:.4f}, Train_Accuracy is : {:.4f}%, Test_accuracy is : {:.4f}%".format(running_loss/len(data_train),
                                                                                            100*running_correct/len(data_train),
                                                                                            100*testing_correct/len(data_test)))
