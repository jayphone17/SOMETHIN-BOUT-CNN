import matplotlib.pyplot as plt
import torch
import torchvision.utils
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
from torchvision import *
import MNIST_Recog_Mark2

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

data_test = datasets.MNIST(root = "../data/", transform = transform, train=False)
data_loader_test = torch.utils.data.DataLoader(dataset = data_test, batch_size = 6, shuffle = True)

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
#为了检测效果，随机抽取一部分图片，并验证是否准确。
#对其结果进行可视化

model = Model()
if torch.cuda.is_available():
    model.cuda()
#定义损失函数
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print(model)

#模型训练步骤
epoch_n = 10

X_test,y_test = next(iter(data_loader_test))
X_test, y_test = X_test.cuda(), y_test.cuda()
inputs = Variable(X_test)
pred = model(inputs)
_,pred = torch.max(pred,1)

print("Predict Label is:",(i for i in pred))
print("Real Label is :",[i for i in y_test])

img = torchvision.utils.make_grid(X_test)
img = img.cpu().numpy().transpose(1,2,0)

std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean
plt.imshow(img)
plt.show()