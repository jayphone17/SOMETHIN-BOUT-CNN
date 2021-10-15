import torch
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import copy

class Content_loss(torch.nn.Module):
    def __init__(self, weight, target):
        #weight是权重参数，用来控制内容和风格对最后合成图像的影响程度
        #target是通过卷积获取到的输入图像中的内容
        super(Content_loss).__init__()
        self.weight = weight
        self.target = target.detach()*weight
        #用来对提取到的内容进行锁定，不需要进行梯度下降？
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input):
        #用来计算输入图像和内容图像之间的损失值
        self.loss = self.loss_fn(input*self.weight, self.target)
        return input

    def backward(self):
        #根据计算得到的损失值进行反向传播，并返回损失值。
        self.loss.backward(retain_graph = True)
        return self.loss