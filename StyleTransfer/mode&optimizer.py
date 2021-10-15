import torch
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import copy

#提取卷积层中特征提取中的相关部分
cnn = models.vgg16(pretrained = True).features
#指定我们需要在哪一层提取内容以及风格
content_layer = ["Conv3"]
style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4"]
#用于保存内容损失以及风格损失的列表
content_losses = []
style_losses = []
#对最后得到的融合图片的影响权重
content_weight = 1
style_weight = 100

new_model = torch.nn.Sequential()

model = copy.deepcopy(cnn)

gram = gram_matrix()

Use_gpu = torch.cuda.is_available()
if Use_gpu:
    print("Using GPU for training!!!!! ")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    model = model.cuda()

index = 1

for layer in list(model)[:8]:
    if isinstance(layer, torch.nn.Conv2d):
        name = "Conv_"+str(index)
        new_model.add_module(name,layer)
        if name in content_layer:
            target = new_model(content_img).clone()
            content_loss = Content_loss(content_weight,target)
            new_model.add_module("content_loss"+str(index), content_loss)
            content_losses.append(content_loss)
        if name in style_layer:
            target = new_model(style_img).clone()
            target = gram(target)
            style_loss = Style_loss(style_weight, target)
            new_model.add_module("style_loss" + str(index), style_loss)
            style_losses.append(style_loss)
        if isinstance(layer, torch.nn.ReLU):
            name = "Relu_"+str(index)
            new_model.add_module(name,layer)
            index = index+1

        if isinstance(layer, torch.nn.MaxPool2d):
            name = "MaxPool_"+str(index)
            new_model.add_module(name,layer)

#优化参数部分代码
input_img = content_img.clone()
parameter = torch.nn.Parameter(input_img.data)
optimizer = torch.optim.LBFGS([parameter])
        