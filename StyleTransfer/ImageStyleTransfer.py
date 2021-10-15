# 内容度量值：衡量图片之间的内容差异程度
#
# 风格度量值：衡量图片之间的风格差异程度

# 建立神经网络模型，对内容图片中的内容和风格图片的风格进行提取
#
# 以内容图片为基准将其输入建立的模型中，并不断调整内容度量和风格度量值
#
# 让他们趋近于最小！！！
import matplotlib_inline
import torch
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import copy

matplotlib_inline

transform = transforms.Compose([transforms.Resize([224,224]),
                                        transforms.ToTensor()])

def loading(path = None):
    img = Image.open(path)
    img = transform(img)
    img = img.unsqueeze(0)
    return img

content_img = loading("1111.jpg")
content_img = Variable(content_img).cuda()
style_img = loading("2222.jpg")
style_img = Variable(style_img).cuda()

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


class Gram_matrix(torch.nn.Module):
    # 实例参与风格损失的计算。
    def forward(self, input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        return gram.div(a * b * c * d)


class Style_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Style_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()

        self.gram = Gram_matrix()

    def forward(self, input):
        self.Gram = self.gram(input.clone())
        self.Gram.mul_(self.weight)

        self.loss = self.loss_fn(self.Gram, self.target)
        return input

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss

Use_gpu = torch.cuda.is_available()
cnn = models.vgg16(pretrained = True).features

if Use_gpu:
    print("Using GPU for training!!!!! ")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    cnn = cnn.cuda()

model = copy.deepcopy(cnn)

content_layer = ["Conv3"]
style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4"]

content_losses = []
style_losses = []

content_weight = 1
style_weight = 1000

new_model = torch.nn.Sequential()
model = copy.deepcopy(cnn)
gram = Gram_matrix()

if Use_gpu:
    print("Using GPU for training!!!!! ")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    new_model = new_model.cuda()
    gram = gram.cuda()

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

epoch_n = 300
epoch = [0]

while epoch[0] <= epoch_n:
    def closure():
        optimizer.zero_grad()
        style_score = 0
        content_score = 0
        parameter.data.clamp_(0,1)
        new_model(parameter)
        for sl in style_losses:
            style_score += sl.backward()
        for cl in content_losses:
            content_score += cl.backward()
        epoch[0] += 1
        if epoch[0] % 50 == 0:
            print("Epoch: {}".format(epoch[0]))
            print("Style_Loss: {:.4f} Content_Loss: {:.4f}".format(style_score.item(), content_score.item()))
        return style_score + content_score
    optimizer.step(closure)


#图片的输出
# output = parameter.data
# unloader = transforms.ToPILImage()  # reconvert into PIL image
#
# plt.ion()
# plt.figure()
# def imshow(tensor, title=None):
#     image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
#     image = image.view(3, 224, 224)  # remove the fake batch dimension
#     image = unloader(image)
#     plt.imshow(image)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001) # pause a bit so that plots are updated
# imshow(output, title='Output Image')
# plt.ioff()
# plt.show()

