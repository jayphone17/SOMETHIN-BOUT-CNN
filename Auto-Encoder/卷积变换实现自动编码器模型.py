# 使用卷积变换的方式实现自动编码器会比较复杂且有很大的区别

# 卷积变换的方式主要使用卷积层，池化层，上采样层，和激活函数作为神经网络结构的主要组成部分

import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                # 而是因为图片格式是灰度图只有一个channel，需要变成RGB图才可以
                                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                transforms.Normalize([0.5],[0.5])
])

print("downloading datasets")
data_train = datasets.MNIST(root="../data/", transform=transform, train=True, download=True)
data_test = datasets.MNIST(root="../data/", transform=transform, train=False)
print("finished downloading")

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)

images, labels = next(iter(data_loader_train))
print(images.shape)
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean
print([labels[i] for i in range(4)])

# 构建马赛克图片
noisy_images = img + 0.5*np.random.randn(*img.shape)
noisy_images = np.clip(noisy_images, 0., 1.)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self,input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output

model = AutoEncoder()
Use_gpu = torch.cuda.is_available()
if Use_gpu:
    print("Using GPU for training!!!!! ")
    print("显卡数量："+ str(torch.cuda.device_count()))
    print("显卡型号："+torch.cuda.get_device_name(0))
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.MSELoss()
epoch_n = 5
for epoch in range(epoch_n):
    running_loss = 0.0

    print("Epoch: {}/{}".format(epoch, epoch_n))
    print("-"*50)

    for data in data_loader_train:
        X_train, _ = data
        noisy_X_train = X_train + 0.5*torch.randn(X_train.shape)
        noisy_X_train = torch.clamp(noisy_X_train, 0., 1. )

        X_train, noisy_X_train = Variable(X_train.cuda()), Variable(noisy_X_train.cuda())

        image_pre = model(noisy_X_train)
        loss = loss_func(image_pre, X_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print("Loss is : {:.4f}".format(running_loss/len(data_train)))

print("Validation..........")
X_test, _ = next(iter(data_loader_test))
img1 = torchvision.utils.make_grid(X_test)
img1 = img1.numpy().transpose(1,2,0)
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img1 = img1*std+mean
plt.imshow(img1)
plt.show()

noisy_X_test = img1 + 0.5*np.random.randn(*img1.shape)
noisy_X_test = np.clip(noisy_X_test, 0., 1.)
plt.imshow(noisy_X_test)
plt.show()

img2 = X_test + 0.5*torch.randn(X_test.shape)
img2 = torch.clamp(img2, 0., 1.)
img2 = Variable(img2.cuda())
test_pred = model(img2)
img_test = test_pred.data.view(-1,1,28,28)
img2 = torchvision.utils.make_grid(img_test)
img2 = img2.cpu().numpy().transpose(1,2,0)
img2 = img2*std+mean
img2 = np.clip(img2, 0., 1.)
plt.imshow(img2)
plt.show()