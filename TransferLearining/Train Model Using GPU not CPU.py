import torch.nn
from torch.autograd import Variable
from HomeMadeSimplifiedVGGModel import Models
import torch
import torchvision
from torchvision import datasets, transforms
import os
import time
import matplotlib.pyplot as plt


data_dir = "..\data\DogsVSCats"

data_transform = {x:transforms.Compose([transforms.Resize([64,64]),transforms.ToTensor()])
                  for x in ["train","valid"]}

image_datasets = {x:datasets.ImageFolder(root=os.path.join(data_dir,x),transform = data_transform[x])
                  for x in ["train", "valid"]}

dataloader = {x:torch.utils.data.DataLoader(dataset = image_datasets[x],
                                            batch_size = 16,
                                            shuffle = True)
                                            for x in ["train", "valid"]}

x_example, y_example = next(iter(dataloader["train"]))

#热编码，0代表猫，1代表狗
index_classes = image_datasets["train"].class_to_idx

#做事留一手
#为了到时候可以还原原本标签进而具有可识别性
#将原来的标签存储起来：
example_classes = image_datasets["train"].classes

class Models(torch.nn.Module):
    def __init__(self):
        super(Models, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(4*4*512, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(1024,2)
        )
    def forward(self,input):
        x = self.Conv(input)
        x = x.view(-1,4*4*512)
        x = self.Classes(x)
        return x


Use_gpu = torch.cuda.is_available()
model = Models()
if Use_gpu:
    print("Using GPU for training!!!!! ")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    model = model.cuda()

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
epoch_n = 10
time_open = time.time()

for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch, epoch_n-1))
    print("-"*50)

    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training...")
            model.train(True)
        else:
            print("Validating...")
            model.train(False)

        # 初始化loss和accuracy
        running_loss = 0.0
        running_corrects = 0

        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data
            # if Use_gpu:
            X, y = Variable(X.cuda()), Variable(y.cuda())
            # else:
            #     X, y = Variable(X), Variable(y)
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)
            optimizer.zero_grad()
            loss = loss_func(y_pred, y)
            if phase == "train":
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(pred == y.data)

            if batch % 500 == 0 and phase == "train":
                print('Batch{},TrainLoss:{:.4f},TrainAcc:{:.4f}'.format(batch, running_loss / batch,
                                                                        100 * running_corrects / (16 * batch)))
        epoch_loss = running_loss * 16 / len(image_datasets[phase])
        epoch_acc = 100 * running_corrects / len(image_datasets[phase])
        print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))

time_end = time.time() - time_open
print(time_end)
