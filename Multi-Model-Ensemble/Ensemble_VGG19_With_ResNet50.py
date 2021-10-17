# 使用单模型处理某个问题的时候容易遇到泛化瓶颈由于一些客观因素
# 某个模型可能在处理某种情况能力会比较出色
# 但是在处理别的问题会遇到限制
# 所以为了突破泛化瓶颈
# 融合多个模型的优点
#
# 多模型融合主要有三种类型：
# 1. 结果多数表决
# 2. 结果加权平均
# 3. 结果直接平均
#
# *******各个模型的差异性越高，多模型融合预测结果就会越好*******

import matplotlib_inline
import torch
import torchvision
from torchvision import datasets, transforms, models
import os
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

matplotlib_inline

# 1. 数据读取部分
data_dir = "..\data\DogsVSCats"
data_transform = {x:transforms.Compose([transforms.Resize([224,224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
                  for x in ["train","valid"]}
image_datasets = {x:datasets.ImageFolder(root=os.path.join(data_dir,x),
                                         transform = data_transform[x])
                  for x in ["train", "valid"]}
dataloader = {x:torch.utils.data.DataLoader(dataset = image_datasets[x],
                                            batch_size = 16,
                                            shuffle = True)
                                            for x in ["train", "valid"]}

X_example, y_example = next(iter(dataloader["train"]))
example_classes = image_datasets["train"].classes
index_classes = image_datasets["train"].class_to_idx

# 2. 模型构建以及分数优化部分
model1 = models.vgg19(pretrained=True)
model2 = models.resnet50(pretrained=True)

Use_gpu = torch.cuda.is_available()

for param in model1.parameters():
    param.requires_grad = False

model1.classifier = torch.nn.Sequential(torch.nn.Linear(25088,4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p = 0.5),
                                        torch.nn.Linear(4096, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p = 0.5),
                                        torch.nn.Linear(4096, 2)
)

for param in model2.parameters():
    param.requires_grad = False

model2.fc = torch.nn.Linear(2048,2)

if Use_gpu:
    print("Using GPU for training!!!!! ")
    print("显卡数量："+ str(torch.cuda.device_count()))
    print("显卡型号："+torch.cuda.get_device_name(0))
    mode1 = model1.cuda()
    model2 = model2.cuda()

loss_func_1 = torch.nn.CrossEntropyLoss()
loss_func_2 = torch.nn.CrossEntropyLoss()

optimizer1 = torch.optim.Adam(model1.classifier.parameters(), lr = 0.00001)
optimizer2 = torch.optim.Adam(model2.fc.parameters(), lr = 0.00001)

weight1 = 0.6
weight2 = 0.4

epoch_n = 5
time_open = time.time()

# 3. 训练部分

for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch, epoch_n-1))
    print("-"*50)

    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training...")
            model1.train(True)
            model2.train(True)
        else:
            print("Validating...")
            model1.train(False)
            model2.train(False)

        # 初始化loss和accuracy
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_corrects1 = 0.0
        running_corrects2 = 0.0
        blending_running_corrects = 0.0

        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data
            # if Use_gpu:
            X, y = Variable(X.cuda()), Variable(y.cuda())
            # else:
            #     X, y = Variable(X), Variable(y)

            y_pred1 = model1(X)
            y_pred2 = model2(X)

            blending_y_pred = y_pred1 * weight1 + y_pred2 * weight2

            _, pred1 = torch.max(y_pred1.data, 1)
            _, pred2 = torch.max(y_pred2.data, 1)
            _, blending_pred = torch.max(blending_y_pred.data, 1)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss1 = loss_func_1(y_pred1, y)
            loss2 = loss_func_2(y_pred2, y)

            if phase == "train":
                loss1.backward()
                loss2.backward()

                optimizer1.step()
                optimizer2.step()

            running_loss1 += loss1.item()
            running_corrects1 += torch.sum(pred1 == y.data)
            running_loss2 += loss2.item()
            running_corrects2 += torch.sum(pred2 == y.data)
            blending_running_corrects += torch.sum(blending_pred == y.data)


            if batch % 500 == 0 and phase == "train":
                # print("Batch {}, Model_1 TrainLoss: {:.4f}, Model_1 TrainAcc: {:.4f}, \
                # Model_2 TrainLoss: {:.4f}, Model_2 TrainAcc: {:.4f}, Blending_Model_Acc:{.4f}".format(batch,
                #       running_loss1/batch,100*running_corrects1/(16 * batch),
                #       running_loss2/batch,100*running_corrects2/(16 * batch),
                #       100*blending_running_corrects/(16 * batch)))

                # 以上代码报错：'tensor' object has no attribute "4f"
                # 以下代码正确。

                print(
                    'Batch {},Model1 Train Loss:{:.4f},Model1 Train ACC:{:.4f},Model2 Train Loss:{:.4f},Model2 Train ACC:{:.4f},Blending_Model ACC:{:.4f}'
                    .format(batch, running_loss1 / batch, 100 * running_corrects1 / (16 * batch),
                            running_loss2 / batch, 100 * running_corrects2 / (16 * batch),
                            100 * blending_running_corrects / (16 * batch)))

        epoch_loss1 = running_loss1 * 16 / len(image_datasets[phase])
        epoch_acc1 = 100 * running_corrects1 / len(image_datasets[phase])

        epoch_loss2 = running_loss2 * 16 / len(image_datasets[phase])
        epoch_acc2 = 100 * running_corrects2 / len(image_datasets[phase])

        epoch_blending_acc = 100*blending_running_corrects/len(image_datasets[phase])

        # print('Epoch: , Model_1_Loss: {:.4f}, Model_1_Acc: {:.4f}%,\
        #  Model_2_Loss: {:.4f}, Model_2_Acc: {:.4f}%, \
        #  Blending_Model_Acc'.format(phase, epoch_loss1, epoch_acc1, epoch_loss2, epoch_acc2, epoch_blending_acc))

        print(
            'Epoch, Model1 Loss: {:.4f},Model1 ACC: {:.4f}%,Model2 Loss: {:.4f},Model2 ACC: {:.4f}%,Blending_Model ACC: {:.4f}'
            .format(epoch_loss1, epoch_acc1, epoch_loss2, epoch_acc2, epoch_blending_acc))

time_end = time.time() - time_open
print(time_end)
