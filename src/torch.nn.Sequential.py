import torch
from torch.autograd import Variable
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10
x = Variable(torch.randn(batch_n, input_data),requires_grad = False)
y = Variable(torch.randn(batch_n, output_data),requires_grad = False)

# 使用Sequential自动生成和初始化对应维度的权重参数
# 是一种序列容器， 嵌套各种实现神经网络的中具体功能相关的类
# 参数会按照定义好的序列自动传递
# 模块加入的两种方式：
# 1.代码中直接嵌套。
# 2.以orderdict有序字典传入

# 直接嵌套
models = torch.nn.Sequential(
    torch.nn.Linear(input_data, hidden_layer),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer, output_data)
)
print(models)

# 以orderdict有序字典传入
from collections import OrderedDict

models = torch.nn.Sequential(OrderedDict([
    ("linear1", torch.nn.Linear(input_data, hidden_layer)),
    ("relu1", torch.nn.ReLU()),
    ("linear2", torch.nn.Linear(hidden_layer, output_data))
]))
print(models)

# Sequential(
#   (0): Linear(in_features=1000, out_features=100, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=100, out_features=10, bias=True)
# )
# Sequential(
#   (linear1): Linear(in_features=1000, out_features=100, bias=True)
#   (relu1): ReLU()
#   (linear2): Linear(in_features=100, out_features=10, bias=True)
# )

epoch_n = 10000
learning_rate = 1e-4
loss_fn = torch.nn.MSELoss()

# loss_fn = torch.nn.L1Loss()
# loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(epoch_n):
    y_pred = models(x)
    loss = loss_fn(y_pred, y)
    if epoch%1000==0:
        print("Epoch:{}, Loss:{:.4f}".format(epoch, loss.data.item()))
    models.zero_grad()

    loss.backward()

    for param in models.parameters():
        param.data -= param.grad.data*learning_rate
# 通过models.parameters()进行参数更新梯度