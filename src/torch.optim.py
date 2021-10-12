import torch
from torch.autograd import Variable
from collections import OrderedDict

batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10
x = Variable(torch.randn(batch_n, input_data),requires_grad = False)
y = Variable(torch.randn(batch_n, output_data),requires_grad = False)

models = torch.nn.Sequential(OrderedDict([
    ("linear1", torch.nn.Linear(input_data, hidden_layer)),
    ("relu1", torch.nn.ReLU()),
    ("linear2", torch.nn.Linear(hidden_layer, output_data))
]))

epoch_n = 350
learning_rate = 1e-4
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(models.parameters(), lr = learning_rate)

for epoch in range(epoch_n):
    y_pred = models(x)
    loss = loss_fn(y_pred, y)
    print("Epoch:{}, Loss:{:.4f}".format(epoch, loss.data.item()))
    optimizer.zero_grad() #参数梯度归零。

    loss.backward()
    optimizer.step() #对每个节点的参数进行参数更新。