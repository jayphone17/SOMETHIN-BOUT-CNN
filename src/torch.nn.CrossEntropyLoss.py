import torch
from torch.autograd import Variable
loss_f = torch.nn.CrossEntropyLoss()
x = Variable(torch.randn(3,5))

y = Variable(torch.LongTensor(3).random_(5))

loss = loss_f(x,y)
print(loss.data)