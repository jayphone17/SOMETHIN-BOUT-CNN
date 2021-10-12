import torch
from torch.autograd import Variable

loss_f = torch.nn.L1Loss()

x = Variable(torch.randn(100,100))
y = Variable(torch.randn(100,100))

loss = loss_f(x,y)

print(loss.data)