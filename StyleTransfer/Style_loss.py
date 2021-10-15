import torch

# 格拉姆矩阵（Gram Matrix）
# 我们通过卷积神经网络提取风格图片的风格
# 这些风格其实是有数字组成的
# 数字的大小代表了图片中的风格突出成都
# Gram矩阵是矩阵的内积运算
# 在运算过后输入到该矩阵的特征途中的大的数字会变得更大
# 这就相当于图片的风格被放大了
# 放大的风格在参与损失积算
# 便能对最后的合成图片产生更大的影响

class Gram_matrix(torch.nn.Module):
    #实例参与风格损失的计算。
    def forward(self,input):
        a,b,c,d = input.size()
        feature = input.view(a*b, c*d)
        gram = torch.mm(feature, feature.t())
        return gram.div(a*b*c*d)

class Style_loss(torch.nn.Module):
    def __int__(self,weight,target):
        super(Style_loss, self).__int__()
        self.weight = weight
        self.target = target.detach()*weight
        self.loss_fn = torch.nn.MSELoss()

        self.gram = Gram_matrix()

    def forward(self, input):
        self.Gram = self.gram(input.clone())
        self.Gram.mul_(self.weight)
        
        self.loss = self.loss_fn(self.Gram, self.target)
        return input

    def backward(self):
        self.loss.backward(retain_graph = True)
        return self.loss
