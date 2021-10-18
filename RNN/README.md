# RNN

### SOME  ERRORS
1. 维度不匹配错误

   ![](./维度不匹配错误.png)

   发生错误的地方在93行损失函数:

   

   ```python
   loss = loss_func(y_pred, y_train)
   ```

   怀疑是交叉熵损失函数以及view()的使用上有问题。

   解决了再回来更新

2. ……

