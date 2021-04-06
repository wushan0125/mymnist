import datetime
import numpy as np
import torch
from torch import nn
from model import net
from torch.autograd import Variable #tensor不能反向传播，variable可以反向传播
from dataloader import train_data
import matplotlib.pyplot as plt

# 交叉熵损失函数
'''
    二分类中的交叉熵损失函数：L = -[ylog(y)'+(1-y)log(1-y')]，其中，y'为预测值
'''
# criterion = torch.nn.BCELoss()
# criterion = torch.nn.MSELoss()
criterion = torch.nn.BCEWithLogitsLoss()

# 采用的优化器为：随机梯度下降，超参数：学习率η
optimizer = torch.optim.SGD(net.parameters(),1e-1)
# 采用的优化器为：Adam，超参数：学习率η
# optimizer = torch.optim.Adam([{'params':net.parameters()}],betas=(0.5, 0.999))

#损失值losses与精确值acces
losses = []
acces = []
start = datetime.datetime.now()
# 训练网络 epoch:20次
for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label).float()
        #前向传播
        out = net(im)
        out=out.squeeze(1)
        loss = criterion(out,label)
        #把梯度置零
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #更新网络权重
        optimizer.step()
        '''
        python0.4.0之前，loss是封装了(1,)张量的Variable，
        但Python0.4.0的loss现在是一个零维的标量，对标量进行索引是没有意义的，
        故使用loss.item()可以从标量中获取python数字
        '''
        train_loss += loss.item()
        result=out.data>=0.5
        label=label==1
        num_correct = (result == label).sum().item() #item()将一个值的张量变为标量，统计预测对的数
        acc = num_correct / im.shape[0]
        train_acc += acc
    losses.append(train_loss/len(train_data))
    acces.append(train_acc/len(train_data))
    print('epoch:{},Train Loss:{:.6f}, Train Acc: {:.6f}'.format(e, train_loss/len(train_data), train_acc/len(train_data)))

np.save('BCEWithLogits_losses.npy', losses)
np.save('BCEWithLogits_acces.npy', acces)

plt.title('train loss:Epoch_{}'.format(e+1))
plt.plot(np.arange(len(losses)),losses)
plt.show()

plt.title('train acc:Epoch_{}'.format(e+1))
plt.plot(np.arange(len(acces)),acces)
plt.show()

end = datetime.datetime.now()
print("training_time:{}".format(end-start))
# 保存训练好的模型与参数
torch.save(net.state_dict(), "BCEWithLogits.pth")
# net._save_to_state_dict('model.pt')