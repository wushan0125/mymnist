import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataloader import test_data
import matplotlib.pyplot as plt
from model import net

criterion=torch.nn.BCELoss()
eval_losses = []
eval_acces = []
eval_loss = 0
eval_acc = 0

#加载模块参数
model = net
model.load_state_dict(torch.load("ReLU.pth"))
#把dropout关掉，不进行参数更新
model.eval()

for im, label in test_data:
    im = Variable(im)
    label = Variable(label)
    out = model(im)
    # print(out.squeeze(-1))
    # print(label)
    loss = criterion(out, label)
    eval_loss += loss.item()
    _, pred = out.max(1)
    num_correct = (pred == label).sum().item()
    acc = num_correct / im.shape[0]
    eval_acc += acc
    eval_losses.append(loss.item()/im.shape[0])
    eval_acces.append(acc)


print('整体损失值：Eval Loss: {:.6f}, Eval Acc:{:.6f}'.format(eval_loss/len(test_data), eval_acc/len(test_data)))

np.save('b_64_test_losses.npy',eval_losses)
np.save('b_64_test_acces.npy',eval_acces)

plt.plot(np.arange(len(eval_losses)),eval_losses)
plt.title('b_64_test loss')
plt.show()

plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('b_64_test acc')
plt.show()
