import os
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

train_set = []
test_set = []

#MNIST数据集进行二分类任务，可将≥5的作为正实例（1），<5的作为负实例（-1）
#数据预处理
def data_tf(x):
    x = np.array(x,dtype='float32')/255
    x = (x-0.5)/0.5 #normalization
    x = x.reshape((-1,)) #flatten，使多维张量变为一维向量
    x = torch.from_numpy(x) #将x转换为torch类型
    return x

#取出训练集与测试集总的数据，这里取出的数据形式表现为[列表中的元组]，即图片与标签的组合
train_set_r = torchvision.datasets.MNIST(root='./mnist',train=True,transform=data_tf,download=True)
test_set_r = torchvision.datasets.MNIST(root='./mnist', train=False,transform=data_tf,download=True)

for x in range(2):
    for i in range(len(train_set_r)):
        img,target = train_set_r.__getitem__(index=i)
        if target == x:
            train_set.append(train_set_r[i])
    for j in range(len(test_set_r)):
        img,target = test_set_r.__getitem__(index=j)
        if target == x:
            test_set.append(test_set_r[j])
'''数据集的载入：
    shuffle=True：每一个epoch过程中会打乱数据顺序，重新随机选择
    '''
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)




