from torch import nn

net = nn.Sequential(
    #输入图像像素个数为784：28*28；输出图像像素个数为400
    nn.Linear(784,400),
    #模型线性表达能力：激活函数：ReLU
    nn.Tanh(),
    nn.Linear(400,200),
    nn.Tanh(),
    nn.Linear(200,100),
    nn.Tanh(),
    nn.Linear(100,1),
    nn.Sigmoid()
)