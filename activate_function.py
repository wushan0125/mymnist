import numpy as np
import matplotlib.pyplot as plt

ReLU_losses = np.load('activate_function/ReLU_losses.npy')
Sigmoid_losses = np.load('activate_function/Sigmoid_losses.npy')
Tanh_losses = np.load('activate_function/Tanh_losses.npy')

fig, ax = plt.subplots()
ax.plot(np.arange(len(ReLU_losses)), ReLU_losses, label="ReLu")
ax.plot(np.arange(len(Sigmoid_losses)), Sigmoid_losses, label="Sigmoid")
ax.plot(np.arange(len(Tanh_losses)), Tanh_losses, label="Tanh")
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('activate_function_loss')
ax.legend()
plt.show()

ReLU_acces = np.load('activate_function/ReLU_acces.npy')
Sigmoid_acces = np.load('activate_function/Sigmoid_acces.npy')
Tanh_acces = np.load('activate_function/Tanh_acces.npy')

fig, ax = plt.subplots()
ax.plot(np.arange(len(ReLU_acces)), ReLU_acces, label="ReLu")
ax.plot(np.arange(len(Sigmoid_acces)), Sigmoid_acces, label="Sigmoid")
ax.plot(np.arange(len(Tanh_acces)), Tanh_acces, label="Tanh")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
ax.set_title('activate_function_acc')
ax.legend()
plt.show()

f = open("activate_function/time.txt", encoding='utf-8')
times = []
for i in range(3):
    num = f.readline().split('：')[1]
    times.append(float(num))
plt.title('activate_function_time')
plt.bar(['ReLU', 'Sigmoid', 'Tanh'], times, width=0.2)
plt.show()