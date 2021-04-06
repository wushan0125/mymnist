import numpy as np
import matplotlib.pyplot as plt

SGD_losses = np.load('optimizer/SGD_losses.npy')
Adam_losses = np.load('optimizer/Adam_losses.npy')

fig, ax = plt.subplots()
ax.plot(np.arange(len(SGD_losses)), SGD_losses, label="SGD")
ax.plot(np.arange(len(Adam_losses)), Adam_losses, label="Adam")
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('optimizer_loss')
ax.legend()
plt.show()

SGD_acces = np.load('optimizer/SGD_acces.npy')
Adam_acces = np.load('optimizer/Adam_acces.npy')

fig, ax = plt.subplots()
ax.plot(np.arange(len(SGD_acces)), SGD_acces, label="SGD")
ax.plot(np.arange(len(Adam_acces)), Adam_acces, label="Adam")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
ax.set_title('optimizer_acc')
ax.legend()
plt.show()

f = open("optimizer/time.txt", encoding='utf-8')
times = []
for i in range(2):
    num = f.readline().split('：')[1]
    times.append(float(num))
plt.title('optimizer_time')
plt.bar(['SGD', 'Adam'], times, width=0.2)
plt.show()