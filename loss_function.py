import numpy as np
import matplotlib.pyplot as plt

BCE_losses = np.load('loss_function/BCE_losses.npy')
MSE_losses = np.load('loss_function/MSE_losses.npy')
BCEWithLogits_losses = np.load('loss_function/BCEWithLogits_losses.npy')

fig, ax = plt.subplots()
ax.plot(np.arange(len(BCE_losses)), BCE_losses, label="BCE")
ax.plot(np.arange(len(MSE_losses)), MSE_losses, label="MSE")
ax.plot(np.arange(len(BCEWithLogits_losses)), BCEWithLogits_losses, label="BCEWithLogits")
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('loss_function_loss')
ax.legend()
plt.show()

BCE_acces = np.load('loss_function/BCE_acces.npy')
MSE_acces = np.load('loss_function/MSE_acces.npy')
BCEWithLogits_acces = np.load('loss_function/BCEWithLogits_acces.npy')

fig, ax = plt.subplots()
ax.plot(np.arange(len(BCE_acces)), BCE_acces, label="BCE")
ax.plot(np.arange(len(MSE_acces)), MSE_acces, label="MSE")
ax.plot(np.arange(len(BCEWithLogits_acces)), BCEWithLogits_acces, label="BCEWithLogits")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
ax.set_title('loss_function_acc')
ax.legend()
plt.show()

f = open("loss_function/time.txt", encoding='utf-8')
times = []
for i in range(3):
    num = f.readline().split('：')[1]
    times.append(float(num))
plt.title('time')
plt.bar(['BCE', 'MSE', 'BCEWithLogits'], times, width=0.2)
plt.show()