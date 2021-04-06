import numpy as np
import matplotlib.pyplot as plt

b_16_losses = np.load('batch_size/b_16_losses.npy')
b_32_losses = np.load('batch_size/b_32_losses.npy')
b_64_losses = np.load('batch_size/b_64_losses.npy')
b_128_losses = np.load('batch_size/b_128_losses.npy')
b_256_losses = np.load('batch_size/b_256_losses.npy')

fig, ax = plt.subplots()
ax.plot(np.arange(len(b_16_losses)), b_16_losses, label="b_16")
ax.plot(np.arange(len(b_32_losses)), b_32_losses, label="b_32")
ax.plot(np.arange(len(b_64_losses)), b_64_losses, label="b_64")
ax.plot(np.arange(len(b_128_losses)), b_128_losses, label="b_128")
ax.plot(np.arange(len(b_256_losses)), b_256_losses, label="b_256")
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('batch_size_loss')
ax.legend()
plt.show()

b_16_acces = np.load('batch_size/b_16_acces.npy')
b_32_acces = np.load('batch_size/b_32_acces.npy')
b_64_acces = np.load('batch_size/b_64_acces.npy')
b_128_acces = np.load('batch_size/b_128_acces.npy')
b_256_acces = np.load('batch_size/b_256_acces.npy')

fig, ax = plt.subplots()
ax.plot(np.arange(len(b_16_acces)), b_16_acces, label="b_16")
ax.plot(np.arange(len(b_32_acces)), b_32_acces, label="b_32")
ax.plot(np.arange(len(b_64_acces)), b_64_acces, label="b_64")
ax.plot(np.arange(len(b_128_acces)), b_128_acces, label="b_128")
ax.plot(np.arange(len(b_256_acces)), b_256_acces, label="b_256")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
ax.set_title('batch_size_acc')
ax.legend()
plt.show()

f = open("batch_size/time.txt", "r")
times = []
for i in range(5):
    line = f.readline()
    times.append(float(line))
plt.title('batch_size_time')
plt.bar(['16', '32', '64', '128', '256'], times, width=0.2)
plt.show()