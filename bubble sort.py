from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update(data): #draw line
    line.set_ydata(data)
    return line,

def data_gen(): #generate data
    for i in range(n-1):
        if a[i] > a[i+1]:
            a[i],a[i+1] = a[i+1],a[i]
    yield a

fig, ax = plt.subplots()
n = 200
a = [np.random.randint(1,100) for i in range(n)] 
line, = ax.plot(a,'go',lw=2,alpha=0.8)
ax.plot(np.arange(n),np.arange(n)/2,'k-',lw=40,alpha=0.1)
ax.set_title('Bubble sort')
ax.set_ylim(0, 100)
ax.set_xlabel('Inverse rank')
ax.set_ylabel('Value')

ani = animation.FuncAnimation(fig, update, data_gen, interval=100)
plt.show()
