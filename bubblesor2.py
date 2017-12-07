from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update(data):
    del ax.collections[0]
    ax.scatter(np.arange(n),data,c=data,s=2*np.array(data),alpha=0.5)
    return

def data_gen(): #generate data
    for i in range(n-1):
        if a[i] > a[i+1]:
            a[i],a[i+1] = a[i+1],a[i]

    yield a

fig = plt.figure()
ax = fig.add_subplot(111)
m,n = 100,100
a = [np.random.randint(1,m) for i in range(m)]
ax.scatter(np.arange(n),a,c=a,s=2*np.array(a),alpha=0.5)
ax.plot(np.arange(n),m/n*np.arange(n),'k-',lw=30,alpha=0.1)
ax.set_title('Bubble sort')
ax.set_ylim(0, m)
ax.set_xlim(0, n)
ax.set_xlabel('Inverse rank')
ax.set_ylabel('Value')

ani = animation.FuncAnimation(fig, update, data_gen, interval=100)
plt.show()
