import numpy as np
import matplotlib.pyplot as plt
# %matplotlib ipympl


# import os, sys
# sys.path.append('rf_mapping')
# print(os.getcwd())

# from src.rf_mapping.hook import ConvUnitCounter

# np.random.seed(19680801)
# data = np.random.random((50, 50, 50))

# fig, ax = plt.subplots()

# for i in range(len(data)):
#     ax.cla()
#     ax.imshow(data[i])
#     ax.set_title("frame {}".format(i))
#     plt.pause(0.1)
#     plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

def init():
    line.set_ydata([np.nan] * len(x))
    return line,

def animate(i):
    line.set_ydata(np.sin(x + i / 100))  # update the data.
    return line,

ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=2, blit=True, save_count=50)

plt.show()