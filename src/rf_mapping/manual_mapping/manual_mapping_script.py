import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Cursor
import torch
import torchvision as models


# x = np.linspace(0, 10, 100)
# y = np.exp(x**0.5) * np.sin(5*x)

# fig = plt.figure()
# ax = fig.subplots()
# ax.plot(x, y, color='b')
# ax.grid()

# # Defining cursor
# cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=1)

# annot = ax.annotate("hi", xy=(0, 0), xy_text=(40, 40), textcoords="offset_points")

# plt.show()


# def plot_unit_circle():
#     angs = np.linspace(0, 2 * np.pi, 10**6)
#     rs = np.zeros_like(angs) + 1
#     xs = rs * np.cos(angs)
#     ys = rs * np.sin(angs)
#     plt.plot(xs, ys)


# def mouse_move(event):
#     x, y = event.xdata, event.ydata
#     plt.plot(x, y)
#     plt.show()
#     print(x, y)


# plt.connect('motion_notify_event', mouse_move)
# plot_unit_circle()
# plt.axis('equal')
# plt.show()


# fig, ax = plt.subplots(dpi=100, figsize=(5, 5))
# x = np.arange(0, 6, 0.1)
# plt.plot(x, np.sin(x), 'r')

# fig2, ax2 = plt.subplots(dpi=100, figsize=(5, 5))

# circle = plt.Circle((0, 0), 1, color='b', fill=False)
# ax2.add_artist(circle)
# line, = ax2.plot([],[])
# ax2.set_xlim(-2, 2)
# ax2.set_ylim(-2, 2)

# def plot_ray(angle, y):
#     length = y / np.sin(angle)
#     line.set_data([0, length * np.cos(angle)], [0, length * np.sin(angle)])

# def mouse_move(event):
#     x = event.xdata
#     y = event.ydata
#     if x is not None and y is not None:
#         angle = x
#         plot_ray(angle, y)
#         fig2.canvas.draw_idle()


# cid = fig.canvas.mpl_connect('motion_notify_event', mouse_move)

# plt.show(block=True)


import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

def mouse_event(event):
    print('x: {} and y: {}'.format(event.xdata, event.ydata))

fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', mouse_event)

x = np.linspace(-10, 10, 100)
y = np.sin(x)

plt.plot(x, y)

plt.show()
