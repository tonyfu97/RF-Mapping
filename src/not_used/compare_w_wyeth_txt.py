import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(25,5))
for i in range(2,6):
    wyeth_path = f'/Users/tonyfu/Desktop/n01_conv{i}_nat50k_r.txt'
    wyeth_conv_nat_r = pd.read_csv(wyeth_path, sep=' ', header=None)
    wyeth_conv_nat_r.columns = ['UNIT', 'MAX_R', 'MIN_R', 'MAX_10_AVG', 'MIN_10_AVG']

    my_path = f'/Users/tonyfu/Desktop/Bair Lab/borderownership/results/ground_truth/top_n/alexnet/conv{i}_responses.npy'
    my_conv_nat_r = np.load(my_path)
    my_conv_nat_r = np.mean(np.sort(my_conv_nat_r, axis=1)[:, -10:, 0], axis=1)

    plt.subplot(1,4,i-1)
    plt.scatter(wyeth_conv_nat_r.MAX_10_AVG, my_conv_nat_r)
    plt.xlabel('wyeth')
    plt.ylabel('tony')
    plt.title(f"conv{i} r")
    plt.gca().set_aspect('equal')
    plt.xlim(0, 300)
    plt.ylim(0, 300)

plt.show()


plt.figure(figsize=(25,5))
for i in range(2,6):
    wyeth_path = f'/Users/tonyfu/Desktop/n01_conv{i}_bar4a_minmax.txt'
    wyeth_conv_bar_r = pd.read_csv(wyeth_path, sep=' ', header=None)
    wyeth_conv_bar_r.columns = ['UNIT', 'MIN_STIM_I', 'MIN_R', 'MAX_STIM_I', 'MAX_R']

    my_path = f'/Users/tonyfu/Desktop/Bair Lab/borderownership/results/rfmp4a/mapping/alexnet/conv{i}_top5000_responses.txt'
    my_conv_bar_r = pd.read_csv(my_path, sep=' ', header=None)
    my_conv_bar_r.columns = ['UNIT', 'RANK', 'STIM_I', 'MAX_R']
    my_conv_bar_r = my_conv_bar_r.loc[my_conv_bar_r.RANK == 1, ['UNIT', 'STIM_I', 'MAX_R']]

    plt.subplot(1,4,i-1)
    plt.scatter(wyeth_conv_bar_r.MAX_R, my_conv_bar_r.MAX_R)
    plt.xlabel('wyeth')
    plt.ylabel('tony')
    plt.title(f"conv{i} r")
    plt.gca().set_aspect('equal')
    plt.xlim(0, 45)
    plt.ylim(0, 45)

plt.show()
