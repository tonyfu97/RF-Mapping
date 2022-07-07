""" 
Calculates the mean and std of imagenet dataset used.

Tony Fu, June 25, 2022
"""
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append('..')
import constants as c

# Please specify some details here:
num_images = 50000

# Please double-check the directories:
img_dir = c.IMG_DIR
img_names = [f"{i}.npy" for i in range(num_images)]

###############################################################################

# Script guard.
if __name__ == "__main__":
    user_input = input("This code takes time to run. Are you sure? "\
                       "Enter 'y' to proceed. Type any other key to stop: ")
    if user_input == 'y':
        pass
    else: 
        raise KeyboardInterrupt("Interrupted by user")

avg_mean = np.ones((3,), dtype=np.float32)
avg_std = np.ones((3,), dtype=np.float32)
for img_name in tqdm(img_names):
    img_path = os.path.join(img_dir, img_name)
    img = np.load(img_path)
    print("mean")
    print(np.mean(np.mean(img, axis=1), axis=1))
    print("std")
    print(np.std(np.std(img, axis=1), axis=1))
    avg_mean += np.mean(np.mean(img, axis=1), axis=1)
    avg_std += np.std(np.std(img, axis=1), axis=1)
    
avg_mean /= num_images
avg_std /= num_images

print(f"avg_mean = {avg_mean}")
print(f"avg_std = {avg_std}")

# avg_mean = [-0.01618503 -0.01468056 -0.01345447]
# avg_std = [0.09958131 0.0980152  0.10090139]
