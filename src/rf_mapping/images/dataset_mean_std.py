""" 
Calculates the mean and std of imagenet dataset used.

Tony Fu, June 25, 2022
"""
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append('../../..')
import src.rf_mapping.constants as c

# Please specify some details here:
num_images = 50000

# Please double-check the directories:
img_dir = c.IMG_DIR
img_names = [f"{i}.npy" for i in range(num_images)]

###############################################################################

avg_mean = np.ones((3,), dtype=np.float32)
avg_std = np.ones((3,), dtype=np.float32)
for img_name in tqdm(img_names):
    img_path = os.path.join(img_dir, img_name)
    img = np.load(img_path)

    avg_mean += np.mean(np.mean(img, axis=1), axis=1)
    avg_std += np.std(img, axis=(1,2))
    
avg_mean /= num_images
avg_std /= num_images

print(f"avg_mean = {avg_mean}")
print(f"avg_std = {avg_std}")

# Results (old data):
# avg_mean = [-0.01618503 -0.01468056 -0.01345447]
# avg_std = [0.45679083 0.44849625 0.44975275]

# Results (new data: already normalized, October 1st)
# avg_mean = [-5.2506704e-05 -9.0828044e-06 -6.2262559e-05]
# avg_std = [0.9999631  1.0000983  0.99994624]
