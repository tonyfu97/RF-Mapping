"""
Find the good number of bars used for our mapping methods.

Step 1. Map the RF using the specificied numbers of stimuli.

THIS SCRIPT TAKES A LOT OF TIME TO RUN!

Tony Fu, August 18th, 2022
"""
import os
import sys
from time import time

import numpy as np
from torchvision import models

sys.path.append('../../..')
from src.rf_mapping.bar import rfmp4a_run_01b, rfmp4c7o_run_01
from src.rf_mapping.grating import sin1_run_01b
from src.rf_mapping.pasu_shape import pasu_bw_run_01b
import src.rf_mapping.constants as c

# Please specify some details here:
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
this_is_a_test_run = False
batch_size = 10
conv_i_to_run = 1  # conv_i = 1 means Conv2
rfmp_name = 'rfmp4c7o'
# num_stim_list = [50, 100, 250, 500, 750, 1000, 1500, 2000]
num_stim_list = [3000, 5000, 10000]

result_dir = os.path.join(c.REPO_DIR, 'results', 'test_num_stim', rfmp_name)

###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take time to run. Are you sure? [y/n] ")
#     if user_input == 'y':
#         pass
#     else:
#         raise KeyboardInterrupt("Interrupted by user")

layer_name = f"conv{conv_i_to_run+1}"

if __name__ == '__main__':
    for num_stim in num_stim_list:
        # Create the result directory if it does not exist.
        this_result_dir = os.path.join(result_dir, model_name, layer_name, str(num_stim))
        if os.path.exists(result_dir) and not os.path.exists(this_result_dir):
            os.makedirs(this_result_dir)
        print(f"Result directory: {this_result_dir}...")
        
        start = time()
        if rfmp_name == 'rfmp4a':
            rfmp4a_run_01b(model, model_name, this_result_dir, _debug=this_is_a_test_run,
                           batch_size=batch_size, num_bars=num_stim, conv_i_to_run=conv_i_to_run)
        elif rfmp_name == 'rfmp4c7o':
            rfmp4c7o_run_01(model, model_name, this_result_dir, _debug=this_is_a_test_run,
                            batch_size=batch_size, num_bars=num_stim, conv_i_to_run=conv_i_to_run)
        elif rfmp_name == 'rfmp_sin1':
            sin1_run_01b(model, model_name, this_result_dir, _debug=this_is_a_test_run,
                         batch_size=batch_size, conv_i_to_run=conv_i_to_run)
        elif rfmp_name == 'pasu':
            pasu_bw_run_01b(model, model_name, this_result_dir, _debug=this_is_a_test_run,
                            batch_size=batch_size, conv_i_to_run=conv_i_to_run)
        end = time()

        total_time = end - start        
        
        speed_txt_path = os.path.join(result_dir, model_name, layer_name, f"speed.txt")
        with open(speed_txt_path, 'a') as f:
            f.write(f"{num_stim} {total_time}\n")
