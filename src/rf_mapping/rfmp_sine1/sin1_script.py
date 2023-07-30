"""
Receptive field mapping paradigm sin1 (sinosoidal gratings).

Note: all code assumes that the y-axis points downward.

Tony Fu, Sep 14th, 2022
"""
import os
import sys

from torchvision import models

sys.path.append('../../..')
from src.rf_mapping.grating import sin1_run_01b
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
response_thr = 0.5

# Please double-check the directories:
if this_is_a_test_run:
    result_dir = os.path.join(c.RESULTS_DIR, 'rfmp_sin1', 'mapping', 'test')
else:
    result_dir = os.path.join(c.RESULTS_DIR, 'rfmp_sin1', 'mapping', model_name)

###############################################################################

# Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take time to run. Are you sure? [y/n] ")
#     if user_input == 'y':
#         pass
#     else:
#         raise KeyboardInterrupt("Interrupted by user")

if __name__ == '__main__':
    sin1_run_01b(model, model_name, result_dir, _debug=this_is_a_test_run,
                 batch_size=batch_size, response_thr=response_thr)
