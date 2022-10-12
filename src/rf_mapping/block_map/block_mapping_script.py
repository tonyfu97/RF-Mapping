"""
Use small square color block (on a background of zeros) to map the RF.

Note: all code assumes that the y-axis points downward.

Tony Fu, October 11, 2022
"""
import os
import sys

from torchvision import models

sys.path.append('../../..')
from src.rf_mapping.block import block_map_run
import src.rf_mapping.constants as c

# Please specify some details here:
model = models.alexnet(pretrained=True).to(c.DEVICE)
model_name = 'alexnet'
# model = models.vgg16(pretrained=True).to(c.DEVICE)
# model_name = 'vgg16'
# model = models.resnet18(pretrained=True).to(c.DEVICE)
# model_name = "resnet18"
this_is_a_test_run = False

# Please double-check the directories:
if this_is_a_test_run:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'block', 'mapping', 'test')
else:
    result_dir = os.path.join(c.REPO_DIR, 'results', 'block', 'mapping', model_name)

###############################################################################

# # Script guard
# if __name__ == "__main__":
#     print("Look for a prompt.")
#     user_input = input("This code may take time to run. Are you sure? [y/n] ")
#     if user_input == 'y':
#         pass
#     else: 
#         raise KeyboardInterrupt("Interrupted by user")

if __name__ == "__main__":
    block_map_run(model, result_dir, _debug=this_is_a_test_run,
                  batch_size=100, response_thres=0.5)
