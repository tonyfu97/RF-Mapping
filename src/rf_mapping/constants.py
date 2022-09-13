# git update-index --assume-unchanged src/rf_mapping/constants.py
# git update-index --skip-worktree src/rf_mapping/constants.py
import torch
# Use 'mps' if using Apple silicon
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPO_DIR = '/Users/tonyfu/Desktop/Bair Lab/borderownership'
IMG_DIR = '/Users/tonyfu/Desktop/Bair Lab/top_and_bottom_images/images'
TEXTURE_DIR = '/Users/tonyfu/Desktop/Bair Lab/describable_textures_dataset/images'
