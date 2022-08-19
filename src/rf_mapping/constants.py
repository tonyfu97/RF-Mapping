import torch
# Use 'mps' if using Apple silicon
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPO_DIR = '/home/tfu/git_repos/borderownership'
IMG_DIR = '/data/images_npy'

