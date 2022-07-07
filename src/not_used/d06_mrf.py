"""
RFMp4a - Paradigm 4a Examples
See http://wartburg.biostr.washington.edu/loc/course/artiphys/data/rfmap_par.html

Wyeth Bair, date unknown

Modified slightly by Tony Fu, July 1, 2022
"""
import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt
#
#  Example code for MRF mapping strategy
#

###
###  AlexNet - mapping paradigm 4a  (RFMp41)
###
mod = models.alexnet(pretrained=True)
i_conv1 =  0   # Store layer indices ("i_...") for particular layers
i_conv2 =  3
i_conv3 =  6
i_conv4 =  8
i_conv5 = 10

exec(open('./d06_util_bargen.py').read())
exec(open('./d06_util_sinmap.py').read())

###          model layers to run      xn  maxRF  net   layer  show
#            ------------------------ --- ----  ----  ------- ----
rfmap_top_4a(mod.features[:i_conv2+1], 63,  51, 'n01','conv2',0)
rfmap_top_4a(mod.features[:i_conv3+1],127,  99, 'n01','conv3',0)
rfmap_top_4a(mod.features[:i_conv4+1],159, 131, 'n01','conv4',0)
rfmap_top_4a(mod.features[:i_conv5+1],191, 163, 'n01','conv5',0)

# Color
rfmap_top_4c6o(mod.features[:i_conv2+1], 63,  51, 'n01','conv2',0)

# Sine patches  (~7+? min, stim+map)
rfmap_top_sin1(mod.features[ : i_conv2+1], 63,  51, 'n01','conv2',0)
mv n01_stat_conv2_mrf_sin1.txt stat/
#  Start on my laptop at 10.20 am 6 SFs, 4 phases

rfmap_top_sin1(mod.features[:i_conv3+1],127,  99, 'n01','conv3',0)
mv n01_stat_conv3_mrf_sin1.txt stat/

#rfmap_top_sin1(mod.features[:i_conv4+1],159, 131, 'n01','conv4',0)



mod = models.vgg16(pretrained=True)
i_conv1  =  0   # Store layer indices ("i_...") for particular layers
i_conv2  =  2
i_conv3  =  5
i_conv4  =  7
i_conv5  = 10
i_conv6  = 12
i_conv7  = 14
i_conv8  = 17
i_conv9  = 19
i_conv10 = 21
i_conv11 = 24
i_conv12 = 26
i_conv13 = 28

exec(open('./d06_util_bargen.py').read())

###          model layers to run       xn  maxRF  net   layer  show
#            ------------------------  --- -----  ---  ------- ----
rfmap_top_4a(mod.features[:i_conv6 +1], 44,  32, 'n03','conv6' ,0)
rfmap_top_4a(mod.features[:i_conv7 +1], 54,  40, 'n03','conv7' ,0)
rfmap_top_4a(mod.features[:i_conv8 +1], 72,  60, 'n03','conv8' ,0)
rfmap_top_4a(mod.features[:i_conv9 +1], 88,  76, 'n03','conv9' ,0)
rfmap_top_4a(mod.features[:i_conv10+1],104,  92, 'n03','conv10',0)
rfmap_top_4a(mod.features[:i_conv11+1],144, 132, 'n03','conv11',0)
rfmap_top_4a(mod.features[:i_conv12+1],176, 164, 'n03','conv12',0)

### WYETH - I'm trying these on 'thwak' (conv12 runs out of memory)
rfmap_top_4c6o(mod.features[:i_conv6 +1], 44,  32, 'n03','conv6' ,0)
rfmap_top_4c6o(mod.features[:i_conv7 +1], 54,  40, 'n03','conv7' ,0)
rfmap_top_4c6o(mod.features[:i_conv8 +1], 72,  60, 'n03','conv8' ,0)
rfmap_top_4c6o(mod.features[:i_conv9 +1], 88,  76, 'n03','conv9' ,0)
rfmap_top_4c6o(mod.features[:i_conv10+1],104,  92, 'n03','conv10',0)

[CANT RUN ON THWAK, unless I use the "12" option rather than "7"]
rfmap_top_4c6o(mod.features[:i_conv11+1],144, 132, 'n03','conv11',0)
rfmap_top_4c6o(mod.features[:i_conv12+1],176, 164, 'n03','conv12',0)


# WYETH NOTE:  VVG16, Conv10 failed on 'kamblipoochi' (has 64 GB RAM)
#    I had to run this and deeper layers on 'teff'



###############################################################################
#
#  Test the position of RFs in AlexNet as the input image size is changed
#
#  This is how I found the special numbers that make the stimulus be centered
#  on the RF of the unit at the middle of the spatial grid.
#
###############################################################################
import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt

### (1) Initialization (AlexNet)
mod = models.alexnet(pretrained=True)

i_conv1 =  0   # Store layer indices ("i_...") for particular layers
i_conv2 =  3   # unit 10 is good
i_conv3 =  6   # unit 19 is good
i_conv4 =  8   # unit 17 is good
i_conv5 = 10   # unit  0 is good
m = mod.features[:i_conv2+1]

#   **** USE Either the lines above (alexnet), or the lines below (vgg16)

### (1) Initialization (VGG)
mod = models.vgg16(pretrained=True)
i_conv1  =  0   # Store layer indices ("i_...") for particular layers
i_conv2  =  2
i_conv3  =  5
i_conv4  =  7
i_conv5  = 10
i_conv6  = 12  # max_rf  32
i_conv7  = 14  # max_rf  40
i_conv8  = 17  # max_rf  60
i_conv9  = 19  # max_rf  76
i_conv10 = 21  # max_rf  92
i_conv11 = 24  # max_rf 132
i_conv12 = 26  # max_rf 164
i_conv13 = 28
m = mod.features[:i_conv9+1]

exec(open('./d06_util_bargen.py').read())

fgval = 1.0
bgval = 0.0
### ALEXNET  These values work!
xn =  63   # conv2: pix  6-56   *** MID PIX ***  7x7 grid
xn = 127   # conv3: pix 14-112  *** MID PIX ***  7x7 grid
xn = 159   # conv4: pix 14-144  *** MID PIX ***  9x9 grid
xn = 191   # conv5: pix 14-176  *** MID PIX *** 11x11 grid

xn = 227   # 

### VGG16  These values work!
xn =  44   # conv6:  pix  6-37   *** MID PIX *** 11x11 grid
xn =  52   # conv7:  pix  6-45   *** MID PIX ***  9x9 grid
xn =  72   # conv8:  pix  6-65   *** MID PIX ***  9x9 grid
xn =  88   # conv9:  pix  6-81   *** MID PIX *** 11x11 grid
xn = 104   # conv10: pix  6-97   *** MID PIX *** 13x13 grid
xn = 144   # conv11: pix  6-137  *** MID PIX ***  9x 9 grid
xn = 176   # conv12: pix  6-169  *** MID PIX *** 11x11 grid


d = stimset_lines(xn,xn,fgval,bgval)  # Create the line stimuli
ttstim = torch.tensor(d)      # Convert numpy to torch.tensor
r = m.forward(ttstim)         # r[stim, zfeature, x, y] Run the model
print(r.shape)
mi = int((len(r[0,0,0]) - 1)/2)   # index for middle of spatial array
#mi = 0
#
#  Plot the responses for any unit 'k'
#
k = 0   # Unit 0, for example, change this to other values
td = r[:,k,mi,mi].cpu().detach()  # Get responses for middle units only
nd = td.numpy()                   # Convert back to numpy
# Now plot 1st half for the horizontal map, then 2nd 1/2 for vertical map
plt.plot(nd[ 0:xn],   marker='o'); plt.title("Unit " + str(k)); plt.show()
plt.plot(nd[xn:2*xn], marker='o'); plt.title("Unit " + str(k)); plt.show()


#
#  Or, use this loop to go through both (horiz and vert) maps for many units
#
for k in range(192):   # 192 is for Conv2, other layers have other values
  td = r[:,k,mi,mi].cpu().detach()  # Get responses for middle units only
  nd = td.numpy()                   # Convert back to numpy
  
  plt.plot(nd[ 0:xn],   marker='o'); plt.title("Unit " + str(k)); plt.show()
  plt.plot(nd[xn:2*xn], marker='o'); plt.title("Unit " + str(k)); plt.show()



###############################################################################
###
###  Compute ratio of max response
###
###     I use the following routine to compute 'f_nat' and merge it with 
###  the MRF statistics files, and then to write .txt and .npy formats.
###
###############################################################################
import numpy as np

def stat_convert(netname,layname,pname):
  #
  #  netname - network ID, e.g., 'n01', 'n03' ...
  #  layname - layer name, e.g., 'conv2' ...
  #  pname   - mapping paradigm name, e.g., 'mrf_4a', 'mrf_4c6o'
  #
  lpstr = layname + "_" + pname   # string w/ layer and paradigm
  infile_top  =     "../vis/" + netname + "_stat_" + layname + "_t10_r.npy"
  infile_mrf  =     "./stat/" + netname + "_stat_" + lpstr + ".txt"
  outfile_txt = "./stat_new/" + netname + "_stat_" + lpstr + ".txt"
  outfile_npy = "./stat_new/" + netname + "_stat_" + lpstr + ".npy"
  print("  infile_top = ",infile_top)
  print("  infile_mrf = ",infile_mrf)
  print("  outfile_txt = ",outfile_txt)
  print("  outfile_npy = ",outfile_npy)
  top_r = np.load(infile_top)  # top 10 responses
  
  with open(infile_mrf, 'r') as fin:
    lines = fin.read().splitlines()    # Read orig MRF stats
  
  n = len(lines)
  rbar = np.empty(n)
  for i in range(n):
    rbar[i] = float(lines[i].split()[1])  # Exract response to mapping stim
  
  brat = np.empty(n)
  for i in range(n):
    brat[i] = rbar[i] / top_r[i].mean() # (max map resp.) / (mean top10 image)
  
  with open(outfile_txt,'a') as outfile:
    for i in range(n):
      xp = " %8.4f\n" % brat[i]
      wn = outfile.write(lines[i])  # Write original line
      wn = outfile.write(xp)  # Write ratio ('wn' is # of bytes written)
  
  d = np.empty((n,5))  # Make numpy array, to save in numpy format
  for i in range(n):
    for j in range(4):
      d[i][j] = float(lines[i].split()[j+1])
    d[i][4] = brat[i]
  
  np.save(outfile_npy,d)


### OLD FORMAT - need to add 3rd argument to these
stat_convert("n01","conv2","mrf_4a")
stat_convert("n01","conv3","mrf_4a")
stat_convert("n01","conv4","mrf_4a")
stat_convert("n01","conv5","mrf_4a")
d = np.load('./stat_new/n01_stat_conv5_mrf_4a.npy')

stat_convert("n03","conv7" ,"mrf_4a")
stat_convert("n03","conv8" ,"mrf_4a")
stat_convert("n03","conv9" ,"mrf_4a")
stat_convert("n03","conv10","mrf_4a")
stat_convert("n03","conv11","mrf_4a")
stat_convert("n03","conv12","mrf_4a")

### New format
stat_convert("n01","conv2","mrf_4c6o")
stat_convert("n01","conv3","mrf_4c6o")
stat_convert("n01","conv4","mrf_4c6o")
stat_convert("n01","conv5","mrf_4c6o")

stat_convert("n03","conv10","mrf_4c6o")

#stat_convert("n03","conv6","mrf_4c6o")
stat_convert("n03","conv7","mrf_4c6o")
stat_convert("n03","conv8","mrf_4c6o")
stat_convert("n03","conv9","mrf_4c6o")

stat_convert("n03","conv11","mrf_4c6o")
stat_convert("n03","conv12","mrf_4c6o")



stat_convert("n01","conv2","mrf_sin1")



conv_txt_2_npy("zz_n01_conv2_mrf_4a.txt","n01_stat_conv2_mrf_4a")
conv_txt_2_npy("zz_n01_conv3_mrf_4a.txt","n01_stat_conv3_mrf_4a")
conv_txt_2_npy("zz_n01_conv4_mrf_4a.txt","n01_stat_conv4_mrf_4a")
conv_txt_2_npy("zz_n01_conv5_mrf_4a.txt","n01_stat_conv5_mrf_4a")

p = np.load("n01_stat_conv2_mrf_4a.npy")

##############################################################################