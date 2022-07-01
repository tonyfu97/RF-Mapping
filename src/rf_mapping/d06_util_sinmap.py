import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
exec(open('./d03_mkstim.py').read())
import pickle


#
#  Run the stimulus and update response list
### WYETH COPIED VERBATIM FROM 'd06_util_bargen.py'  SHOULD UNIFY
### WYETH COPIED VERBATIM FROM 'd06_util_bargen.py'  SHOULD UNIFY
### WYETH COPIED VERBATIM FROM 'd06_util_bargen.py'  SHOULD UNIFY
### WYETH COPIED VERBATIM FROM 'd06_util_bargen.py'  SHOULD UNIFY
### WYETH COPIED VERBATIM FROM 'd06_util_bargen.py'  SHOULD UNIFY
def mrfmap_run_01(srlist,dstim,m):
  #
  #  srlist - response list to append
  #   dstim - stimulus data as numpy array
  #
  
  nstim = len(dstim)                # Number of stimuli
  ttstim = torch.tensor(dstim)      # Convert numpy to torch.tensor
  r = m.forward(ttstim)             # r[stim, zfeature, x, y] Run the model
  #print(r.shape)
  
  mi = int((len(r[0,0,0]) - 1)/2)   # index for middle of spatial array
  zn = len(r[0])                    # Number of unique units
  td = r[:,:,mi,mi].cpu().detach()  # Get responses for middle units only
  nd = td.numpy()                   # Convert back to numpy
  
  #  Store the responses for each unit, for each stimulus in 'srlist',
  #     appending to this list.
  if len(srlist) == 0:  # If 'srlist' empty, create empty lists for each 'zn'
    for i in range(zn):
      srlist.append([])
  
  for i in range(zn):         # For each unique unit (feature index)
    for j in range(nstim):
      srlist[i].append(nd[j,i])

#######################################.#######################################
#                                                                             #
#                               MRFMAP_MAKE_MAP_1S                            #
#                                                                             #
#   *** These comments need to be organized...                                #
#                                                                             #
#  Create a sorted index list 'isort' that goes from the largest to
#     smallest response in 'srlist[z]'
#     Take the largest response:
#       rmax = srlist[z][ isort[0] ]
#     and define a threshold response to be some fraction 'frac' of this:
#       rmin = r_thr * rmax   # threhold response level
#
#     Set our MRF map 'mrf[xn][yn]'  to all zeros (assuming images are
#       'xn' by 'yn' pixels.
#
#   ...
#
###############################################################################
def mrfmap_make_map_1s(xn,srlist,splist,zi,r_thr):
  #  xn     - size of map (pix)
  #  srlist - responses
  #  splist - stimulus parameters
  #  zi     - index of unit within response list
  #  r_thr  - threshold fraction of response
  
  tr = srlist[zi]               # a pointer to response array for 'zi' unit
  isort = np.argsort(tr)        # Get indices that sort the list (least 1st)
  isort = np.flip(isort)        # Flip array so that largest comes first
  rmax = tr[isort[0]]           # Maximum response in list
  rmin = r_thr * rmax           # threshold response value
  mrfmap = np.full((xn,xn),0.0) # MRF map starts with all zeros
  
  #print("  Unit",zi," Max response: ",rmax)
  
  #splist_plot(splist,isort[0])   # Plot best stimulus
  
  # For i = 0,1,2,... and while the srlist[z][i] > rmin
  #   i.  Generate a stimulus image 'img' from the parameters in 'splist[i]'
  #  ii.  For each pixel (x,y) in 'img' that differs from the background,
  #       if *all* such pixels are zero in 'mrf[x][y]', then set all of these
  #       pixels in mrf[x][y] to 1.
  
  nstim = len(splist)
  for si in range(nstim):      # for each stimulus
    if tr[isort[si]] < rmin:   # Stop loop at first subthreshold response
      break
    
    pd = splist[isort[si]]     # Parameter dictionary for this stimulus
    clist = stimfr_sine_thresh_list(xn,xn,pd['x0'],pd['y0'],pd['size'],
                                    pd['sf'],pd['ph'],pd['theta'])
    
    tot = 0.0
    for c in clist:
      tot += mrfmap[c[0],c[1]]  # Add values in current mrfmap where stim > thr
    
    if (tot <= 0.0):  # If this stimulus does not touch any part of the MRF
      rval = tr[isort[si]] / rmax  # Normalized response: 1 down to r_thr
      for c in clist:
        mrfmap[c[0],c[1]] = rval   # Set all stim positions to norm'd resp.
      print("  Added stimlus ",isort[si])
  
  return mrfmap, rmax

#######################################.#######################################
#                                                                             #
#                              STIMSET_MAKE_SINMAP                            #
#                                                                             #
#  Create a stimulus set for sine gratings, with three major                  #
#  dimensions of variation:  x- and y-position, and orientation.              #
#                                                                             #
#  The input parameters specify the range of x-values to use (and these       #
#  are replicated to use for the y-range as well, and the delta-orientation   #
#  'dori' to use for varying orientation.                                     #
#                                                                             #
#  The other parameters are held fixed across the set:                        #
#                                                                             #
###############################################################################
def stimset_make_sinmap(splist,xn,xmin,xmax,dx,ori0,dori,diam,sf,ph):
  #
  #  splist - stimulus parameter list - APPEND TO THIS LIST
  #  xn     - horizontal and vertical image size
  #  xmin   - Minimum x-coord (pix)
  #  xmax   - Maximum x-coord (pix)
  #  dx     - Delta-x, position step size (pix)
  #  ori0   - Starting value for ori (degr)
  #  dori   - Delta-orientation, step size (degr)
  #  diam   - Diameter (pix)
  #  sf     - Spatial frequency (cyc/pix)
  #  ph     - Phase (deg)
  #
  
  yn = xn  # Assuming image is square
  xlist = ylist = np.arange(xmin,xmax+1,dx)
  orilist = np.arange(ori0, 180.0, dori)
  
  #  This will be a 4D array with the following dimensions:
  #    [nstim, 3, xn, yn]
  #
  contrast =  1.0  # Contrast
  bgval    =  0.0  # Background luminance
  
  nstim = len(xlist) * len(ylist) * len(orilist)
  print("  Creating ", nstim, " stimuli.")
  
  d = np.empty((nstim,3,xn,yn), dtype='float32')
  
  k = 0
  for i in xlist:
    for j in ylist:
      for o in orilist:
        #s = stimfr_bar(xn,yn,i,j,o,blen,bwid,aa,fgval,bgval)
        s = stimfr_sine(xn,yn,i,j,diam,sf,ph,o,bgval,contrast)
        #s= stimfr_sine(xn,yn,x0,y0,diam,sf,phase,theta,bgval,contrast):
        
        d[k][0] = s
        d[k][1] = s
        d[k][2] = s
        k += 1
        # This makes a dictionary 'tp'
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "size":diam,
              "sf":sf, "contrast":contrast, "bgval":bgval, "ph":ph}
        splist.append(tp)
  
  return d

#######################################.#######################################
#                                                                             #
#                              SINMAP_RUN4_AS_7                               #
#                                                                             #
#  Run 4 sets of diams at multi-scale grid, but for the fourth set, divide    #
#  it into 4 groups, for a total of 7 stimulus sets.                          #
#                                                                             #
#  Also, run for each of 4 phases.                                            #
#                                                                             #
###############################################################################
def sinmap_run4_as_7(splist,srlist,m,xn,max_rf,sf,showflag):
  #
  #  splist   - stimulus parameter list
  #  srlist   - response list
  #  m        - model
  #  xn       - stimulus image size
  #  max_rf   - maximum RF size (pix)
  #  sf       - spatial frequency
  #  showflag - 1-show the grid, 0-do not show
  #
  n = 7
  gdiam = np.array([48/64 * max_rf,     #  Array of grating diameters
                    24/64 * max_rf,
                    12/64 * max_rf,
                     6/64 * max_rf,
                     6/64 * max_rf,
                     6/64 * max_rf,
                     6/64 * max_rf])
  
  dori = np.array([22.5, 22.5, 22.5, 90.0, 90.0, 90.0, 90.0]) # Ori increment
  ori0 = np.array([ 0.0,  0.0,  0.0,  0.0, 22.5, 45.0, 67.5]) # Initial ori.
  
  dx = gdiam / 2.0                     #  Array of grid spacing
  
  xmax = np.empty(n)
  for i in range(n):
    xmax[i] = round((max_rf/dx[i]) / 2.0) * dx[i]
  
  phase = np.array([ 0.0, 90.0, 180.0, 270.0 ])
  
  #xxn = int(round(1.5*xn))
  #if (showflag == 1):
  #  for i in range(n):  ### SHOW THE GRID
  #    o0 = ori0[i]   # Initial orientation (deg)
  #    do = dori[i]   # orientation step size (deg)
  #    t = stimset_barmap_showrange(xxn,max_rf,-xmax[i],xmax[i],dx[i],o0,do,
  #                                 blen[i],arat*blen[i],aa)
  #    plt.imshow(np.transpose(t, (1, 2, 0)))  # Make RGB dimension be last
  #    plt.show()
  
  for ph in phase:  
    for i in range(n):
      o0 = ori0[i]   # Initial orientation (deg)
      do = dori[i]   # orientation step size (deg)
      d = stimset_make_sinmap(splist,xn,-xmax[i],xmax[i],dx[i],o0,do,gdiam[i],
                              sf,ph)
      #d=stimset_make_barmap_bw(splist,xn,-xmax[i],xmax[i],dx[i],o0,do,blen[i],
      #                           arat*blen[i],aa)
      mrfmap_run_01(srlist,d,m)   # Show stim 'd' to model 'm', update 'srlist'


#######################################.#######################################
#                                                                             #
#                                RFMAP_TOP_SIN1                               #
#                                                                             #
#  Oct 7, 2020                                                                #
#  I am trying to follow what I did for bars, but do it for gratings, for     #
#  MRF mapping.                                                               #
#                                                                             #
###############################################################################
def rfmap_top_sin1(m,xn,max_rf,netid,layname,showflag):
  #
  #  m        - model to run
  #  xn       - size (pix) of stimulus image
  #  max_rf   - maximum RF size (pix) for units being mapped
  #  netid    - e.g. 'n01'
  #  layname  - e.g. 'conv2'
  #  showflag - 1-show RF maps, 0-do not show
  #
  
  splist = []  # splist[i] - list of parameters that generated stimulus 'i'
               #             Each entry is a dictionary
  srlist = []  # srlist[z][i] - responses for unit 'z' for each stimulus 'i'
               #             This is a list of a list of floats
  
  ### SF varies down this list of commands
  print("  SF 0.025")
  sinmap_run4_as_7(splist,srlist,m,xn,max_rf,0.025,0)
  print("  SF 0.05")
  sinmap_run4_as_7(splist,srlist,m,xn,max_rf,0.05,0)
  print("  SF 0.10")
  sinmap_run4_as_7(splist,srlist,m,xn,max_rf,0.10,0)
  print("  SF 0.20")
  sinmap_run4_as_7(splist,srlist,m,xn,max_rf,0.20,0)
  print("  SF 0.40")
  sinmap_run4_as_7(splist,srlist,m,xn,max_rf,0.40,0)
  
  print("  Length of stimulus parameter list:",len(splist))
  print("  Length of model response list:  ",len(srlist[0]))
  # Note, there are 113,280 stimuli for all five SFs together
  
  statfile = netid + '_stat_' + layname + '_mrf_sin1.txt'
  pdffile  = netid + '_'      + layname + '_psin1.pdf'
  spfile   = netid + '_'      + layname + '_sin1_sp.pickle'
  srfile   = netid + '_'      + layname + '_sin1_sr.npy'
  
  with open(spfile, 'wb') as f:  #  Dump the stimlist
    pickle.dump(splist, f)       #    to a pickle file.
  srnp = np.array(srlist)        #  Dump the srlist
  np.save(srfile,srnp)           #    to a .npy file
  
  mid = (xn-1)/2   # Middle pixel of the map
  r_thr = 0.10
  zn = len(srlist)
  with PdfPages(pdffile) as pdf:
    with open(statfile, 'a') as outfile:
      for zi in range(zn):
        mrfmap, rmax = mrfmap_make_map_1s(xn,srlist,splist,zi,r_thr)
        com0, com1, radius = mapstat_comr_1(mrfmap,0.9)
        xp = "%3d %8.4f %5.1f %5.1f %5.1f" % (zi,rmax,com0-mid,com1-mid,radius)
        print(xp)
        wn = outfile.write(xp + "\n")  # 'wn' is number of bytes written?
        #
        plt.imshow(mrfmap)   # 2D array as image
        plt.title(netid + "  " + layname + "   Unit " + str(zi))
        pdf.savefig()
        if (showflag == 1):
          plt.show()


#######################################.#######################################
#                                                                             #
#                                RFMAP_SPR_READ                               #
#                                                                             #
#  2021 May 20                                                                #
#  This routine reads in the pickle file containing 'splist' and 'srlist'     #
#    generated during an RF mapping run.                                      #
#                                                                             #
###############################################################################
def rfmap_spr_read(infile):
  #
  #  infile   - e.g. './map_spr/n01_conv2_sin1_spr.pickle'
  #
  with open(infile, 'rb') as f:
    splist = pickle.load(f)
  
  print("  Length of stimulus parameter list:",len(splist))
  #print("  Length of model response list:  ",len(srlist[0]))
  #print("  Number of units:  ",len(srlist))
  
  return splist


# Jun 17, 2021 - reading back mapping stats:
#
# exec(open('./d06_util_sinmap.py').read())
# splist = rfmap_spr_read("map_spr/n01_conv2_sin1_sp.pickle")
# srnp = np.load("map_spr/n01_conv2_sin1_sr.npy")
# srnp.shape
