import math
import numpy as np
import torch
import matplotlib.pyplot as plt

def stimfr_sine(xn,yn,x0,y0,diam,sf,phase,theta,bgval,contrast):
#
#  Return a numpy array that has a circular patch of sinusoidal grating
#  in the range -1.0 to 1.0.
#
#  xn    - (int) horizontal width of returned array
#  yn    - (int) vertical height of returned array
#  x0    - (float) horizontal offset of center (pix)
#  y0    - (float) vertical offset of center (pix)
#  diam  - (float) diameter (pix)
#  sf    - (float) cycles/pix
#  phase - (float) spatial phase (deg)
#  theta - (float) orientation (deg)
#  bgval - (float) background value [-1.0 ... 1.0]
#  contrast - (float) [0.0 ... 1.0]
#
  s = np.full((yn,xn), bgval, dtype='float32')  # Fill w/ BG value
  
  #print("sf ", sf)
  #print("theta ", theta)
  #print("phase ", phase)
  #print("bgval ", bgval)
  #print("contrast ", contrast)
  
  # Set the phase origin to the center of the patch
  cxi = (xn-1.0)/2.0
  cyi = (yn-1.0)/2.0
  rad2 = (diam/2.0 * diam/2.0)
  
  twopif = 2.0*math.pi*sf
  ph = phase/180.0 * math.pi
  nx = math.cos(theta/180.0 * math.pi)
  ny = math.sin(theta/180.0 * math.pi)
  
  for i in range(0,xn):
    x = i-cxi - x0
    dx2 = (i-(cxi+x0))*(i-(cxi+x0))
    for j in range(0,yn):
      y = j-cyi - y0
      dy2 = (j-(cxi+y0))*(j-(cxi+y0))
      if (dx2+dy2 < rad2):
        s[yn-1-j,i] = contrast * math.cos(twopif*(nx*x + ny*y) - ph)
  
  return s


def stimfr_sine_thresh_list(xn,yn,x0,y0,diam,sf,phase,theta):
#
#  MUST MATCH EXACTLY TO 'stimfr_sine' above
#  Returns a list of coordinates inside grating patch
#
  
  # Set the phase origin to the center of the patch
  cxi = (xn-1.0)/2.0
  cyi = (yn-1.0)/2.0
  rad2 = (diam/2.0 * diam/2.0)
  
  twopif = 2.0*math.pi*sf
  ph = phase/180.0 * math.pi
  nx = math.cos(theta/180.0 * math.pi)
  ny = math.sin(theta/180.0 * math.pi)
  
  clist = []  # Start with empty list of coordinates
  for i in range(0,xn):
    x = i-cxi - x0
    dx2 = (i-(cxi+x0))*(i-(cxi+x0))
    for j in range(0,yn):
      y = j-cyi - y0
      dy2 = (j-(cxi+y0))*(j-(cxi+y0))
      if (dx2+dy2 < rad2):
        #s[yn-1-j,i] = contrast * math.cos(twopif*(nx*x + ny*y) - ph)
        clist.append([yn-1-j,i])
  
  return clist

#######################################.#######################################
#                                                                             #
#                                PARSET_SINE_1                                #
#                                                                             #
#  Return lists of sine grating parameters for varying SF,                    #
#  orientation (theta), spatial phase and size, based on a maximum size       #
#  value 'smax'                                                               #
#                                                                             #
###############################################################################
def parset_sine_1(smax):
  
  sr2 = math.sqrt(2.0)
  
  sf    = np.asarray([0.016, 0.032, 0.064, 0.128, 0.256])
  theta = np.asarray([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5])
  size  = np.asarray([smax/(8*sr2), smax/8, smax/(4*sr2),
           smax/ 4, smax/(2*sr2), smax/2, smax/sr2, smax])
  phase = np.asarray([0, 45, 90, 135, 180, 225, 270, 315])
  
  return sf,theta,size,phase

#######################################.#######################################
#                                                                             #
#                                PARSET_SINE_2                                #
#                                                                             #
###############################################################################
def parset_sine_2(smax):
  
  sr2 = math.sqrt(2.0)
  
  sf    = np.asarray([0.016, 0.023, 0.032, 0.045, 0.064,
                      0.091, 0.128, 0.181, 0.256, 0.362])
  theta = np.asarray([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5])
  size  = np.asarray([smax/(8*sr2), smax/8, smax/(4*sr2),
           smax/4, smax/(2*sr2), smax/2, smax/sr2, smax])
  phase = np.asarray([0, 45, 90, 135, 180, 225, 270, 315])
  
  return sf,theta,size,phase

#######################################.#######################################
#                                                                             #
#                            STIMSET_SINE_SIZE_BATCH                          #
#                                                                             #
###############################################################################
def stimset_sine_size_batch(xn,off0,off1,b0,bn,sset):
  #
  #  xn    - width and height of stimulus (pix)
  #  off0  - offset for centering along the 1st axis (rows)
  #  off1  - offset for centering along the 2nd axis (columns)
  #  b0    - batch start index
  #  bn    - batch size
  #  sset - name of stimulus set, e.g., 'sine_1', 'sine_2'
  #
  
  if (sset == 'sine_1'):
    sf,theta,size,phase = parset_sine_1(xn)
  elif (sset == 'sine_2'):
    sf,theta,size,phase = parset_sine_2(xn)
  else:
    return None, -1
  
  nstim = len(sf) * len(theta) * len(size) * len(phase)  # total stimuli
  
  # Compute the number of stimuli to generate for this batch
  if (nstim - b0 >= bn):
    ngen = bn               # Generate a full batch
  else:
    ngen = nstim - b0       # Generate partial batch
  
  if (b0 < 0) or (ngen < 1):
    return np.empty(0), 0
  
  #print("Pars:",xn,off0,off1,b0,bn," Ngen = ",ngen)
  
  d = np.empty((ngen,3,xn,xn), dtype='float32')
  
  bgval =  0.0   # Mean gray background
  con   =  1.0   # Full contrast
  
  k = 0
  for s in sf:
    for t in theta:
      for z in size:
        for p in phase:
          if (k >= b0) & (k < b0+bn):
            d[k-b0][0] = stimfr_sine(xn,xn,off1,-off0,z,s,p,t,bgval,con)
            d[k-b0][1] = d[k-b0][0]
            d[k-b0][2] = d[k-b0][0]
          k += 1
  
  return d, nstim       # Return a batch of images, and the total set size

#######################################.#######################################
#                                                                             #
#                            STIMSET_SINE_SIZE_SINGLE                         #
#                                                                             #
#  Return a set of 1 stimulus, at the specified indices.                      #
#                                                                             #
###############################################################################
def stimset_sine_size_single(xn,off0,off1,isf,ith,isz,iph,sset):
  #
  #  xn    - width and height of stimulus (pix)
  #  off0  - offset for centering along the 1st axis (rows)
  #  off1  - offset for centering along the 2nd axis (columns)
  #  isf   - index into SF dimension
  #  ith   - index into theta dimension
  #  isz   - index into size dimension
  #  iph   - index into phase dimension
  #  sset - name of stimulus set, e.g., 'sine_1', 'sine_2'
  #
  
  if (sset == 'sine_1'):
    sf,theta,size,phase = parset_sine_1(xn)
  elif (sset == 'sine_2'):
    sf,theta,size,phase = parset_sine_2(xn)
  else:
    return None
  
  d = np.empty((1,3,xn,xn), dtype='float32')
  
  bgval =  0.0   # Mean gray background
  con   =  1.0   # Full contrast
  
  s =    sf[isf]
  t = theta[ith]
  z =  size[isz]
  p = phase[iph]
  d[0][0] = stimfr_sine(xn,xn,off1,-off0,z,s,p,t,bgval,con)
  d[0][1] = d[0][0]
  d[0][2] = d[0][0]
  
  return d       # Return a batch of 1 stimuli

#######################################.#######################################
#                                                                             #
#                               STIMSET_BASELINE                              #
#                                                                             #
###############################################################################
def stimset_baseline(xn,bgval):
  #
  #  xn    - width and height of stimulus (pix)
  #  bgval - value of input image
  #
  
  nstim = 1
  d = np.empty((nstim,3,xn,xn), dtype='float32')
  
  s = np.full((xn,xn), bgval, dtype='float32')  # Fill w/ BG value
  
  d[0][0] = s = np.full((xn,xn), bgval, dtype='float32')
  d[0][1] = d[0][0]
  d[0][2] = d[0][0]
  
  return d

#######################################.#######################################
#                                                                             #
#                             GET_FOURIER_HARMONIC                            #
#                                                                             #
#  WYETH - this was copied from C-code, needs to be vectorized                #
#        - also could add frequency (wavelength) parameter                    #
#                                                                             #
###############################################################################
def get_fourier_harmonic(d,order):
  #
  #  d     - data array
  #  order - index of fourier component to compute, 1=f1, 2=f2, etc, 0=DC
  #
  if (order == 0):
    return d.sum()/len(d), 0.0   # Return average of array
  
  todd = teven = 0.0
  n = len(d)
  for i in range(n):
    x = i/(n/order) * 2.0*math.pi
    todd  += 2.0 * math.sin(x) * d[i]
    teven += 2.0 * math.cos(x) * d[i]
  
  teven /= n
  todd  /= n
  ampl = math.sqrt(todd*todd + teven*teven)
  theta = math.atan2(todd,teven) * 180.0/math.pi
  
  return ampl,theta
  
  #import matplotlib.pyplot as plt
  #a = np.arange(0,2.0*math.pi,math.pi/4)
  #sa = np.sin(a)
  #plt.plot(sa)
  #plt.show()
  #
  #amp,thet = get_fourier_harmonic(sa,1)
