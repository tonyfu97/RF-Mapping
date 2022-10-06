"""
Code for generating sinewave gratings.

Note: origin is at the center, and the y-axis points downward.

Credit: Wyeth Bair
Original script names: d06_util_bargen_new.py
                       d06_util_sinmap.py
                       d03_mkstim.py

September 13, 2022


There are many functions in this script. Here is the summary:
1. stimfr_sine(xn,yn,x0,y0,diam,sf,phase,theta,bgval,contrast)
        Returns 2D circular patch of sine.
2. parset_sine_1(smax) and parset_sine_2(smax)
        Returns parameter lists: sf, theta, size, phase
3. stimset_sine_size_batch(xn,off0,off1,b0,bn,sset)
        Returns a batch of stimfr_sine() (4D) and nstim.
4. stimset_sine_size_single(xn,off0,off1,isf,ith,isz,iph,sset)
        Return a "batch" of one stimfr_sine() (4D).
5. stimset_baseline(xn,bgval)
        Returns a uniform-color stimuli (3D).
6. get_fourier_harmonic(d,order)
        Returns the order-th frequency component of d in (ampl, theta) format.
7. stimset_stim_get_sin(d)
        Given the stimulus dictionary d, returns the stimfr_sine() (2D).
8. stimset_show_stim_sin(d)
        plt.imshow() the sine specificed by the stimulus dictionary d.
9. mrfmap_run_01(srlist,dstim,truncated_model)
        Doesn't return, but appends (2D: unit_i & stim_i) responses to srlist.
10. mrfmap_make_map_1s(xn,srlist,splist,zi,r_thr)
        Make non-overlap map.
11. stim_dapp_sin_xyo_bw(splist,xn,xlist,orilist,diam,sf)
        Doesn't return, but appends parameter dictionaries to splist.
12. stimset_gridx_barmap(max_rf,blen)
        Returns coordinates. Always center.
13. stimset_dict_rfmp_sin1(xn,max_rf)
        Calls functions 11 and 12 to create and return splist.
"""
import os
import sys
import math
import warnings

import concurrent.futures
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('../..')
from src.rf_mapping.image import make_box
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.files import delete_all_npy_files
from src.rf_mapping.spatial import (xn_to_center_rf,
                                    calculate_center,
                                    get_rf_sizes,)
from src.rf_mapping.net import get_truncated_model
import src.rf_mapping.constants as c
from src.rf_mapping.stimulus import *


#######################################.#######################################
#                                                                             #
#                                  IMPORT JIT                                 #
#                                                                             #
#  Numba may not work with the lastest version of NumPy. In that case, a      #
#  do-nothing decorator also named jit is used.                               #
#                                                                             #
###############################################################################
try:
    from numba import jit
except:
    warnings.warn("grating.py cannot import Numba. Grating are generated without jit.")
    def jit(func):
        """
        A do-nothing decorator in place of the actual njit in case that Python
        cannot import Numba.
        """
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper


#######################################.#######################################
#                                                                             #
#                                  STIMFR_SINE                                #
#                                                                             #
###############################################################################
@jit
def stimfr_sine(xn,yn,x0,y0,diam,sf,phase,theta,bgval,contrast):
    """
    Return a numpy array that has a circular patch of sinusoidal grating
    in the range -1.0 to 1.0.

    xn    - (int) horizontal width of returned array
    yn    - (int) vertical height of returned array
    x0    - (float) horizontal offset of center (pix)
    y0    - (float) vertical offset of center (pix)
    diam  - (float) diameter (pix)
    sf    - (float) cycles/pix
    phase - (float) spatial phase (deg)
    theta - (float) orientation (deg)
    bgval - (float) background value [-1.0 ... 1.0]
    contrast - (float) [0.0 ... 1.0]
    """
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
            dy2 = (j-(cyi+y0))*(j-(cyi+y0))
            if (dx2+dy2 < rad2):
                s[yn-1-j,i] = contrast * math.cos(twopif*(nx*x + ny*y) - ph)
    
    return s


# if __name__ == "__main__":
#     def binarize_map(s, thres=0):
#         s[s > thres] = 1
#         s[s < thres] = 0
#     s1 = stimfr_sine(200,150,0,0,100,0.2,0,45,0,1)
#     s2 = stimfr_sine(200,150,0,0,100,0.2,135,47,0,1)
#     binarize_map(s1)
#     binarize_map(s2)
#     plt.imshow(s1 + s2, cmap='gray')
#     plt.show()


if __name__ == "__main__":
    spatial_freqs = np.array([ 0.02, 0.04, 0.08, 0.16, 0.32 ])
    phase = np.array([ 0.0, 90.0, 180.0, 270.0 ])
    orilist = np.arange(0.0, 180.0, 22.5)
    
    plt.figure(figsize=(50,10))
    plot_size = (100, 500)
    s = np.zeros((plot_size))
    for i, sf in enumerate(spatial_freqs):
        s += stimfr_sine(plot_size[1], plot_size[0],-200+100*i,0,100,sf,0,0,0,1)
    plt.imshow(s, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(40,10))
    plot_size = (100, 400)
    s = np.zeros((plot_size))
    for i, p in enumerate(phase):
        s += stimfr_sine(plot_size[1], plot_size[0],-150+100*i,0,90,0.02,p,0,0,1)
    plt.imshow(s, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(80,10))
    plot_size = (100, 800)
    s = np.zeros((plot_size))
    for i, ori in enumerate(orilist):
        s += stimfr_sine(plot_size[1], plot_size[0],-350+100*i,0,90,0.02,0,ori,0,1)
    plt.imshow(s, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
    plt.show()

        
    # for p in phase:
    #     for ori in orilist:
    #         s = stimfr_sine(200,200,0,0,190,sf,p,ori,0,1)
    #         plt.imshow(s, cmap='gray', vmin=-1, vmax=1)
    #         plt.axis('off')
    #         plt.title(f"{sf}, {p}, {ori}")
    #         plt.show()


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
    """
    xn    - width and height of stimulus (pix)
    off0  - offset for centering along the 1st axis (rows)
    off1  - offset for centering along the 2nd axis (columns)
    b0    - batch start index
    bn    - batch size
    sset  - name of stimulus set, e.g., 'sine_1', 'sine_2'
    """
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


######################################.#######################################
#                                                                             #
#                            STIMSET_SINE_SIZE_SINGLE                         #
#                                                                             #
#  Return a set of 1 stimulus, at the specified indices.                      #
#                                                                             #
###############################################################################
def stimset_sine_size_single(xn,off0,off1,isf,ith,isz,iph,sset):
    """
    xn    - width and height of stimulus (pix)
    off0  - offset for centering along the 1st axis (rows)
    off1  - offset for centering along the 2nd axis (columns)
    isf   - index into SF dimension
    ith   - index into theta dimension
    isz   - index into size dimension
    iph   - index into phase dimension
    sset - name of stimulus set, e.g., 'sine_1', 'sine_2'
    """
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
    """
    xn    - width and height of stimulus (pix)
    bgval - value of input image
    """
    nstim = 1
    d = np.empty((nstim,3,xn,xn), dtype='float32')

    d[0][0] = np.full((xn,xn), bgval, dtype='float32')
    d[0][1] = d[0][0]
    d[0][2] = d[0][0]
    
    return d


#######################################.#######################################
#                                                                             #
#                             GET_FOURIER_HARMONIC                            #
#                                                                             #
#  WYETH - this was copied from C-code, needs to be vectorized                #
#        - also could add frequency (wavelength) parameter                    #
#  TONY  - vectorize on Sep 13, 2022                                          #
#                                                                             #
###############################################################################
@jit
def get_fourier_harmonic(d,order):
    """
    d     - 1D data array
    order - index of fourier component to compute, 1=f1, 2=f2, etc, 0=DC
    """
    if (order == 0):
        return d.sum()/len(d), 0.0   # Return average of array
    
    # Unvectorized version:
    # todd = teven = 0.0
    # n = len(d)
    # for i in range(n):
    #     x = i/(n/order) * 2.0*math.pi
    #     todd  += 2.0 * math.sin(x) * d[i]
    #     teven += 2.0 * math.cos(x) * d[i]
    # teven /= n
    # todd  /= n
    
    # Vectorized version:
    n = len(d)
    x = np.arange(n) / (n/order) * 2 * math.pi
    todd  = np.sum(2 * np.sin(x) * d) / n
    teven = np.sum(2 * np.cos(x) * d) / n

    ampl = math.sqrt(todd*todd + teven*teven)
    theta = math.atan2(todd,teven) * 180.0/math.pi
    return ampl,theta
    

if __name__ == "__main__":
    a = np.arange(0,2.0*math.pi,math.pi/4)
    sa = np.sin(a)
    plt.plot(sa)
    plt.show()
    
    for order in range(5):
        amp,thet = get_fourier_harmonic(sa,order)
        print(order, amp, thet)


#######################################.#######################################
#                                                                             #
#                             STIMSET_STIM_GET_SIN                            #
#                                                                             #
#  Return the stimulus image for paradigm sin1.                               #
#                                                                             #
###############################################################################
def stimset_stim_get_sin(d):
    """
    d - dictionary of stimulus parameters
    """
    s = stimfr_sine(d['xn'],d['yn'],d['x0'],d['y0'],d['size'],d['sf'],
                    d['ph'],d['theta'],d['bgval'],d['contrast'])
    return s


#######################################.#######################################
#                                                                             #
#                             STIMSET_STIM_SHOW_SIN                           #
#                                                                             #
#  Show a plot of the stimulus described by dictionary 'd' for sin1.          #
#                                                                             #
###############################################################################
def stimset_show_stim_sin(d):
    s = stimset_stim_get_sin(d)
    plt.imshow(s)
    plt.show()


#######################################.#######################################
#                                                                             #
#                                SINMAP_RUN_01b                               #
#                                                                             #
###############################################################################
def sinmap_run_01b(splist, truncated_model, num_units, batch_size=100, _debug=False):
    """
    Presents bars and returns the center responses in array of dimension:
    [num_stim, num_units].

    Parameters
    ----------
    splist     - stimulus parameter list.\n
    truncated_model - neural network up to the layer of interest.\n
    num_units  - number of units/channels.\n
    batch_size - how many stimuli to present at once.\n
    _debug     - if true, reduce the number of stimuli and plot them.\n
    """
    stim_i = 0
    num_stim = len(splist)
    xn = splist[0]['xn']
    yn = splist[0]['yn']
    center_responses = np.zeros((num_stim, num_units))

    while (stim_i < num_stim):
        if _debug and stim_i > 200:
            break
        print_progress(f"Presenting {stim_i}/{num_stim} stimuli...")
        real_batch_size = min(batch_size, num_stim-stim_i)
        stim_batch = np.zeros((real_batch_size, 3, yn, xn))

        # Create a batch of bars.
        for i in range(real_batch_size):
            params = splist[stim_i + i]
            new_stim= stimfr_sine(params['xn'], params['yn'],
                                  params['x0'], params['y0'],
                                  params['size'], params['sf'], params['ph'],
                                  params['theta'], 
                                  params['bgval'], params['contrast'])
            # Replicate new bar t o all color channel.
            stim_batch[i, 0] = new_stim
            stim_batch[i, 1] = new_stim
            stim_batch[i, 2] = new_stim

        # Present the patch of bars to the truncated model.
        with torch.no_grad():  # turn off gradient calculations for speed.
            y = truncated_model(torch.tensor(stim_batch).type('torch.FloatTensor').to(c.DEVICE))
        yc, xc = calculate_center(y.shape[-2:])
        center_responses[stim_i:stim_i+real_batch_size, :] = y[:, :, yc, xc].detach().cpu().numpy()
        stim_i += real_batch_size

    return center_responses


#######################################.#######################################
#                                                                             #
#                                MAKE_STIMMAPS                                #
#                                                                             #
###############################################################################
def make_stimmaps(splist, center_responses, unit_i, _debug=False,
                  num_stim=500, response_thr=0.5, stim_thr=0.2):
    """
    Parameters
    ----------
    splist           - stimulus parameter list.\n
    center_responses - responses of center unit in [stim_i, unit_i] format.\n
    unit_i           - unit's index.\n
    response_thr     - stimulus w/ a reponse below response_thr * rmax will be
                       excluded.\n
    stim_thr         - bar pixels w/ a value below stim_thr will be excluded.\n
    _debug           - if true, print ranking info.\n

    Returns
    -------
    The weighted_max_map, weighted_min_map, non_overlap_max_map, and
    non_overlap_min_map of one unit.
    """
    print(f"{unit_i} done.")

    xn = splist[0]['xn']
    yn = splist[0]['yn']
    
    weighted_max_map = np.zeros((yn, xn))
    weighted_min_map = np.zeros((yn, xn))
    non_overlap_max_map = np.zeros((yn, xn))
    non_overlap_min_map = np.zeros((yn, xn))

    isort = np.argsort(center_responses[:, unit_i])  # Ascending
    r_max = center_responses[:, unit_i].max()
    r_min = center_responses[:, unit_i].min()
    r_range = max(r_max - r_min, 1)

    # Initialize bar counts
    num_weighted_max_bars = 0
    num_weighted_min_bars = 0
    num_non_overlap_max_bars = 0
    num_non_overlap_min_bars = 0

    if _debug:
        print(f"unit {unit_i}: r_max: {r_max:7.2f}, max bar idx: {isort[::-1][:5]}")

    for max_stim_i in isort[::-1]:
        response = center_responses[max_stim_i, unit_i]
        params = splist[max_stim_i]
        # Note that the background color are set to 0, while the foreground
        # values are always positive.
        new_stim = stimfr_sine(params['xn'], params['yn'],
                               params['x0'], params['y0'],
                               params['size'], params['sf'], params['ph'],
                               params['theta'], 
                               params['bgval'], params['contrast'])
        if response > max(r_max * response_thr, 0):
            has_included = add_non_overlap_map(new_stim, non_overlap_max_map, stim_thr)
            add_weighted_map(new_stim, weighted_max_map, response)
            # counts the number of bars in each map
            num_weighted_max_bars += 1
            if has_included:
                num_non_overlap_max_bars += 1
        else:
            break

    for min_stim_i in isort:
        response = center_responses[min_stim_i, unit_i]
        params = splist[min_stim_i]
        new_stim = stimfr_sine(params['xn'], params['yn'],
                               params['x0'], params['y0'],
                               params['size'], params['sf'], params['ph'],
                               params['theta'], 
                               params['bgval'], params['contrast'])
        if response < min(r_min * response_thr, 0):
            has_included = add_non_overlap_map(new_stim, non_overlap_min_map, stim_thr)
            add_weighted_map(new_stim, weighted_min_map, (r_max - response)/r_range)
            # counts the number of bars in each map
            num_weighted_min_bars += 1
            if has_included:
                num_non_overlap_min_bars += 1
        else:
            break

    return weighted_max_map, weighted_min_map,\
           non_overlap_max_map, non_overlap_min_map,\
           num_weighted_max_bars, num_weighted_min_bars,\
           num_non_overlap_max_bars, num_non_overlap_min_bars


#######################################.#######################################
#                                                                             #
#                             STIM_DAPP_SIN_XYO_BW                            #
#                                                                             #
#  Create dictionary entries for bar stimuli varying in these parameters:     #
#    x location                                                               #
#    y location                                                               #
#    orientation                                                              #
#    luminance contrast polarity                                              #
#                                                                             #
#  The input parameters specify the range of x-values to use, and these       #
#  are replicated for the y-range as well.                                    #
#  'dori' to use for varying orientation.                                     #
#                                                                             #
#  The other bar parameters are held fixed:  length, width, anti-aliasing.    #
#                                                                             #
###############################################################################
def stim_dapp_sin_xyo_bw(splist,xn,xlist,orilist,diam,sf):
    #
    #  splist  - stimulus parameter list - APPEND TO THIS LIST
    #  xn      - horizontal and vertical image size
    #  xlist   - list of x-coordinates (pix)
    #  orilist - list of orientation values (degr)
    #  diam    - grating diameter / size (pix)
    #  sf      - spatial frequency
    #
    
    yn = xn        # Assuming image is square
    ylist = xlist  # Use same coordinates for y grid locations
    #print(xlist)   # Testing
    
    con   =  1.0  # Contrast
    bgval =  0.0  # Background luminance
    
    phase = np.array([ 0.0, 90.0, 180.0, 270.0 ])
    
    nstim = len(xlist) * len(ylist) * len(orilist) * 2
    # print("  Creating ", nstim, " stimulus dictionary entries.")
    
    for i in xlist:
        for j in ylist:
            for o in orilist:
                for ph in phase:
                    tp = {"xn":xn, "yn":yn, "x0":i, "y0":j,
                          "size":diam, "sf":sf, "ph":ph,
                          "theta":o, "bgval":bgval, "contrast":con}
                    splist.append(tp)


#######################################.#######################################
#                                                                             #
#                            STIMSET_DICT_RFMP_SIN1                           #
#                                                                             #
#  Return the stimulus parameter dictionary with the appropriate entries      #
#  for the entire stimulus set for RF mapping paradigm "sin1".                #
#                                                                             #
###############################################################################
def stimset_dict_rfmp_sin1(xn,max_rf):
    """
    xn     - stimulus image size (pix)
    max_rf - maximum RF size (pix)
    """
    splist = []  # List of dictionary entries, one per stimulus image
    
    #  There are 4 diams
    diam = np.array([48/64 * max_rf,    #  Array of grating diameters
                     24/64 * max_rf,
                     12/64 * max_rf,
                      6/64 * max_rf])

    sf = np.array([ 0.02, 0.04, 0.08, 0.16, 0.32 ])
    
    #  There are 16 orientations, even spaced around 360 deg starting at 0 deg
    orilist = np.arange(0.0, 180.0, 22.5)
    
    for d in diam:
        xlist = stimset_gridx_map(max_rf,d)
        for s in sf:
            stim_dapp_sin_xyo_bw(splist,xn,xlist,orilist,d,s)
    
    # print("  Length of stimulus parameter list:",len(splist))
    
    return splist


#######################################.#######################################
#                                                                             #
#                                 SIN1_RUN_01b                                #
#                                                                             #
###############################################################################
def sin1_run_01b(model, model_name, result_dir, _debug=False, batch_size=10,
                 response_thr=0.5, conv_i_to_run=None):
    """
    Map the RF of all conv layers in model using RF mapping paradigm sin1.
    
    Parameters
    ----------
    model      - neural network.\n
    model_name - name of neural network. Used for txt file naming.\n
    result_dir - directories to save the npy, txt, and pdf files.\n
    _debug     - if true, only run the first 10 units of every layer.
    """
    xn_list = xn_to_center_rf(model, image_size=(999, 999))  # Get the xn just big enough.
    unit_counter = ConvUnitCounter(model)
    layer_indices, nums_units = unit_counter.count()
    _, max_rfs = get_rf_sizes(model, (999, 999), layer_type=nn.Conv2d)
    # Note that the image sizes above are set to (999, 999). This change
    # was made so that layers with RF larger than (227, 227) could be properly
    # centered during bar mapping.

    # Set paths
    tb1_path = os.path.join(result_dir, f"{model_name}_sin1_tb1.txt")
    tb20_path = os.path.join(result_dir, f"{model_name}_sin1_tb20.txt")
    tb100_path = os.path.join(result_dir, f"{model_name}_sin1_tb100.txt")
    weighted_counts_path = os.path.join(result_dir, f"{model_name}_sin1_weighted_counts.txt")
    non_overlap_counts_path = os.path.join(result_dir, f"{model_name}_sin1_non_overlap_counts.txt")
    
    # Delete previous files
    delete_all_npy_files(result_dir)
    if os.path.exists(tb1_path):
        os.remove(tb1_path)
    if os.path.exists(tb20_path):
        os.remove(tb20_path)
    if os.path.exists(tb100_path):
        os.remove(tb100_path)
    if os.path.exists(weighted_counts_path):
        os.remove(weighted_counts_path)
    if os.path.exists(non_overlap_counts_path):
        os.remove(non_overlap_counts_path)
    
    for conv_i in range(len(layer_indices)):
        layer_name = f"conv{conv_i + 1}"
        print(f"\n{layer_name}\n")
        # Get layer-specific info
        xn = xn_list[conv_i]
        layer_idx = layer_indices[conv_i]
        num_units = nums_units[conv_i]
        max_rf = max_rfs[conv_i][0]
        splist = stimset_dict_rfmp_sin1(xn, max_rf)

        # Array initializations
        weighted_max_maps = np.zeros((num_units, max_rf, max_rf))
        weighted_min_maps = np.zeros((num_units, max_rf, max_rf))
        non_overlap_max_maps = np.zeros((num_units, max_rf, max_rf))
        non_overlap_min_maps = np.zeros((num_units, max_rf, max_rf))
        padding = (xn - max_rf) // 2

        # Present all bars to the model
        truncated_model = get_truncated_model(model, layer_idx)
        center_responses = sinmap_run_01b(splist, truncated_model, num_units,
                                          batch_size=batch_size, _debug=False)

        # Append to txt files that summarize the top and bottom bars.
        summarize_TB1(splist, center_responses, layer_name, tb1_path)
        summarize_TBn(splist, center_responses, layer_name, tb20_path, top_n=20)
        summarize_TBn(splist, center_responses, layer_name, tb100_path, top_n=100)
        
        # This block of code contained in the following while-loop used to be
        # a bottleneck of the program because it is all computed by a single
        # CPU core. Improvement by multiprocessing was implemented on August
        # 15, 2022 to solve the problem.
        batch_size = os.cpu_count() // 2
        unit_i = 0
        while (unit_i < num_units):
            if _debug and unit_i >= 20:
                break
            real_batch_size = min(batch_size, num_units - unit_i)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(make_stimmaps,
                                       [splist for _ in range(real_batch_size)],
                                       [center_responses for _ in range(real_batch_size)],
                                       [i for i in np.arange(unit_i, unit_i + real_batch_size)],
                                       [_debug for _ in range(real_batch_size)],
                                       [None for _ in range(real_batch_size)],
                                       [response_thr for _ in range(real_batch_size)]
                                       )
            # Crop and save maps to layer-level array
            for result_i, result in enumerate(results):
                weighted_max_maps[unit_i + result_i] = result[0][padding:padding+max_rf, padding:padding+max_rf]
                weighted_min_maps[unit_i + result_i] = result[1][padding:padding+max_rf, padding:padding+max_rf]
                non_overlap_max_maps[unit_i + result_i] = result[2][padding:padding+max_rf, padding:padding+max_rf]
                non_overlap_min_maps[unit_i + result_i] = result[3][padding:padding+max_rf, padding:padding+max_rf]
                # Record the number of bars used in each map (append to txt files).
                record_stim_counts(weighted_counts_path, layer_name, unit_i + result_i,
                                  result[4], result[5])
                record_stim_counts(non_overlap_counts_path, layer_name, unit_i + result_i,
                                  result[6], result[7])
            unit_i += real_batch_size

        # Save the maps of all units.
        weighte_max_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_max_sinemaps.npy")
        weighted_min_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_min_sinemaps.npy")
        non_overlap_max_maps_path = os.path.join(result_dir, f"{layer_name}_non_overlap_max_sinemaps.npy")
        non_overlap_min_maps_path = os.path.join(result_dir, f"{layer_name}_non_overlap_min_sinemaps.npy")
        np.save(weighte_max_maps_path, weighted_max_maps)
        np.save(weighted_min_maps_path, weighted_min_maps)
        np.save(non_overlap_max_maps_path, non_overlap_max_maps)
        np.save(non_overlap_min_maps_path, non_overlap_min_maps)
        
        # Save the splist in a text file.
        splist_path = os.path.join(result_dir, f"{layer_name}_splist.txt")
        if os.path.exists(splist_path):
            os.remove(splist_path)
        record_splist(splist_path, splist)
        
        # Save the indicies and responses of top and bottom 5000 stimuli.
        top_n = min(5000, len(splist)//2)
        max_center_reponses_path = os.path.join(result_dir, f"{layer_name}_top{top_n}_responses.txt")
        min_center_reponses_path = os.path.join(result_dir, f"{layer_name}_bot{top_n}_responses.txt")
        if os.path.exists(max_center_reponses_path):
            os.remove(max_center_reponses_path)
        if os.path.exists(min_center_reponses_path):
            os.remove(min_center_reponses_path)
        record_center_responses(max_center_reponses_path, center_responses, top_n, is_top=True)
        record_center_responses(min_center_reponses_path, center_responses, top_n, is_top=False)

        # Make pdf for the layer.
        weighted_pdf_path = os.path.join(result_dir, f"{layer_name}_weighted_sinemaps.pdf")
        make_map_pdf(weighted_max_maps, weighted_min_maps, weighted_pdf_path)
        non_overlap_pdf_path = os.path.join(result_dir, f"{layer_name}_non_overlap_sinemaps.pdf")
        make_map_pdf(non_overlap_max_maps, non_overlap_min_maps, non_overlap_pdf_path)


#######################################.#######################################
#                                                                             #
#                              MAKE_SIN1_GRID_PDF                             #
#                                                                             #
###############################################################################
def make_sin1_grid_pdf(pdf_path, model):
    xn_list = xn_to_center_rf(model, image_size=(999,999))  # Get the xn just big enough.
    img_size = 227
    layer_indices, max_rfs = get_rf_sizes(model, (img_size, img_size), layer_type=nn.Conv2d)
    num_layers = len(max_rfs)

    # Array of bar lengths
    size_ratios = np.array([48/64,
                            24/64,
                            12/64,
                             6/64])
    size_ratio_str = np.array(['48/64',
                               '24/64',
                               '12/64',
                                '6/64'])

    spatial_freqs = np.array([ 0.02, 0.04, 0.08, 0.16, 0.32 ])
    # phase = np.array([ 0.0, 90.0, 180.0, 270.0 ])
    # orilist = np.arange(0.0, 180.0, 22.5)

    with PdfPages(pdf_path) as pdf:
        for size_i, size_ratio in enumerate(size_ratios):
            for sf in spatial_freqs:
                plt.figure(figsize=(4*num_layers, 5))
                plt.suptitle(f"Size = {size_ratio_str[size_i]} M, Spatial freq = {sf}", fontsize=24)

                for conv_i, max_rf in enumerate(max_rfs):
                    layer_name = f"conv{conv_i + 1}"
                    layer_index = layer_indices[conv_i]
                    # Get layer-specific info
                    xn = xn_list[conv_i]
                    max_rf = max_rf[0]

                    # Set bar parameters
                    size = size_ratio * max_rf
                    xlist = stimset_gridx_map(max_rf,size)

                    # Plot the stimulus
                    stim = stimfr_sine(xn,xn,0,0,size,sf,22.5,0,0,1)
                    plt.subplot(1, num_layers, conv_i+1)
                    plt.imshow(stim, cmap='gray', vmin=-1, vmax=1)
                    plt.title(f"{layer_name}\n(idx={layer_index}, maxRF={max_rf}, xn={xn})")

                    # Plot the bar centers (i.e., the "grids").
                    for y0 in xlist:
                        for x0 in xlist:
                            plt.plot(y0+xn/2, x0+xn/2, 'k.', alpha=0.4)

                    # Highlight maximum RF
                    padding = (xn - max_rf)//2
                    rect = make_box((padding-1, padding-1, padding+max_rf-1, padding+max_rf-1), linewidth=1)
                    ax = plt.gca()
                    ax.add_patch(rect)
                    # plt.axis('off')
                    # ax.invert_yaxis()
        
                pdf.savefig()
                plt.show()
                plt.close()


# Generate a RFMP-Sin1 grid pdf for AlexNet
if __name__ == "__main__":
    # model = models.resnet18()
    # model_name = 'resnet18'
    model = models.alexnet()
    model_name = 'alexnet'
    pdf_path = os.path.join(c.REPO_DIR,'results','rfmp_sin1','mapping', 'test',
                            f'{model_name}_test_grid.pdf')
    # make_sin1_grid_pdf(pdf_path, model)
