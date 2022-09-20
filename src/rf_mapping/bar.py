"""
Code for generating bar stimuli.

Note: The y-axis points downward. (Negative sign)

July 15, 2022
"""
import os
import sys
import math
import warnings

import concurrent.futures
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
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
#                                  IMPORT JIT                                #
#                                                                             #
#  Numba may not work with the lastest version of NumPy. In that case, a      #
#  do-nothing decorator also named jit is used.                              #
#                                                                             #
###############################################################################
try:
    from numba import jit
except:
    warnings.warn("bar.py cannot import Numba. Bars are generated without njit.")
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
#                                   ROTATE                                    #
#                                                                             #
###############################################################################
@jit
def rotate(dx, dy, theta_deg):
    """
    Applies the rotation matrix:
        [dx, dy] * [[a, b], [c, d]]  = [dx_r, dy_r]
    
    To undo rotation, apply again with negative theta_deg.
    Modified from Dr. Wyeth Bair's d06_mrf.py
    """
    thetar = theta_deg * math.pi / 180.0
    rot_a = math.cos(thetar); rot_b = math.sin(thetar)
    rot_c = math.sin(thetar); rot_d = -math.cos(thetar)
    # Usually, the negative sign appears in rot_c instead of rot_d, but the
    # y-axis points downward in our case.
    dx_r = rot_a*dx + rot_c*dy
    dy_r = rot_b*dx + rot_d*dy
    return dx_r, dy_r


#######################################.#######################################
#                                                                             #
#                              STIMFR_BAR_COLOR                               #
#                                                                             #
###############################################################################
@jit  # sped up by 182x
def stimfr_bar(xn, yn, x0, y0, theta, blen, bwid, aa, fgval, bgval):
    """
    Parameters
    ----------
    xn    - (int) horizontal width of returned array\n
    yn    - (int) vertical height of returned array\n
    x0    - (float) horizontal offset of center (pix)\n
    y0    - (float) vertical offset of center (pix)\n
    theta - (float) orientation (pix)\n
    blen  - (float) length of bar (pix)\n
    bwid  - (float) width of bar (pix)\n
    aa    - (float) length scale for anti-aliasing (pix)\n
    fgval - (float) bar luminance [0..1]\n
    bgval - (float) background luminance [0..1]
    
    Returns
    -------
    Return a numpy array that has a bar on a background:
    """
    s = np.full((yn,xn), bgval, dtype='float32')  # Fill w/ BG value
    dval = fgval - bgval  # Luminance difference
  
    # Maximum extent of unrotated bar corners in zero-centered frame:
    dx = bwid/2.0
    dy = blen/2.0

    # Rotate top-left corner from (-dx, dy) to (dx1, dy1)
    dx1, dy1 = rotate(-dx, dy, theta)
    
    # Rotate top-right corner from (dx, dy) to (dx2, dy2)
    dx2, dy2 = rotate(dx, dy, theta)

    # Maximum extent of rotated bar corners in zero-centered frame:
    maxx = aa + max(abs(dx1), abs(dx2))
    maxy = aa + max(abs(dy1), abs(dy2))
    
    # Center of stimulus field
    xc = (xn-1.0)/2.0
    yc = (yn-1.0)/2.0

    # Define the 4 corners a box that contains the rotated bar.
    bar_left_i   = round(xc + x0 - maxx)
    bar_right_i  = round(xc + x0 + maxx) + 1
    bar_top_i    = round(yc + y0 - maxy)
    bar_bottom_i = round(yc + y0 + maxy) + 1

    bar_left_i   = clip(bar_left_i  , 0, xn)
    bar_right_i  = clip(bar_right_i , 0, xn)
    bar_top_i    = clip(bar_top_i   , 0, yn)
    bar_bottom_i = clip(bar_bottom_i, 0, yn)

    for i in range(bar_left_i, bar_right_i):  # for i in range(0,xn):
        xx = i - xc - x0  # relative to bar center
        for j in range (bar_top_i, bar_bottom_i):  # for j in range (0,yn):
            yy = j - yc - y0  # relative to bar center
            x, y = rotate(xx, yy, -theta)  # rotate back

            # Compute distance from bar edge, 'db'
            if x > 0.0:
                dbx = bwid/2 - x  # +/- indicates inside/outside
            else:
                dbx = x + bwid/2

            if y > 0.0:
                dby = blen/2 - y  # +/- indicates inside/outside
            else:
                dby = y + blen/2

            if dbx < 0.0:  # x outside
                if dby < 0.0:
                    db = -math.sqrt(dbx*dbx + dby*dby)  # Both outside
                else:
                    db = dbx
            else:  # x inside
                if dby < 0.0:  # y outside
                    db = dby
                else:  # Both inside - take the smallest distance
                    if dby < dbx:
                        db = dby
                    else:
                        db = dbx

            if aa > 0.0:
                if db > aa:
                    f = 1.0  # This point is inside the bar
                elif db < -aa:
                    f = 0.0  # This point is outside the bar
                else:  # Use sinusoidal sigmoid
                    f = 0.5 + 0.5*math.sin(db/aa * 0.25*math.pi)
            else:
                if db >= 0.0:
                    f = 1.0  # inside
                else:
                    f = 0.0  # outside

            s[j, i] += f * dval  # add a fraction 'f' of the 'dval'

    return s


# Test
# if __name__ == "__main__":
#     xn = 200
#     yn = 300
#     # rotate test
#     for theta in np.linspace(0, 180, 8):
#         bar = stimfr_bar(xn, yn, 0, 0, theta, 100, 50, 1, 1, -1)
#         plt.imshow(bar, cmap='gray')
#         plt.title(f"theta = {theta:.2f}")
#         plt.show()

#     # move from top to bottom
#     for y0 in np.linspace(-yn//2, yn//2, 5):
#         bar = stimfr_bar(xn, yn, 0, y0, 45, 80, 30, 2, 1, 0)
#         plt.imshow(bar, cmap='gray')
#         plt.title(f"y = {y0:.2f}")
#         plt.show()
        

#######################################.#######################################
#                                                                             #
#                               STIMFR_BAR_COLOR                              #
#                                                                             #
###############################################################################
# njit slowed down by 4x
def stimfr_bar_color(xn,yn,x0,y0,theta,blen,bwid,aa,r1,g1,b1,r0,g0,b0):
    """
    Parameters
    ----------
    xn    - (int) horizontal width of returned array\n
    yn    - (int) vertical height of returned array\n
    x0    - (float) horizontal offset of center (pix)\n
    y0    - (float) vertical offset of center (pix)\n
    theta - (float) orientation (pix)\n
    blen  - (float) length of bar (pix)\n
    bwid  - (float) width of bar (pix)\n
    laa   - (float) length scale for anti-aliasing (pix)\n
    r1,g1,b1 - bar color\n
    r0,g0,b0 - background color\n
    
    Returns
    -------
    Return a numpy array (3, yn, xn) that has a bar on a background.
    """
    s = np.empty((3, yn,xn), dtype='float32')
    s[0] = stimfr_bar(xn, yn, x0, y0, theta, blen, bwid, aa, r1, r0)
    s[1] = stimfr_bar(xn, yn, x0, y0, theta, blen, bwid, aa, g1, g0)
    s[2] = stimfr_bar(xn, yn, x0, y0, theta, blen, bwid, aa, b1, b0)
    return s


# Test
if __name__ == '__main__':
    # xn = 20
    # yn = 20
    # colors = {'red':     (1, 0, 0),
    #           'green':   (0, 1, 0),
    #           'blue':    (0, 0, 1),
    #           'yellow':  (1, 1, 0),
    #           'magenta': (1, 0, 1),
    #           'cyan':    (0, 1, 1),
    #           'white':   (1, 1, 1),
    #           'black':   (0, 0, 0)}
    # for color_name, (r1, g1, b1) in colors.items():
    #     bar = stimfr_bar_color(xn, yn, 0, 0, 40, 8, 3, 0.5,
    #                            r1, g1, b1,
    #                            0.5, 0.5, 0.5)

    bar = stimfr_bar_color(15, 15, -4.125, -4.125, 20, 8.25, 4.125, 0.5,
                            1, -1, 1, -1, 1, -1)
    plt.imshow(np.transpose(bar, (1,2,0)))
    plt.show()

    bar = stimfr_bar_color(15, 15, -4.125, -4.125, 20, 8.25, 4.125, 0.5,
                            1, -1, -1, -1, 1, 1)
    plt.imshow(np.transpose(bar, (1,2,0)))
    plt.show()

    bar = stimfr_bar_color(15, 15, -4.125, -4.125, 20, 8.25, 4.125, 0.5,
                           -1, 1, -1, -1, -1, 1)
    plt.imshow(np.transpose(bar, (1,2,0)))
    plt.show()


#######################################.#######################################
#                                                                             #
#                             STIM_DAPP_BAR_XYO_BW                            #
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
def stim_dapp_bar_xyo_bw(splist,xn,xlist,orilist,blen,bwid,aa):
    """
    Parameters
    ----------
    splist  - stimulus parameter list - APPEND TO THIS LIST\n
    xn      - horizontal and vertical image size\n
    xlist   - list of x-coordinates (pix)\n
    orilist - list of orientation values (degr)\n
    blen    - Length of bar (pix)\n
    bwid    - Width of bar (pix)\n
    aa      - Anti-aliasing space constant (pix)\n
    """
    yn = xn        # Assuming image is square
    ylist = xlist  # Use same coordinates for y grid locations
    
    fgval =  1.0  # Foreground luminance
    bgval = -1.0  # Background luminance
    
    # nstim = len(xlist) * len(ylist) * len(orilist) * 2
    # print("  Creating ", nstim, " stimulus dictionary entries.")
    
    for i in xlist:
        for j in ylist:
            for o in orilist:
                tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen,
                      "wid":bwid, "aa":aa, "fgval":fgval, "bgval":bgval}
                splist.append(tp)
                
                # Now swap 'bgval' and 'fgval' to make opposite contrast
                tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen,
                      "wid":bwid, "aa":aa, "fgval":bgval, "bgval":fgval}
                splist.append(tp)


#######################################.#######################################
#                                                                             #
#                            STIM_DAPP_BAR_XYO_RGB7O                          #
#                                                                             #
#  For each x, y and orientation in the lists,                                #
#  creates dictionary entries for 14 color/BW conditions, 7 pairs in both     #
#  ordering of foreground an background.                                      #
#                                                                             #
###############################################################################
def stim_dapp_bar_xyo_rgb7o(splist,xn,xlist,orilist,blen,bwid,aa):
    """
    Parameters
    ----------
    splist  - stimulus parameter list - APPEND TO THIS LIST\n
    xn      - horizontal and vertical image size\n
    xlist   - list of x-coordinates (pix)\n
    orilist - list of orientation values (degr)\n
    blen    - Length of bar (pix)\n
    bwid    - Width of bar (pix)\n
    aa      - Anti-aliasing space constant (pix)
    """
    yn = xn        # Assuming image is square
    ylist = xlist  # Use same coordinates for y grid locations
    #print(xlist)   # Testing
    
    a0 = -1.0  # Amplitude low
    a1 =  1.0  # Amplitude high
    
    # nstim = len(xlist) * len(ylist) * len(orilist) * 2
    # print("  Creating ", nstim, " stimulus dictionary entries.")

    for i in xlist:
        for j in ylist:
            for o in orilist:
                # 111 v 000          B&W
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a1, "g1":a1, "b1":a1,"r0":a0, "g0":a0, "b0":a0}
                # splist.append(tp)
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a0, "g1":a0, "b1":a0,"r0":a1, "g0":a1, "b0":a1}
                # splist.append(tp)
                
                # # 100 v 010      red-green
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a1, "g1":a0, "b1":a0,"r0":a0, "g0":a1, "b0":a0}
                # splist.append(tp)
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a0, "g1":a1, "b1":a0,"r0":a1, "g0":a0, "b0":a0}
                # splist.append(tp)
                
                # # 100 v 001      red-blue
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a1, "g1":a0, "b1":a0,"r0":a0, "g0":a0, "b0":a1}
                # splist.append(tp)
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a0, "g1":a0, "b1":a1,"r0":a1, "g0":a0, "b0":a0}
                # splist.append(tp)
                
                # # 010 v 001      green-blue
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a0, "g1":a1, "b1":a0,"r0":a0, "g0":a0, "b0":a1}
                # splist.append(tp)
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a0, "g1":a0, "b1":a1,"r0":a0, "g0":a1, "b0":a0}
                # splist.append(tp)
                
                # # 110 v 001     yellow-blue
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a1, "g1":a1, "b1":a0,"r0":a0, "g0":a0, "b0":a1}
                # splist.append(tp)
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a0, "g1":a0, "b1":a1,"r0":a1, "g0":a1, "b0":a0}
                # splist.append(tp)
                
                # # 101 v 010    purple-green
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a1, "g1":a0, "b1":a1,"r0":a0, "g0":a1, "b0":a0}
                # splist.append(tp)
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a0, "g1":a1, "b1":a0,"r0":a1, "g0":a0, "b0":a1}
                # splist.append(tp)
                
                # # 011 v 100    cyan-red
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a0, "g1":a1, "b1":a1,"r0":a1, "g0":a0, "b0":a0}
                # splist.append(tp)
                # tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "wid":bwid,
                #       "aa":aa, "r1":a1, "g1":a0, "b1":a0,"r0":a0, "g0":a1, "b0":a1}
                # splist.append(tp)
                
                # Added on Sep 6th to replace all backgrounds with (0, 0, 0), i.e, gray.
                colors = [(a1, a1, a1),
                          (a0, a0, a0),
                          (a1, a0, a0),
                          (a0, a1, a0),
                          (a0, a0, a1),
                          (a1, a1, a0),
                          (a1, a0, a1),
                          (a0, a1, a1)]
                
                for r1, g1, b1 in colors:
                    tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o,
                          "len":blen, "wid":bwid, "aa":aa,
                          "r1":r1, "g1":g1, "b1":b1,
                          "r0":0, "g0":0, "b0":0}
                    splist.append(tp)


#######################################.#######################################
#                                                                             #
#                            STIMSET_DICT_RFMP_4A                             #
#                                                                             #
#  Return the stimulus parameter dictionary with the appropriate entries      #
#  for the entire stimulus set for RF mapping paradigm "4a".                  #
#                                                                             #
###############################################################################
def stimset_dict_rfmp_4a(xn,max_rf):
    """
    Parameters
    ----------
    xn     - stimulus image size (pix)\n
    max_rf - maximum RF size (pix)\n
    
    Returns
    -------
    splist - List of dictionary entries, one per stimulus image.
    """
    splist = []

    #  There are 4 bar lengths
    barlen = np.array([48/64 * max_rf,    #  Array of bar lengths
                       24/64 * max_rf,
                       12/64 * max_rf,
                        6/64 * max_rf])

    #  There are 3 aspect ratios
    arat = np.array([1/2, 1/5, 1/10])   # Array of aspect ratios

    #  There are 16 orientations, even spaced around 360 deg starting at 0 deg
    orilist = np.arange(0.0, 180.0, 22.5)

    #  This constant sets how much blurring occurs at the edges of the bars
    aa =  0.5      # Antialias distance (pix)

    for bl in barlen:
        xlist = stimset_gridx_map(max_rf,bl)
        for ar in arat:
            stim_dapp_bar_xyo_bw(splist,xn,xlist,orilist,bl,ar*bl,aa)

    # print("  Length of stimulus parameter list:",len(splist))
    return splist


# HERE IS AN EXAMPLE OF HOW TO CALL THE CODE ABOVE:
if __name__ == "__main__":
    s = stimset_dict_rfmp_4a(11,11)


#######################################.#######################################
#                                                                             #
#                            STIMSET_DICT_RFMP_4C7O                           #
#                                                                             #
#  Return the stimulus parameter dictionary with the appropriate entries      #
#  for the entire stimulus set for RF mapping paradigm "4c7o".                #
#                                                                             #
###############################################################################
def stimset_dict_rfmp_4c7o(xn,max_rf):
    """
    Parameters
    ----------
    xn     - stimulus image size (pix)\n
    max_rf - maximum RF size (pix)
    """
    splist = []  # List of dictionary entries, one per stimulus image
    
    #  There are 4 bar lengths
    barlen = np.array([48/64 * max_rf,    #  Array of bar lengths
                       24/64 * max_rf,
                       12/64 * max_rf,
                        6/64 * max_rf])
    
    #  There are 3 aspect ratios
    arat = np.array([1/2, 1/5, 1/10])   # Array of apsect ratios
    
    #  There are 16 orientations, even spaced around 360 deg starting at 0 deg
    orilist = np.arange(0.0, 180.0, 22.5)
    
    #  This constant sets how much blurring occurs at the edges of the bars
    aa =  0.5  # Antialias distance (pix)
    
    for bl in barlen:
        xlist = stimset_gridx_map(max_rf,bl)
        for ar in arat:
            stim_dapp_bar_xyo_rgb7o(splist,xn,xlist,orilist,bl,ar*bl,aa)
    
    # print("  Length of stimulus parameter list:",len(splist))
    return splist


# HERE IS AN EXAMPLE OF HOW TO CALL THE CODE ABOVE:
if __name__ == "__main__":
    s = stimset_dict_rfmp_4c7o(79,51)


#######################################.#######################################
#                                                                             #
#                                BARMAP_RUN_01b                               #
#                                                                             #
###############################################################################
def barmap_run_01b(splist, truncated_model, num_units, batch_size=100,
                   _debug=False, has_color=False):
    """
    Presents bars and returns the center responses in array of dimension:
    [num_stim, num_units].

    Parameters
    ----------
    splist     - bar stimulus parameter list.\n
    truncated_model - neural network up to the layer of interest.\n
    num_units  - number of units/channels.\n
    batch_size - how many bars to present at once.\n
    _debug     - if true, reduce the number of bars and plot them.\n
    """
    bar_i = 0
    num_stim = len(splist)
    xn = splist[0]['xn']
    yn = splist[0]['yn']
    center_responses = np.zeros((num_stim, num_units))

    while (bar_i < num_stim):
        if _debug and bar_i > 200:
            break
        print_progress(f"Presenting {bar_i}/{num_stim} stimuli...")
        real_batch_size = min(batch_size, num_stim-bar_i)
        bar_batch = np.zeros((real_batch_size, 3, yn, xn))

        # Create a batch of bars.
        for i in range(real_batch_size):
            params = splist[bar_i + i]

            if has_color:
                new_bar = stimfr_bar_color(params['xn'], params['yn'],
                                           params['x0'], params['y0'],
                                           params['theta'],
                                           params['len'], params['wid'], 
                                           params['aa'],
                                           params['r1'], params['g1'], params['b1'],
                                           params['r0'], params['g0'], params['b0'])
                bar_batch[i] = new_bar
            else:
                new_bar = stimfr_bar(params['xn'], params['yn'],
                                     params['x0'], params['y0'],
                                     params['theta'], params['len'], params['wid'], 
                                     params['aa'], params['fgval'], params['bgval'])
                # Replicate new bar to all color channel.
                bar_batch[i, 0] = new_bar
                bar_batch[i, 1] = new_bar
                bar_batch[i, 2] = new_bar

        # Present the patch of bars to the truncated model.
        with torch.no_grad():  # turn off gradient calculations for speed.
            y = truncated_model(torch.tensor(bar_batch).type('torch.FloatTensor').to(c.DEVICE))
        yc, xc = calculate_center(y.shape[-2:])
        center_responses[bar_i:bar_i+real_batch_size, :] = y[:, :, yc, xc].detach().cpu().numpy()
        bar_i += real_batch_size

    return center_responses


#######################################.#######################################
#                                                                             #
#                                 MAKE_BARMAPS                                #
#                                                                             #
###############################################################################
def make_barmaps(splist, center_responses, unit_i, _debug=False, has_color=False, 
                 num_bars=500, response_thr=0.8, stim_thr=0.2):
    """
    Parameters
    ----------
    splist           - bar stimulus parameter list.\n
    center_responses - responses of center unit in [stim_i, unit_i] format.\n
    unit_i           - unit's index.\n
    response_thr     - bar w/ a reponse below response_thr * rmax will be
                       excluded.\n
    stim_thr         - bar pixels w/ a value below stim_thr will be excluded.\n
    _debug           - if true, print ranking info.\n

    Returns
    -------
    The weighted_max_map, weighted_min_map, non_overlap_max_map, and
    non_overlap_min_map of one unit.
    
    TODO: This function is a bottleneck. Must uses multiprocessing to
          parallelize the computations on multiple cpu cores.
    """
    print(f"{unit_i} done.")

    xn = splist[0]['xn']
    yn = splist[0]['yn']
    
    if has_color:
        weighted_max_map = np.zeros((3, yn, xn))
        weighted_min_map = np.zeros((3, yn, xn))
        non_overlap_max_map = np.zeros((3, yn, xn))
        non_overlap_min_map = np.zeros((3, yn, xn))
    else:
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

    for max_bar_i in isort[::-1]:
        response = center_responses[max_bar_i, unit_i]
        params = splist[max_bar_i]
        # Note that the background color are set to 0, while the foreground
        # values are always positive.
        if has_color:
            # new_bar = stimfr_bar_color(params['xn'], params['yn'],
            #                             params['x0'], params['y0'],
            #                             params['theta'],
            #                             params['len'], params['wid'], 
            #                             params['aa'],
            #                             max(params['r1'], 0), max(params['g1'], 0), max(params['b1'], 0),
            #                             0, 0, 0)
            new_bar = stimfr_bar_color(params['xn'], params['yn'],
                                           params['x0'], params['y0'],
                                           params['theta'],
                                           params['len'], params['wid'], 
                                           params['aa'],
                                           params['r1'], params['g1'], params['b1'],
                                           params['r0'], params['g0'], params['b0'])
        else:
            new_bar = stimfr_bar(params['xn'], params['yn'],
                                params['x0'], params['y0'],
                                params['theta'], params['len'], params['wid'],
                                0.5, 1, 0)
        # if (response - r_min) > r_range * response_thr:
        if num_weighted_max_bars < num_bars:
            has_included = add_non_overlap_map(new_bar, non_overlap_max_map, stim_thr)
            add_weighted_map(new_bar, weighted_max_map, (response - r_min)/r_range)
            # counts the number of bars in each map
            num_weighted_max_bars += 1
            if has_included:
                num_non_overlap_max_bars += 1
        else:
            break

    for min_bar_i in isort:
        response = center_responses[min_bar_i, unit_i]
        params = splist[min_bar_i]
        if has_color:
            # new_bar = stimfr_bar_color(params['xn'], params['yn'],
            #                             params['x0'], params['y0'],
            #                             params['theta'],
            #                             params['len'], params['wid'], 
            #                             params['aa'],
            #                             max(params['r1'], 0), max(params['g1'], 0), max(params['b1'], 0),
            #                             0, 0, 0)
            new_bar = stimfr_bar_color(params['xn'], params['yn'],
                                           params['x0'], params['y0'],
                                           params['theta'],
                                           params['len'], params['wid'], 
                                           params['aa'],
                                           params['r1'], params['g1'], params['b1'],
                                           params['r0'], params['g0'], params['b0'])
        else:
            new_bar = stimfr_bar(params['xn'], params['yn'],
                                params['x0'], params['y0'],
                                params['theta'], params['len'], params['wid'],
                                0.5, 1, 0) 
        # if (response - r_min) < r_range * (1 - response_thr):
        if num_weighted_min_bars < num_bars:
            has_included = add_non_overlap_map(new_bar, non_overlap_min_map, stim_thr)
            add_weighted_map(new_bar, weighted_min_map, (r_max - response)/r_range)
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
#                                RFMP4a_RUN_01b                               #
#                                                                             #
###############################################################################
def rfmp4a_run_01b(model, model_name, result_dir, _debug=False, batch_size=100,
                   num_bars=500, conv_i_to_run=None):
    """
    Map the RF of all conv layers in model using RF mapping paradigm 4a.
    
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
    tb1_path = os.path.join(result_dir, f"{model_name}_rfmp4a_tb1.txt")
    tb20_path = os.path.join(result_dir, f"{model_name}_rfmp4a_tb20.txt")
    tb100_path = os.path.join(result_dir, f"{model_name}_rfmp4a_tb100.txt")
    weighted_counts_path = os.path.join(result_dir, f"{model_name}_rfmp4a_weighted_counts.txt")
    non_overlap_counts_path = os.path.join(result_dir, f"{model_name}_rfmp4a_non_overlap_counts.txt")
    
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
        # In case we would like to run only one layer...
        if conv_i is not None:
            if conv_i != conv_i_to_run:
                continue

        layer_name = f"conv{conv_i + 1}"
        print(f"\n{layer_name}\n")
        # Get layer-specific info
        xn = xn_list[conv_i]
        layer_idx = layer_indices[conv_i]
        num_units = nums_units[conv_i]
        max_rf = max_rfs[conv_i][0]
        splist = stimset_dict_rfmp_4a(xn, max_rf)

        # Array initializations
        weighted_max_maps = np.zeros((num_units, max_rf, max_rf))
        weighted_min_maps = np.zeros((num_units, max_rf, max_rf))
        non_overlap_max_maps = np.zeros((num_units, max_rf, max_rf))
        non_overlap_min_maps = np.zeros((num_units, max_rf, max_rf))
        padding = (xn - max_rf)//2

        # Present all bars to the model
        truncated_model = get_truncated_model(model, layer_idx)
        center_responses = barmap_run_01b(splist, truncated_model,
                                          num_units, batch_size=batch_size,
                                          _debug=_debug)

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
                results = executor.map(make_barmaps,
                                       [splist for _ in range(real_batch_size)],
                                       [center_responses for _ in range(real_batch_size)],
                                       [i for i in np.arange(unit_i, unit_i + real_batch_size)],
                                       [_debug for _ in range(real_batch_size)],
                                       [False for _ in range(real_batch_size)],
                                       [num_bars for _ in range(real_batch_size)],
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

        # for unit_i in range(num_units):
        #     make_barmaps(splist, center_responses, unit_i)

        # # Create maps of top/bottom bar average maps.
        # for unit_i in range(num_units):
        #     if _debug and (unit_i > 10):
        #         break
        #     print_progress(f"Making maps for unit {unit_i}...")
        #     weighted_max_map, weighted_min_map,\
        #     non_overlap_max_map, non_overlap_min_map,\
        #     num_weighted_max_bars, num_weighted_min_bars,\
        #     num_non_overlap_max_bars, num_non_overlap_min_bars=\
        #                         make_barmaps(splist, center_responses, unit_i,
        #                                      response_thr=0.1, stim_thr=0.2,
        #                                      _debug=_debug)
        #     # Crop and save maps to layer-level array
        #     weighted_max_maps[unit_i] = weighted_max_map[padding:padding+max_rf, padding:padding+max_rf]
        #     weighted_min_maps[unit_i] = weighted_min_map[padding:padding+max_rf, padding:padding+max_rf]
        #     non_overlap_max_maps[unit_i] = non_overlap_max_map[padding:padding+max_rf, padding:padding+max_rf]
        #     non_overlap_min_maps[unit_i] = non_overlap_min_map[padding:padding+max_rf, padding:padding+max_rf]

        #     # Record the number of bars used in each map (append to txt files).
        #     record_bar_counts(weighted_counts_path, layer_name, unit_i,
        #                       num_weighted_max_bars, num_weighted_min_bars)
        #     record_bar_counts(non_overlap_counts_path, layer_name, unit_i,
        #                       num_non_overlap_max_bars, num_non_overlap_min_bars)

        # Save the maps of all units.
        weighte_max_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_max_barmaps.npy")
        weighted_min_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_min_barmaps.npy")
        non_overlap_max_maps_path = os.path.join(result_dir, f"{layer_name}_non_overlap_max_barmaps.npy")
        non_overlap_min_maps_path = os.path.join(result_dir, f"{layer_name}_non_overlap_min_barmaps.npy")
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
        weighted_pdf_path = os.path.join(result_dir, f"{layer_name}_weighted_barmaps.pdf")
        make_map_pdf(weighted_max_maps, weighted_min_maps, weighted_pdf_path)
        non_overlap_pdf_path = os.path.join(result_dir, f"{layer_name}_non_overlap_barmaps.pdf")
        make_map_pdf(non_overlap_max_maps, non_overlap_min_maps, non_overlap_pdf_path)

#######################################.#######################################
#                                                                             #
#                                RFMAP_TOP_4C7O                               #
#                                                                             #
###############################################################################
def rfmp4c7o_run_01(model, model_name, result_dir, _debug=False, batch_size=100,
                    num_bars=500, conv_i_to_run=None):
    """
    Map the RF of all conv layers in model using RF mapping paradigm 4c7o,
    which is like paradigm 4a, but with 6 additional colors.
    
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
    # Note that the image_size upper bounds are set to (999, 999). This change
    # was made so that layers with RF larger than (227, 227) could be properly
    # centered during bar mapping.

    # Set paths
    tb1_path = os.path.join(result_dir, f"{model_name}_rfmp4c7o_tb1.txt")
    tb20_path = os.path.join(result_dir, f"{model_name}_rfmp4c7o_tb20.txt")
    tb100_path = os.path.join(result_dir, f"{model_name}_rfmp4c7o_tb100.txt")
    weighted_counts_path = os.path.join(result_dir, f"{model_name}_rfmp4c7o_weighted_counts.txt")
    non_overlap_counts_path = os.path.join(result_dir, f"{model_name}_rfmp4c7o_non_overlap_counts.txt")

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
        # In case we would like to run only one layer...
        if conv_i is not None:
            if conv_i != conv_i_to_run:
                continue
        
        layer_name = f"conv{conv_i + 1}"
        print(f"\n{layer_name}\n")
        # Get layer-specific info
        xn = xn_list[conv_i]
        layer_idx = layer_indices[conv_i]
        num_units = nums_units[conv_i]
        max_rf = max_rfs[conv_i][0]
        splist = stimset_dict_rfmp_4c7o(xn, max_rf)

        # Array initializations
        weighted_max_maps = np.zeros((num_units, 3, max_rf, max_rf))
        weighted_min_maps = np.zeros((num_units, 3, max_rf, max_rf))
        non_overlap_max_maps = np.zeros((num_units, 3, max_rf, max_rf))
        non_overlap_min_maps = np.zeros((num_units, 3, max_rf, max_rf))
        padding = (xn - max_rf)//2

        # Present all bars to the model
        truncated_model = get_truncated_model(model, layer_idx)
        center_responses = barmap_run_01b(splist, truncated_model,
                                          num_units, batch_size=batch_size,
                                          _debug=_debug, has_color=True)

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
                results = executor.map(make_barmaps,
                                       [splist for _ in range(real_batch_size)],
                                       [center_responses for _ in range(real_batch_size)],
                                       [i for i in np.arange(unit_i, unit_i + real_batch_size)],
                                       [_debug for _ in range(real_batch_size)],
                                       [True for _ in range(real_batch_size)],
                                       [num_bars for _ in range(real_batch_size)],
                                       )
            # Crop and save maps to layer-level array
            for result_i, result in enumerate(results):
                weighted_max_maps[unit_i + result_i] = result[0][:,padding:padding+max_rf, padding:padding+max_rf]
                weighted_min_maps[unit_i + result_i] = result[1][:,padding:padding+max_rf, padding:padding+max_rf]
                non_overlap_max_maps[unit_i + result_i] = result[2][:,padding:padding+max_rf, padding:padding+max_rf]
                non_overlap_min_maps[unit_i + result_i] = result[3][:,padding:padding+max_rf, padding:padding+max_rf]
                # Record the number of bars used in each map (append to txt files).
                record_stim_counts(weighted_counts_path, layer_name, unit_i + result_i,
                                  result[4], result[5])
                record_stim_counts(non_overlap_counts_path, layer_name, unit_i + result_i,
                                  result[6], result[7])
            unit_i += real_batch_size

        # Save the maps of all units.
        weighte_max_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_max_barmaps.npy")
        weighted_min_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_min_barmaps.npy")
        non_overlap_max_maps_path = os.path.join(result_dir, f"{layer_name}_non_overlap_max_barmaps.npy")
        non_overlap_min_maps_path = os.path.join(result_dir, f"{layer_name}_non_overlap_min_barmaps.npy")
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
        # Note: the splist and center responses are recorded as text files
        #       because we are interested in reducing the number of bar
        #       stimuli. We intend to drop the small bars if they are not
        #       commonly found as the top and bottom N bars.
        top_n = min(5000, len(splist)//2)
        max_center_reponses_path = os.path.join(result_dir, f"{layer_name}_top5000_responses.txt")
        min_center_reponses_path = os.path.join(result_dir, f"{layer_name}_bot5000_responses.txt")
        if os.path.exists(max_center_reponses_path):
            os.remove(max_center_reponses_path)
        if os.path.exists(min_center_reponses_path):
            os.remove(min_center_reponses_path)
        record_center_responses(max_center_reponses_path, center_responses, top_n, is_top=True)
        record_center_responses(min_center_reponses_path, center_responses, top_n, is_top=False)
        
        # Make pdf for the layer.
        weighted_pdf_path = os.path.join(result_dir, f"{layer_name}_weighted_barmaps.pdf")
        make_map_pdf(np.transpose(weighted_max_maps, (0,2,3,1)),
                     np.transpose(weighted_min_maps, (0,2,3,1)),
                     weighted_pdf_path)
        non_overlap_pdf_path = os.path.join(result_dir, f"{layer_name}_non_overlap_barmaps.pdf")
        make_map_pdf(np.transpose(non_overlap_max_maps, (0,2,3,1)),
                     np.transpose(non_overlap_min_maps, (0,2,3,1)),
                     non_overlap_pdf_path)


# Test
# if __name__ == '__main__':
#     dummy_map = np.zeros((11, 11))
#     dummy_map[2, 4] = 1
#     com0, com1, radius = mapstat_comr_1(dummy_map, 0.9)
#     print(com0, com1, radius)


#######################################.#######################################
#                                                                             #
#                            MAKE_RFMP4a_GRID_PDF                             #
#                                                                             #
###############################################################################
def make_rfmp4a_grid_pdf(pdf_path, model):
    xn_list = xn_to_center_rf(model, image_size=(999,999))  # Get the xn just big enough.
    img_size = 227
    layer_indices, max_rfs = get_rf_sizes(model, (img_size, img_size), layer_type=nn.Conv2d)
    num_layers = len(max_rfs)

    # Array of bar lengths
    barlen_ratios = np.array([48/64,
                              24/64,
                              12/64,
                               6/64])
    barlenstr = np.array(['48/64',
                          '24/64',
                          '12/64',
                          '6/64'])
    # Array of aspect ratios
    arat = np.array([1/2, 1/5, 1/10])

    with PdfPages(pdf_path) as pdf:
        for blr_i, bl_ratio in enumerate(barlen_ratios):
            for ar in arat:
                plt.figure(figsize=(4*num_layers, 5))
                plt.suptitle(f"Bar Length = {barlenstr[blr_i]} M, Aspect Ratio = {ar}", fontsize=24)

                for conv_i, max_rf in enumerate(max_rfs):
                    layer_name = f"conv{conv_i + 1}"
                    layer_index = layer_indices[conv_i]
                    # Get layer-specific info
                    xn = xn_list[conv_i]
                    max_rf = max_rf[0]

                    # Set bar parameters
                    bl = bl_ratio * max_rf
                    bw = bl * ar
                    xlist = stimset_gridx_map(max_rf,bl)

                    # Plot the bar
                    bar = stimfr_bar(xn, xn, 0, 0, 15, bl, bw, 0.5, 1, 0.5)
                    plt.subplot(1, num_layers, conv_i+1)
                    plt.imshow(bar, cmap='gray', vmin=0, vmax=1)
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
                    # ax.invert_yaxis()
        
                pdf.savefig()
                plt.show()
                plt.close()


# Generate a RFMP4a grid pdf for AlexNet
if __name__ == "__main__":
    # model = models.resnet18()
    # model_name = 'resnet18'
    model = models.alexnet()
    model_name = 'alexnet'
    pdf_path = os.path.join(c.REPO_DIR,'results','rfmp4a','mapping', 'test',
                            f'{model_name}_test_grid.pdf')
    make_rfmp4a_grid_pdf(pdf_path, model)
