"""
Code for generating bar stimuli.

Note: The y-axis points downward.

July 15, 2022
"""
import os
import sys
import math

import numpy as np
from numba import njit
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights, VGG16_Weights
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from hook import ConvUnitCounter
from files import delete_all_npy_files, delete_all_file_of_extension
from spatial import (xn_to_center_rf,
                     truncated_model,
                     calculate_center,
                     get_rf_sizes,)
import constants as c


#######################################.#######################################
#                                                                             #
#                                    CLIP                                     #
#                                                                             #
###############################################################################
@njit
def clip(val, vmin, vmax):
    """Limits value to be vmin <= val <= vmax"""
    if vmin > vmax:
        raise Exception("vmin should be smaller than vmax.")
    val = min(val, vmax)
    val = max(val, vmin)
    return val


#######################################.#######################################
#                                                                             #
#                                   ROTATE                                    #
#                                                                             #
###############################################################################
@njit
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
@njit  # sped up by 182x
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
if __name__ == "__main__":
    xn = 200
    yn = 300
    # rotate test
    for theta in np.linspace(0, 180, 8):
        bar = stimfr_bar(xn, yn, 0, 0, theta, 100, 50, 1, 1, -1)
        plt.imshow(bar, cmap='gray')
        plt.title(f"theta = {theta:.2f}")
        plt.show()

    # move from top to bottom
    for y0 in np.linspace(-yn//2, yn//2, 5):
        bar = stimfr_bar(xn, yn, 0, y0, 45, 80, 30, 2, 1, 0)
        plt.imshow(bar, cmap='gray')
        plt.title(f"y = {y0:.2f}")
        plt.show()
        

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
    xn = 200
    yn = 240
    colors = {'red':     (1, 0, 0),
              'green':   (0, 1, 0),
              'blue':    (0, 0, 1),
              'yellow':  (1, 1, 0),
              'magenta': (1, 0, 1),
              'cyan':    (0, 1, 1),
              'white':   (1, 1, 1),
              'black':   (0, 0, 0)}
    for color_name, (r1, g1, b1) in colors.items():
        bar = stimfr_bar_color(xn, yn, 10, 0, 45, 80, 30, 2,
                               r1, g1, b1,
                               0.5, 0.5, 0.5)
        plt.imshow(np.transpose(bar, (1,2,0)))
        plt.title(color_name)
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
#                             STIMSET_GRIDX_BARMAP                            #
#                                                                             #
#  Given a bar length and maximum RF size (both in pixels), return a list     #
#  of the x-coordinates of the grid points relative to the center of the      #
#  image field.                                                               #
#                                                                             #
#  I believe the following are true:                                          #
#  (1) The center coordinate "0.0" will always be included                    #
#  (2) There will be an odd number of coordinates                             #
#  (3) The extreme coordinates will never be more then half of a bar length   #
#      outside of the maximum RF ('max_rf')                                   #
#                                                                             #
###############################################################################
def stimset_gridx_barmap(max_rf,blen):
    """
    Parameters
    ----------
    max_rf - maximum RF size (pix)\n
    blen   - bar length (pix)\n
    """
    dx = blen / 2.0                       # Grid spacing is 1/2 of bar length
    xmax = round((max_rf/dx) / 2.0) * dx  # Max offset of grid point from center
    xlist = np.arange(-xmax,xmax+1,dx)

    return xlist


# Test
if __name__ == '__main__':
    max_rf = 49
    blen = 5
    print(stimset_gridx_barmap(max_rf,blen))


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
        xlist = stimset_gridx_barmap(max_rf,bl)
        for ar in arat:
            stim_dapp_bar_xyo_bw(splist,xn,xlist,orilist,bl,ar*bl,aa)

    # print("  Length of stimulus parameter list:",len(splist))

    return splist




#  HERE IS AN EXAMPLE OF HOW TO CALL THE CODE ABOVE:

if __name__ == "__main__":
    s = stimset_dict_rfmp_4a(11,11)


#######################################.#######################################
#                                                                             #
#                               PRINT_PROGRESS                                #
#                                                                             #
###############################################################################
def print_progress(text):
    """
    Prints progress (whatever text) without printing a new line everytime.
    """
    sys.stdout.write('\r')
    sys.stdout.write(text)
    sys.stdout.flush()


#######################################.#######################################
#                                                                             #
#                                BARMAP_RUN_01b                               #
#                                                                             #
###############################################################################
def barmap_run_01b(splist, model, layer_idx, num_units, batch_size=100,
                  _debug=False):
    """
    Presents bars and returns the center responses in array of dimension:
    [num_stim, num_units].

    Parameters
    ----------
    splist     - bar stimulus parameter list.\n
    model      - neural network.\n
    layer_idx  - index of the layer of interest.\n
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
        if _debug and bar_i > 500:
            break
        real_batch_size = min(batch_size, num_stim-bar_i)
        bar_batch = np.zeros((real_batch_size, 3, yn, xn))

        # Create a batch of bars.
        for i in range(real_batch_size):
            params = splist[bar_i + i]
            new_bar = stimfr_bar(params['xn'], params['yn'],
                                 params['x0'], params['y0'],
                                params['theta'], params['len'], params['wid'], 
                                params['aa'], params['fgval'], params['bgval'])
            # Replicate new bar to all color channel.
            bar_batch[i, 0] = new_bar
            bar_batch[i, 1] = new_bar
            bar_batch[i, 2] = new_bar

        # Present the patch of bars to the truncated model.
        y, _ = truncated_model(torch.tensor(bar_batch).type('torch.FloatTensor'),
                               model, layer_idx)
        yc, xc = calculate_center(y.shape[-2:])
        center_responses[bar_i:bar_i+real_batch_size, :] = y[:, :, yc, xc].detach().numpy()
        print_progress(f"Presenting {bar_i}/{num_stim} stimuli...")
        bar_i += real_batch_size

    return center_responses


#######################################.#######################################
#                                                                             #
#                         MRFMAP_MAKE_NON_OVERLAP_MAP                         #
#                                                                             #
###############################################################################
def mrfmap_make_non_overlap_map(splist, center_responses, unit_i, response_thr=0.1,
                                stim_thr=0.2, _debug=False):
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
    Returns the non-overlapping sums of the top and bottom bars of one unit.
    """
    xn = splist[0]['xn']
    yn = splist[0]['yn']
    max_map = np.zeros((yn, xn))
    min_map = np.zeros((yn, xn))

    isort = np.argsort(center_responses[:, unit_i])  # Ascending
    r_max = center_responses[:, unit_i].max()
    r_min = center_responses[:, unit_i].min()

    if _debug:
        print(f"unit {unit_i}: r_max: {r_max:7.2f}, max bar idx: {isort[::-1][:5]}")

    for max_bar_i in isort[::-1]:
        if center_responses[max_bar_i, unit_i] < abs(response_thr * r_max):
            break
        params = splist[max_bar_i]
        new_bar = stimfr_bar(params['xn'], params['yn'],
                             params['x0'], params['y0'],
                            params['theta'], params['len'], params['wid'],
                            0.5, 1, 0)
        # Binarize new_bar
        new_bar[new_bar < stim_thr] = 0
        new_bar[new_bar >= stim_thr] = 1
        # Only add the new bar if it is not overlapping with any existing bars.
        if not np.any(np.logical_and(max_map>0, new_bar>0)):
            max_map += new_bar

    for min_bar_i in isort:
        if center_responses[min_bar_i, unit_i] > -abs(response_thr * r_min):
            break
        params = splist[min_bar_i]
        new_bar = stimfr_bar(params['xn'], params['yn'],
                             params['x0'], params['y0'],
                            params['theta'], params['len'], params['wid'],
                            0.5, 1, 0)
        # Binarize new_bar
        new_bar[new_bar < stim_thr] = 0
        new_bar[new_bar >= stim_thr] = 1
        # Only add the new bar if it is not overlapping with any existing bars.
        if not np.any(np.logical_and(min_map>0, new_bar>0)):
            min_map += new_bar

    return max_map, min_map


#######################################.#######################################
#                                                                             #
#                                SUMMARIZE_TB1                                #
#                                                                             #
###############################################################################
def summarize_TB1(splist, center_responses, layer_name, txt_path):
    """
    Summarize the top bar and bottom bar in a .txt file in format:
    layer_name, unit_i, top_idx, top_x, top_y, top_r, bot_idx, bot_x, bot_y

    Parameters
    ----------
    splist           - bar stimulus parameter list.\n
    center_responses - responses of center unit in [stim_i, unit_i] format.\n 
    model_name       - name of the model. Used for file naming.\n
    layer_name       - name of the layer. Used as file entries/primary key.\n
    txt_path         - path name of the file, must end with '.txt'\n
    """
    num_units = center_responses.shape[1]  # shape = [stim, unit]

    with open(txt_path, 'a') as f:
        for unit_i in range(num_units):
            isort = np.argsort(center_responses[:, unit_i])  # Ascending
            top_i = isort[-1]
            bot_i = isort[0]
            
            top_r = center_responses[top_i, unit_i]
            bot_r = center_responses[bot_i, unit_i]
            
            top_bar = splist[top_i]
            bot_bar = splist[bot_i]
            f.write(f"{layer_name:6} {unit_i:3} ")
            f.write(f"{top_i:5} {top_bar['x0']:6.2f} {top_bar['y0']:6.2f} {top_r:10.4f}")
            f.write(f"{bot_i:5} {bot_bar['x0']:6.2f} {bot_bar['y0']:6.2f} {bot_r:10.4f}\n")


#######################################.#######################################
#                                                                             #
#                                SUMMARIZE_TBn                                #
#                                                                             #
###############################################################################
def summarize_TBn(splist, center_responses, layer_name, txt_path, top_n=20):
    """
    Summarize the top n bars and bottom n bars in a .txt file in format:
    layer_name, unit_i, top_avg_x, top_avg_y, bot_avg_x, bot_avg_y

    Parameters
    ----------
    splist           - the bar stimulus parameter list.\n
    center_responses - the responses of center unit in [stim_i, unit_i] format.\n 
    model_name       - name of the model. Used for file naming.\n
    layer_name       - name of the layer. Used as file entries/primary key.\n
    txt_dir          - the path name of the file, must end with '.txt'\n
    top_n            - the top and bottom N bars to record.
    """
    num_units = center_responses.shape[1]  # shape = [stim, unit]

    with open(txt_path, 'a') as f:
        for unit_i in range(num_units):
            isort = np.argsort(center_responses[:, unit_i])  # Ascending
            
            # Initializations
            top_avg_x = 0
            top_avg_y = 0
            bot_avg_x = 0
            bot_avg_y = 0
            
            for i in range(top_n):
                top_i = isort[-i-1]
                bot_i = isort[i]
            
                top_avg_x += splist[top_i]['x0']/top_n
                top_avg_y += splist[top_i]['y0']/top_n
                bot_avg_x += splist[bot_i]['x0']/top_n
                bot_avg_y += splist[bot_i]['y0']/top_n

            f.write(f"{layer_name:6} {unit_i:3} ")
            f.write(f"{top_avg_x:6.2f} {top_avg_y:6.2f}")
            f.write(f"{bot_avg_x:6.2f} {bot_avg_y:6.2f}\n")


#######################################.#######################################
#                                                                             #
#                                 MAKE_MAP_PDF                                #
#                                                                             #
###############################################################################
def make_map_pdf(max_maps, min_maps, pdf_path, show=False):
    """
    Make a pdf, one unit per page.

    Parameters
    ----------
    maps     - maps with dimensions [unit_i, y, x].\n
    pdf_path - path name of the file, must end with '.pdf'\n
    """
    with PdfPages(pdf_path) as pdf:
        for unit_i, (max_map, min_map) in enumerate(zip(max_maps, min_maps)):
            print_progress(f"Making pdf for unit {unit_i}...")
            plt.figure(figsize=(10, 5))
            plt.suptitle(f"rfmp4a no.{unit_i}", fontsize=20)

            plt.subplot(1,2,1)
            plt.imshow(max_map, cmap='gray')
            plt.title('max')

            plt.subplot(1,2,2)
            plt.imshow(min_map, cmap='gray')
            plt.title('min')

            # if show: plt.show()
            pdf.savefig()
            plt.close()


#######################################.#######################################
#                                                                             #
#                                RFMP4a_RUN_01b                               #
#                                                                             #
###############################################################################
def rfmp4a_run_01b(model, model_name, result_dir, _debug=False):
    """
    Map the RF of all conv layers in model using RF mapping paradigm 4a.
    
    Parameters
    ----------
    model      - neural network.
    model_name - name of neural network. Used for txt file naming.
    result_dir - directories to save the npy, txt, and pdf files.
    _debug     - if true, run in debug mode.
    """
    xn_list = xn_to_center_rf(model)  # Get the xn just big enough.
    unit_counter = ConvUnitCounter(model)
    layer_indices, nums_units = unit_counter.count()
    _, max_rfs = get_rf_sizes(model, (227, 227), layer_type=nn.Conv2d)

    delete_all_npy_files(result_dir)
    delete_all_file_of_extension(result_dir, '.txt')
    for conv_i in range(len(layer_indices)):
        layer_name = f"conv{conv_i + 1}"
        print(f"{layer_name}\n")
        # Get layer-specific info
        xn = xn_list[conv_i]
        layer_idx = layer_indices[conv_i]
        num_units = nums_units[conv_i]
        max_rf = max_rfs[conv_i][0]
        splist = stimset_dict_rfmp_4a(xn, max_rf)
        
        # Array initializations
        max_maps = np.zeros((num_units, max_rf, max_rf))
        min_maps = np.zeros((num_units, max_rf, max_rf))
        padding = (xn - max_rf)//2
        
        center_responses = barmap_run_01b(splist, model, layer_idx,
                                          num_units, batch_size=100,
                                          _debug=_debug)
        
        # Create txt files that summarize the top and bottom bars.
        tb1_path = os.path.join(result_dir, f"{model_name}_rfmp4a_tb1.txt")
        tb20_path = os.path.join(result_dir, f"{model_name}_rfmp4a_tb20.txt")
        tb100_path = os.path.join(result_dir, f"{model_name}_rfmp4a_tb100.txt")
        summarize_TB1(splist, center_responses, layer_name, tb1_path)
        summarize_TBn(splist, center_responses, layer_name, tb20_path, top_n=20)
        summarize_TBn(splist, center_responses, layer_name, tb100_path, top_n=100)

        # Create maps of top/bottom bar average maps.
        for unit_i in range(num_units):
            if _debug and (unit_i > 10):
                break
            print_progress(f"Making maps for unit {unit_i}...")
            max_map, min_map = mrfmap_make_non_overlap_map(splist, center_responses,
                                unit_i, response_thr=0.1, stim_thr=0.2, _debug=_debug)
            max_maps[unit_i] = max_map[padding:padding+max_rf, padding:padding+max_rf]
            min_maps[unit_i] = min_map[padding:padding+max_rf, padding:padding+max_rf]
        
        # Save the maps of all units.
        max_maps_path = os.path.join(result_dir, f"{layer_name}_max_maps.npy")
        min_maps_path = os.path.join(result_dir, f"{layer_name}_min_maps.npy")
        np.save(max_maps_path, max_maps)
        np.save(min_maps_path, min_maps)

        # Make pdf for the layer.
        pdf_path = os.path.join(result_dir, f"{layer_name}_maps.pdf")
        make_map_pdf(max_maps, min_maps, pdf_path, show=_debug)
