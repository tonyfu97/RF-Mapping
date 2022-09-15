"""
Basic functions for presenting and recording stimuli to network. Used by
other .py files like bar.py, pasu_shape.py, and grating.py

Tony Fu, Sep 14, 2022
"""
import sys
import math
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


__all__ = ['clip', 'stimset_gridx_map', 'print_progress',
           'add_weighted_map', 'add_non_overlap_map',
           'summarize_TB1', 'summarize_TBn', 'record_stim_counts',
           'record_splist', 'record_center_responses',
           'mapstat_comr_1', 'make_map_pdf']


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
    warnings.warn("stimulus.py cannot import Numba.")
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
#                                    CLIP                                     #
#                                                                             #
###############################################################################
@jit
def clip(val, vmin, vmax):
    """Limits value to be vmin <= val <= vmax"""
    if vmin > vmax:
        raise Exception("vmin should be smaller than vmax.")
    val = min(val, vmax)
    val = max(val, vmin)
    return val


#######################################.#######################################
#                                                                             #
#                              STIMSET_GRIDX_MAP                              #
#                                                                             #
#  Given a stimulus length and maximum RF size (both in pixels), return a list#
#  of the x-coordinates of the grid points relative to the center of the      #
#  image field.                                                               #
#                                                                             #
#  I believe the following are true:                                          #
#  (1) The center coordinate "0.0" will always be included                    #
#  (2) There will be an odd number of coordinates                             #
#  (3) The extreme coordinates will never be more then half of a stimulus     #
#      length outside of the maximum RF ('max_rf')                            #
#                                                                             #
###############################################################################
def stimset_gridx_map(max_rf,stim_len):
    """
    Parameters
    ----------
    max_rf   - maximum RF size (pix)\n
    stim_len - stimulus length (pix)\n
    """
    dx = stim_len / 2.0                       # Grid spacing is 1/2 of stimulus length
    xmax = round((max_rf/dx) / 2.0) * dx  # Max offset of grid point from center
    xlist = np.arange(-xmax,xmax+1,dx)
    return xlist


# Test
if __name__ == '__main__':
    max_rf = 49
    blen = 5
    print(stimset_gridx_map(max_rf,blen))


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
#                             ADD_WEIGHTED_MAP                                #
#                                                                             #
###############################################################################
def add_weighted_map(new_stim, sum_map, response):
    """
    Add the new_stim, weighted by the unit's rectified response to the
    stimulus, to the sum_map.

    Parameters
    ----------
    new_stim - stimulus to be added to the map.
    sum_map  - cumulative stimulus map.
    response - the unit's response to the new_stim.
    """
    sum_map += new_stim * response


#######################################.#######################################
#                                                                             #
#                             ADD_NON_OVERLAP_MAP                             #
#                                                                             #
###############################################################################
def add_non_overlap_map(new_stim, sum_map, stim_thr):
    """
    Add the new_stim to the map if the new_stim is not overlapping with any
    existing stimuli. The new_stim is first binarized with the {stim_thr}
    threshold to get rid of some of the anti-aliasing pixels.

    Parameters
    ----------
    new_stim - stimulus to be added to the map.\n
    sum_map  - cumulative stimulus map.\n
    stim_thr - stimulus pixels w/ a value below stim_thr will be excluded.

    Returns
    -------
    True if the new_stim has been included in the 
    """
    # Binarize new_stim
    new_stim[new_stim < stim_thr] = 0
    new_stim[new_stim >= stim_thr] = 1
    # Only add the new stim if it is not overlapping with any existing stimuli.
    if not np.any(np.logical_and(sum_map>0, new_stim>0)):
        sum_map += new_stim
        return True
    return False


#######################################.#######################################
#                                                                             #
#                                SUMMARIZE_TB1                                #
#                                                                             #
###############################################################################
def summarize_TB1(splist, center_responses, layer_name, txt_path):
    """
    Summarize the top and bottom stimuli in a .txt file in format:
    layer_name, unit_i, top_idx, top_x, top_y, top_r, bot_idx, bot_x, bot_y

    Parameters
    ----------
    splist           - stimulus parameter list.\n
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
            
            top_stim = splist[top_i]
            bot_stim = splist[bot_i]
            f.write(f"{layer_name:} {unit_i:} ")
            f.write(f"{top_i:} {top_stim['x0']:.2f} {top_stim['y0']:.2f} {top_r:.4f} ")
            f.write(f"{bot_i:} {bot_stim['x0']:.2f} {bot_stim['y0']:.2f} {bot_r:.4f}\n")


#######################################.#######################################
#                                                                             #
#                                SUMMARIZE_TBn                                #
#                                                                             #
###############################################################################
def summarize_TBn(splist, center_responses, layer_name, txt_path, top_n=20):
    """
    Summarize the top- and bottom-n stimuli in a .txt file in format:
    layer_name, unit_i, top_avg_x, top_avg_y, bot_avg_x, bot_avg_y

    Parameters
    ----------
    splist           - the stimulus parameter list.\n
    center_responses - the responses of center unit in [stim_i, unit_i] format.\n 
    model_name       - name of the model. Used for file naming.\n
    layer_name       - name of the layer. Used as file entries/primary key.\n
    txt_dir          - the path name of the file, must end with '.txt'\n
    top_n            - the top and bottom N stimuli to record.
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

                # Equally weighted sum for avg coordinates of stimuli.
                top_avg_x += splist[top_i]['x0']/top_n
                top_avg_y += splist[top_i]['y0']/top_n
                bot_avg_x += splist[bot_i]['x0']/top_n
                bot_avg_y += splist[bot_i]['y0']/top_n

            f.write(f"{layer_name} {unit_i} ")
            f.write(f"{top_avg_x:.2f} {top_avg_y:.2f} ")
            f.write(f"{bot_avg_x:.2f} {bot_avg_y:.2f}\n")


#######################################.#######################################
#                                                                             #
#                              RECORD_STIM_COUNTS                             #
#                                                                             #
###############################################################################
def record_stim_counts(txt_path, layer_name, unit_i, num_max_stim, num_min_stim):
    """Write the numbers of stimuli used in the top and bottom maps."""
    with open(txt_path, 'a') as f:
        f.write(f"{layer_name} {unit_i} {num_max_stim} {num_min_stim}\n")


#######################################.#######################################
#                                                                             #
#                               RECORD_SPLIST                                 #
#                                                                             #
###############################################################################
def record_splist(txt_path, splist):
    """Write the contents of splist into a text file."""
    with open(txt_path, 'a') as f:
        for stimulus_idx, params in enumerate(splist):
            f.write(f"{stimulus_idx}")
            for val in params.values():
                f.write(f" {val}")
            f.write('\n')


#######################################.#######################################
#                                                                             #
#                           RECORD_CENTER_RESPONSES                           #
#                                                                             #
###############################################################################
def record_center_responses(txt_path, center_responses, top_n, is_top):
    """
    Write the indicies and responses of the top- and bottom-N into a text file.
    """
    num_units = center_responses.shape[1]  # in dimension: [stimulus, unit]
    center_responses_sorti = np.argsort(center_responses, axis=0)
    if is_top:
        center_responses_sorti = np.flip(center_responses_sorti, 0)
    with open(txt_path, 'a') as f:
        for unit_i in range(num_units):
            for i, stimulus_idx in enumerate(center_responses_sorti[:, unit_i]):
                if i >= top_n:
                    break
                f.write(f"{unit_i} {i} {stimulus_idx} ")
                f.write(f"{center_responses[stimulus_idx, unit_i]:.4f}\n")
                # Format: unit_i, rank, stimulus_index, response_value


#######################################.#######################################
#                                                                             #
#                                MAPSTAT_COMR_1                               #
#                                                                             #
#  For a 2D array 'map', compute the (x,y) center of mass and the radius      #
#  that contains a fraction 'f' of the area of the map.                       #
#                                                                             #
###############################################################################
def mapstat_comr_1(map,f):
    #
    #
    map = map.copy()
    map = map - map.min()

    xn = len(map)
    list0 = np.sum(map,1)  # Sum along axis 1, to ultimately get COM on axis 0
    list1 = np.sum(map,0)  # Sum along axis 0, to ultimately get COM on axis 1
    total = np.sum(list0)  # Overall total weight of entire map
    xvals = np.arange(xn)  # X-values (pix)
    prod0 = xvals*list0
    prod1 = xvals*list1
    if (total > 0.0):
        com0 = np.sum(prod0)/total
        com1 = np.sum(prod1)/total
    else:
        com0 = com1 = -1

    dist2 = []  # empty list to hold squared distances from COM
    magn = []   # empty list to hold magnitude
    for i in range(xn):
        di2 = (i-com0)*(i-com0)
        for j in range(xn):
            dj2 = (j-com1)*(j-com1)
            if (map[i,j] > 0.0):
                dist2.append(di2 + dj2)
                magn.append(map[i,j])
    
    isort = np.argsort(dist2)   # Get list of indices that sort list (least 1st)
    
    if (com0 == -1):
        return -1, -1, -1
    
    n = len(dist2)
    
    # Go down the sorted list, adding up the magnitudes, until the fractional
    #  criterion is exceeded.  Compute the radius for the final point added.
    #
    tot = 0.0
    k = 0
    for i in range(n):  # for each non-zero position in the map
        k = isort[i]      # Get index 'k' of the next point closest to the COM
        tot += magn[k]
        if (tot/total >= f):
            break
    radius = np.sqrt(dist2[k])
    return com0, com1, radius


#######################################.#######################################
#                                                                             #
#                                 MAKE_MAP_PDF                                #
#                                                                             #
###############################################################################
def make_map_pdf(max_maps, min_maps, pdf_path):
    """
    Make a pdf, one unit per page.

    Parameters
    ----------
    maps     - maps with dimensions [unit_i, y, x] (black and white) or
               [unit_i, y, x, rgb] (color)\n
    pdf_path - path name of the file, must end with '.pdf'\n
    """
    yn, xn = max_maps.shape[1:3]

    with PdfPages(pdf_path) as pdf:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        im1 = ax1.imshow(np.zeros((yn, xn, 3)), vmax=1, vmin=0, cmap='gray')
        im2 = ax2.imshow(np.zeros((yn, xn, 3)), vmax=1, vmin=0, cmap='gray')
        for unit_i, (max_map, min_map) in enumerate(zip(max_maps, min_maps)):
            print_progress(f"Making pdf for unit {unit_i}...")
            fig.suptitle(f"no.{unit_i}", fontsize=20)

            vmax = max_map.max()
            vmin = max_map.min()
            vrange = vmax - vmin
            if math.isclose(vrange, 0):
                vrange = 1
            im1.set_data((max_map-vmin)/vrange)
            ax1.set_title('max')

            vmax = min_map.max()
            vmin = min_map.min()
            vrange = vmax - vmin
            if math.isclose(vrange, 0):
                vrange = 1
            im2.set_data((min_map-vmin)/vrange)
            ax2.set_title('min')

            plt.show()
            pdf.savefig(fig)
            plt.close()
