"""
Code for generating bar stimuli.

Note: The y-axis points downward.

July 15, 2022
"""
import math

import numpy as np
from numba import njit
import matplotlib.pyplot as plt


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
    Return a numpy array that has a bar on a background:

    xn    - (int) horizontal width of returned array
    yn    - (int) vertical height of returned array
    x0    - (float) horizontal offset of center (pix)
    y0    - (float) vertical offset of center (pix)
    theta - (float) orientation (pix)
    blen  - (float) length of bar (pix)
    bwid  - (float) width of bar (pix)
    aa    - (float) length scale for anti-aliasing (pix)
    fgval - (float) bar luminance [0..1]
    bgval - (float) background luminance [0..1]
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
        plt.title(f"{theta:.2f}")
        plt.show()

    # move from left to right
    for x0 in np.linspace(-xn//2, xn//2, 5):
        bar = stimfr_bar(xn, yn, x0, 0, 45, 80, 30, 2, 1, 0)
        plt.imshow(bar, cmap='gray')
        plt.title(f"{x0:.2f}")
        plt.show()
        

#######################################.#######################################
#                                                                             #
#                               STIMFR_BAR_COLOR                              #
#                                                                             #
###############################################################################
# njit slowed down by 4x
def stimfr_bar_color(xn,yn,x0,y0,theta,blen,bwid,aa,r1,g1,b1,r0,g0,b0):
    """
    Return a numpy array (3, yn, xn) that has a bar on a background:

    xn    - (int) horizontal width of returned array
    yn    - (int) vertical height of returned array
    x0    - (float) horizontal offset of center (pix)
    y0    - (float) vertical offset of center (pix)
    theta - (float) orientation (pix)
    blen  - (float) length of bar (pix)
    bwid  - (float) width of bar (pix)
    laa   - (float) length scale for anti-aliasing (pix)
    r1,g1,b1 - bar color
    r0,g0,b0 - background color
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
    splist  - stimulus parameter list - APPEND TO THIS LIST
    xn      - horizontal and vertical image size
    xlist   - list of x-coordinates (pix)
    orilist - list of orientation values (degr)
    blen    - Length of bar (pix)
    bwid    - Width of bar (pix)
    aa      - Anti-aliasing space constant (pix)
    """
    yn = xn        # Assuming image is square
    ylist = xlist  # Use same coordinates for y grid locations
    
    fgval =  1.0  # Foreground luminance
    bgval = -1.0  # Background luminance
    
    nstim = len(xlist) * len(ylist) * len(orilist) * 2
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
    max_rf - maximum RF size (pix)
    blen   - bar length (pix)
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
    xn     - stimulus image size (pix)
    max_rf - maximum RF size (pix)
    """
    splist = []  # List of dictionary entries, one per stimulus image

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

    print("  Length of stimulus parameter list:",len(splist))

    return splist



#
#  HERE IS AN EXAMPLE OF HOW TO CALL THE CODE ABOVE:
#
if __name__ == "__main__":
    s = stimset_dict_rfmp_4a(11,11)



