"""
Code for generating stimuli.

Tony Fu, July 6th, 2022
"""
import sys
import math

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('..')
import constants as c


#######################################.#######################################
#                                                                             #
#                                    CLIP                                     #
#                                                                             #
###############################################################################
@njit
def clip(val, vmin, vmax):
    """Limits value to be vmin <= val <= vmax."""
    if vmin > vmax:
        raise Exception("vmin should be smaller than vmax.")
    val = min(val, vmax)
    val = max(val, vmin)
    return val


#######################################.#######################################
#                                                                             #
#                                  DRAW_BAR                                   #
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
    # Usually, the negative sign appears in c instead of d, but the y-axis
    # points downward in our case.
    dx_r = rot_a*dx + rot_c*dy
    dy_r = rot_b*dx + rot_d*dy
    return dx_r, dy_r


@njit  # sped up by 182x
def draw_bar(xn, yn, x0, y0, theta, blen, bwid, laa, fgval, bgval):
    """
    Creates a bar stimulus. Code originally from by Dr. Wyeth Bair's
    d06_util_bargen.py, modified for readability.

    Parameters
    ----------
    xn : int
        The horizontal width of returned array.
    yn : int
        The vertical height of returned array.
    x0 : float
        The horizontal coordinate (origin is top left) of center (pix).
    y0 : float
        The vertical coordinate (origin is top left) of center (pix).
    theta : float
        The orientation of the bar (deg).
    blen : float
        The length of bar (pix).
    bwid : float
        The width of bar (pix).
    laa : float
        The thickness of anti-aliasing smoothing (pix).
    fgval : float
        The bar luminance. Preferably [-1..1].
    bgval : float
        The background luminance. Preferably [-1..1].

    Returns
    -------
    s : numpy array of size [yn, xn]
        The bar stimulus with a background.
    """
    s = np.full((yn,xn), bgval, dtype='float32')  # Fill w/ BG value
    dval = fgval - bgval  # Luminance difference
  
    # Maximum extent of unrotated bar corners in zero-centered frame:
    dx = bwid/2
    dy = blen/2

    # Rotate top-left corner from (-dx, dy) to (dx1, dy1)
    dx1, dy1 = rotate(-dx, dy, theta)
    
    # Rotate top-right corner from (dx, dy) to (dx2, dy2)
    dx2, dy2 = rotate(dx, dy, theta)

    # Maximum extent of rotated bar corners in zero-centered frame:
    maxx = laa + max(abs(dx1), abs(dx2))
    maxy = laa + max(abs(dy1), abs(dy2))

    # Define the 4 corners a box that contains the rotated bar.
    bar_left_i   = round(x0 - maxx)
    bar_right_i  = round(x0 + maxx) + 1
    bar_top_i    = round(y0 - maxy)
    bar_bottom_i = round(y0 + maxy) + 1

    bar_left_i   = clip(bar_left_i  , 0, xn)
    bar_right_i  = clip(bar_right_i , 0, xn)
    bar_top_i    = clip(bar_top_i   , 0, yn)
    bar_bottom_i = clip(bar_bottom_i, 0, yn)

    for i in range(bar_left_i, bar_right_i):  # for i in range(0,xn):
        xx = i - x0  # relative to bar center
        for j in range (bar_top_i, bar_bottom_i):  # for j in range (0,yn):
            yy = j - y0  # relative to bar center
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

            if laa > 0.0:
                if db > laa:
                    f = 1.0  # This point is inside the bar
                elif db < -laa:
                    f = 0.0  # This point is outside the bar
                else:  # Use sinusoidal sigmoid
                    f = 0.5 + 0.5*math.sin(db/laa * 0.25*math.pi)
            else:
                if db >= 0.0:
                    f = 1.0  # inside
                else:
                    f = 0.0  # outside

            s[j, i] += f * dval  # add a fraction 'f' of the 'dval'

    return s


def _test_draw_bar():
    xn = 200
    yn = 300
    # rotate test
    for theta in np.linspace(0, 180, 10):
        bar = draw_bar(xn, yn, xn//2, yn//2, theta, 100, 50, 1, 1, -1)
        plt.imshow(bar, cmap='gray')
        plt.title(f"{theta}")
        plt.show()

    # move from left to right
    for x0 in np.linspace(0, yn, 10):
        bar = draw_bar(xn, yn, x0, yn//2, 45, 80, 30, 2, 1, 0)
        plt.imshow(bar, cmap='gray')
        plt.title(f"{x0:.2f}")
        plt.show()


if __name__ == "__main__":
    _test_draw_bar()
