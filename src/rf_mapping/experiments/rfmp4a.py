"""
Functions for receptive field mapping paradigm 4a.

Note: all code assumes that the y-axis points downward.

Modified from Dr. Wyeth Bair's d06_mrf.py
Tony Fu, July 4th, 2022
"""
import os
from re import L
import sys
import math
import copy

import numpy as np
from numba import njit
from pathlib import Path
import scipy as sp
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('..')
from hook import get_conv_output_shapes, SpatialIndexConverter, get_rf_sizes
from image import make_box


#######################################.#######################################
#                                                                             #
#                                  DRAW_BAR                                   #
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


@njit
def rotate(dx, dy, theta_deg):
    """
    Applies the rotation matrix:
    [dx, dy] * [[a, b], [c, d]]  = [dx_r, dy_r]
    
    To undo rotation, apply again with negative theta_deg.
    """
    thetar = theta_deg * math.pi / 180.0
    a = math.cos(thetar); b = math.sin(thetar)
    c = math.sin(thetar); d = -math.cos(thetar)
    # Usually, the negative sign is given to c instead of d, but y axis points
    # downward.
    dx_r = a*dx + c*dy
    dy_r = b*dx + d*dy
    return dx_r, dy_r


@njit  # sped up by 182x
def draw_bar(xn, yn, x0, y0, theta, blen, bwid, laa, fgval, bgval):
    """
    Creates a bar stimulus.

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
    # _test_draw_bar()
    pass


#######################################.#######################################
#                                                                             #
#                                   RF GRID                                   #
#                                                                             #
###############################################################################
class RfGrid:
    def __init__(self, model, image_shape):
        self.model = copy.deepcopy(model)
        self.image_shape = image_shape
        self.converter = SpatialIndexConverter(model, image_shape)
    
    def _divide_from_middle(self, start, end, increment):
        """
        For example, if given min = 15, max = 24, increment = 4:

          -15---16---17---18---19---20---21---22---23---24-

        Divides it into:

              |                   |                   |
          -15-|-16---17---18---19-|-20---21---22---23-|-24-
              |                   |                   |
        
        Then rounds the numbers to the nearest intergers:

                |                   |                   |
          -15---16---17---18---19---20---21---22---23---24-
                |                   |                   |
        
        Returns [16, 20, 24] in this case.
        """
        if math.isclose(increment, 0):
            raise ValueError("The increment is too close to zero.")

        middle = (start + end)/2
        indices = [middle]
        
        # Find indices that are smaller than the middle.
        while (indices[-1] - increment >= start):
            indices.append(indices[-1] - increment)
        indices = indices[::-1]

        # Find indices that are larger than the middle.
        while (indices[-1] + increment <= end):
            indices.append(indices[-1] + increment)

        return [round(i) for i in indices]

    def get_grid_coords(self, layer_idx, spatial_index, grid_spacing):
        """
        Generates a list of coordinates that equally divide the receptive
        field of a unit up to some rounding. The grid is centered at the center
        of the receptive field.

        Parameters
        ----------
        layer_idx : int
            The index of the layer. See 'hook.py' module for details.
        spatial_index : int or (int, int)
            The spatial position of the unit of interest. Either in (y, x)
            format or a flatten index. Not in pixels but should be w.r.t.
            the output maps of the layer.
        grid_spacing : float
            The spacing between the grid lines (pix).

        Returns
        -------
        grid_coords : [(int, int), ...]
            The coordinates of the intersections in [(x0, y0), (x1, y1), ...]
            format.
        """
        # Project the unit backward to the image space.
        y_min, x_min, y_max, x_max = self.converter.convert(spatial_index,
                                                            layer_idx,
                                                            end_layer_index=0,
                                                            is_forward=False)
        x_list = self._divide_from_middle(x_min, x_max, grid_spacing)
        y_list = self._divide_from_middle(y_min, y_max, grid_spacing)
        
        grid_coords = []
        for x in x_list:
            for y in y_list:
                grid_coords.append((x, y))

        return grid_coords


#######################################.#######################################
#                                                                             #
#                               CALCULATE CENTER                              #
#                                                                             #
###############################################################################
def calculate_center(output_size):
    """
    Determines what we referred to as the 'spatial center' of a 1D or 2D space.
    Returns the index or indices of the spatial center.
    """
    if isinstance(output_size, (tuple, list, np.ndarray)):
        if len(output_size) != 2:
            raise ValueError("output_size has too many dimensions.")
        c1 = calculate_center(output_size[0])
        c2 = calculate_center(output_size[1])
        return c1, c2
    else:
        return (output_size - 1)//2


if __name__ == "__main__":
    model = models.vgg16()
    model_name = 'vgg16'
    xn = yn = 227
    spatial_index = (11, 11)
    rf_blen_ratios = [3/4, 3/8, 3/16, 3/32]
    rf_blen_ratio_strs = ['3/4', '3/8', '3/16', '3/32']
    aspect_ratios = [1/2, 1/5, 1/10]
    laa = 0.5
    fgval = 1.0
    bgval = 0.5
    
    bar_locations = RfGrid(model, (yn, xn))
    converter = SpatialIndexConverter(model, (yn, xn))
    layer_indices, rf_sizes = get_rf_sizes(model, (yn, xn))
    conv_output_shapes = get_conv_output_shapes(model, (yn, xn))
    num_layers = len(layer_indices)
    
    def box_to_center(box):
        y_min, x_min, y_max, x_max = box
        xc = (x_min + x_max)//2
        yc = (y_min + y_max)//2
        return xc, yc
    
    pdf_dir = Path(__file__).parent.parent.parent.parent.joinpath(f'results/rf_mapping/')
    pdf_path = os.path.join(pdf_dir, f"{model_name}.pdf")
    with PdfPages(pdf_path) as pdf:
        for i, rf_blen_ratio in enumerate(rf_blen_ratios):
            for aspect_ratio in aspect_ratios:
                plt.figure(figsize=(4*num_layers, 5))
                plt.suptitle(f"Bar Length = {rf_blen_ratio_strs[i]} M, aspect_ratio = {aspect_ratio}", fontsize=24)
                for conv_i, layer_index in enumerate(layer_indices):

                    spatial_index = np.array(conv_output_shapes[conv_i][-2:])
                    spatial_index = calculate_center(spatial_index)
                    box = converter.convert(spatial_index, layer_index, 0, is_forward=False)
                    xc, yc = box_to_center(box)
                    
                    rf_size = rf_sizes[conv_i][0]
                    blen = round(rf_blen_ratio * rf_size)
                    bwid = round(aspect_ratio * blen)
                    grid_spacing = blen/2
                    grid_coords = bar_locations.get_grid_coords(layer_index, spatial_index, grid_spacing)
                    grid_coords_np = np.array(grid_coords)
                    bar = draw_bar(xn, yn, xc, yc, 30, blen, bwid, laa, fgval, bgval)

                    plt.subplot(1, num_layers, conv_i+1)
                    plt.imshow(bar, cmap='gray', vmin=0, vmax=1)
                    plt.title(f"conv{conv_i + 1}", fontsize=20)
                    plt.plot(grid_coords_np[:, 0], grid_coords_np[:, 1], 'k.')
                    rect = make_box(box, linewidth=2)
                    boundary = 10
                    plt.xlim([box[1] - boundary, box[3] + boundary])
                    plt.ylim([box[0] - boundary, box[2] + boundary])
                    ax = plt.gca()
                    ax.invert_yaxis()
                    ax.add_patch(rect)

                pdf.savefig()
                plt.show()
                plt.close()

