"""
Functions for receptive field mapping paradigm 4a.

Note: all code assumes that the y-axis points downward.

Tony Fu, July 4th, 2022
"""
import os
import sys

import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('..')
from spatial import (get_conv_output_shapes,
                     calculate_center,
                     get_rf_sizes,
                     RfGrid,
                     SpatialIndexConverter,)
from image import make_box
from stimulus import draw_bar
import constants as c


if __name__ == "__main__":
    model = models.alexnet()
    model_name = 'alexnet'
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
    
    pdf_dir = c.REPO_DIR + f'/results/rfmp4a/'
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

