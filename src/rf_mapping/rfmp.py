"""
Receptive field mapping paradigms.

Tony Fu, July 8, 2022
"""
import sys
import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from spatial import (get_conv_output_shapes,
                     calculate_center,
                     get_rf_sizes,
                     RfGrid,
                     SpatialIndexConverter,)
from image import make_box, preprocess_img_to_tensor
from hook import ConvUnitCounter
from stimulus import draw_bar
from files import delete_all_npy_files
import constants as c


#######################################.#######################################
#                                                                             #
#                                RF Mapper Base                               #
#                                                                             #
###############################################################################
class RfMapper:
    def __init__(self, model, conv_i, image_shape):
        """
        The base class of receptive fields mapping of a single convolutional
        layer.
        
        Parameters
        ----------
        model : torchvision.Module
            The pretrained neural network.
        conv_i : int
            The index of the convolutional layer starting from zero. That is,
            Conv1 should be 0. 
        image_shape : (int, int)
            The dimension of the image in (yn, xn) format (pix).
        """
        self.model = copy.deepcopy(model)
        self.conv_i = conv_i
        self.yn, self.xn = image_shape
        print(f"The RF mapper is for Conv{conv_i + 1} (not Conv{conv_i}) "
              f"with input shape (yn = {self.yn}, xn = {self.xn}).")

        # Get basic info about the conv layer.
        layer_indices, rf_sizes = get_rf_sizes(model, (self.yn, self.xn),
                                               layer_type=nn.Conv2d)
        self.rf_size = rf_sizes[conv_i][0]
        self.num_units = self._get_num_units()
        self.layer_idx = layer_indices[conv_i]  # See hook.py module for 
                                                # indexing convention.
        
        # Locate spatial center of the layer's output.
        self.output_shape = self._get_output_shape()
        self.output_yc, self.output_xc = calculate_center(self.output_shape)
        
        # Locate RF in the pixel space.
        self.converter = SpatialIndexConverter(model, (self.yn, self.xn))
        self.box = self.converter.convert((self.output_yc, self.output_xc),
                                          self.layer_idx, 0, is_forward=False)
        self.box_yc, self.box_xc = self._get_box_center(self.box)

    def _get_num_units(self):
        """Finds how many unique units/kernels are in this layer."""
        unit_counter = ConvUnitCounter(self.model)
        _, nums_units = unit_counter.count()
        return nums_units[self.conv_i]
    
    def _get_output_shape(self):
        """Finds the shape of output layer in (yn, xn) format."""
        conv_output_shapes = get_conv_output_shapes(self.model,
                                                    (self.yn, self.xn))
        return np.array(conv_output_shapes[self.conv_i][-2:])
    
    def _get_box_center(self, box):
        """Finds the center coordinates of the box in (y, x) format."""
        y_min, x_min, y_max, x_max = box
        xc = (x_min + x_max)//2
        yc = (y_min + y_max)//2
        return yc, xc

    def _print_progress(self, num_stimuli):
        """Prints/updates num_stimuli without printing a new line everytime."""
        sys.stdout.write('\r')
        sys.stdout.write(f"num_stimuli = {num_stimuli}")
        sys.stdout.flush()

    def _truncated_model(self, x, model, layer_index):
        """
        Returns the output of the specified layer without forward passing to
        the subsequent layers.

        Parameters
        ----------
        x : torch.tensor
            The input. Should have dimension (1, 3, 2xx, 2xx).
        model : torchvision.model.Module
            The neural network (or the layer if in a recursive case).
        layer_index : int
            The index of the layer, the output of which will be returned. The
            indexing excludes container layers.

        Returns
        -------
        y : torch.tensor
            The output of layer with the layer_index.
        layer_index : int
            Used for recursive cases. Should be ignored.
        """
        # If the layer is not a container, forward pass.
        if (len(list(model.children())) == 0):
            return model(x), layer_index - 1
        else:  # Recurse otherwise.
            for sublayer in model.children():
                x, layer_index = self._truncated_model(x, sublayer, layer_index)
                if layer_index < 0:  # Stop at the specified layer.
                    return x, layer_index
    

#######################################.#######################################
#                                                                             #
#                                BAR RF MAPPER                                #
#                                                                             #
###############################################################################
class BarRfMapper(RfMapper):
    def __init__(self, model, conv_i, image_shape):
        """
        The class of receptive fields mapping of a single convolutional layer
        using bar stimuli. For each unit (a.k.a. kernel), only the response of
        the spatial center is recorded.
        
        Parameters
        ----------
        model : torchvision.Module
            The neural network.
        conv_i : int
            The index of the convolutional layer starting from zero. That is,
            Conv1 should be 0. 
        image_shape : (int, int)
            The dimension of the image in (yn, xn) format (pix).
        """
        super().__init__(model, conv_i, image_shape)
        self.grid_calculator = RfGrid(model, (self.yn, self.xn))
        self.cumulate_threshold = None

        # Array initializations
        self.weighted_bar_sum = np.zeros((self.num_units, self.yn, self.xn))
        self.threshold_bar_sum = np.zeros((self.num_units, self.yn, self.xn))
        self.center_only_bar_sum = np.zeros((self.num_units, self.yn, self.xn))

    def _weighted_cumulate(self, new_bar, bar_sum, unit, response):
        """
        Adds the new_bar, weighted by the unit's response to that bar, to the
        cumulative bar map.
        
        Parameters
        ----------
        new_bar : numpy.array
            The new bar.
        bar_sum : numpy.array
            The cumulated weighted sum of all previous bars. This is modified
            in-place.
        unit : int
            The unit's number.
        response : float
            The unit's response (spatial center only) to the new bar.
        """
        bar_sum[unit, :, :] += new_bar * response

    def _threshold_cumulate(self, new_bar, bar_sum, unit, response):
        """
        Adds to a cumulative map only bars that gave a threshold response.
        
        Parameters
        ----------
        See _weighted_cumulate() for repeated parameters.
        threshold : float
            The unit's response (spatial center only) to the new bar.
        """
        if response > self.cumulate_threshold:
            bar_sum[unit, :, :] += new_bar
    
    def _center_only_cumulate(self, bar_sum, unit, response):
        """
        Adds to a cumulative map only the center points of bars that gave a
        threshold response.
        
        Parameters
        ----------
        See _weighted_cumulate() for repeated parameters.
        bar_sum : numpy.array
            The cumulated weighted sum of all previous bar centers. This is
            modified in-place.
        """
        if response > self.cumulate_threshold:
            bar_sum[unit, self.box_yc, self.box_xc] += response
    
    def _update_bar_sums(self, new_bar, responses):
        """Updates all three cumulate maps at once for all units."""
        for unit in range(self.num_units):
            self._weighted_cumulate(new_bar, self.weighted_bar_sum, unit, responses[unit])
            self._threshold_cumulate(new_bar, self.threshold_bar_sum, unit, responses[unit])
            self._center_only_cumulate(self.center_only_bar_sum, unit, responses[unit])
            
    def _get_center_responses(self, input):
        """Gets the responses of the spatial centers of all units."""
        input_tensor = preprocess_img_to_tensor(input)
        y, _ = self._truncated_model(input_tensor, self.model, self.layer_idx)
        return y[0, :, self.output_yc, self.output_xc].cpu().detach().numpy()

    def map(self):
        raise NotImplementedError("The map method is not implemented.")
    

#######################################.#######################################
#                                                                             #
#                            RF MAPPING PARADIGM 4a                           #
#                                                                             #
###############################################################################
class BarRfMapperP4a(BarRfMapper):
    def __init__(self, model, conv_i, image_shape):
        super().__init__(model, conv_i, image_shape)

        # Bar parameters
        self.rf_blen_ratios = [3/4, 3/8, 3/16, 3/32]
        self.rf_blen_ratio_strs = ['3/4', '3/8', '3/16', '3/32']
        self.aspect_ratios = [1/2, 1/5, 1/10]
        self.thetas = np.arange(0, 180, 22.5)
        self.fgval_bgval_pairs = [(1, -1), (-1, 1)]
        self.laa = 0.5  # anti-alias thickness for bar stimuli

        # Mapping parameters
        self.cumulate_threshold = 1
        self.DEBUG = False

        # Array initializations
        self.all_responses = np.zeros((self.num_units,
                                       len(self.rf_blen_ratios),
                                       len(self.aspect_ratios),
                                       len(self.thetas),
                                       2))

    def set_debug(self, debug):
        """
        If debug is set to True, the number of bar locations is reduced
        significantly in order to allow the map() method to run at much faster
        rate. This is mainly to access if the bar mapper works, and the results
        data should not be used.
        """
        self.DEBUG = debug

    def _map(self, animation=False, unit=None, cumulate_mode=None, bar_sum=None):
        num_stimuli = 0
        for blen_i, rf_blen_ratio in enumerate(self.rf_blen_ratios):
            for bwid_i, aspect_ratio in enumerate(self.aspect_ratios):
                for theta_i, theta in enumerate(self.thetas):
                    for val_i, (fgval, bgval) in enumerate(self.fgval_bgval_pairs):
                        # Some bar parameters
                        blen = round(rf_blen_ratio * self.rf_size)
                        bwid = round(aspect_ratio * blen)
                        grid_spacing = blen/2
                        
                        # Get grid coordinates.
                        grid_coords = self.grid_calculator.get_grid_coords(self.layer_idx, (self.output_yc, self.output_xc), grid_spacing)
                        grid_coords_np = np.array(grid_coords)

                        # Create bars.
                        for grid_coord_i, (xc, yc) in enumerate(grid_coords_np):
                            if self.DEBUG and grid_coord_i > 10:
                                break

                            new_bar = draw_bar(self.xn, self.yn, xc, yc, theta, blen, bwid, self.laa, fgval, bgval)
                            center_responses = self._get_center_responses(new_bar)
                            center_responses[center_responses < 0] = 0  # ReLU
                            self.all_responses[:, blen_i, bwid_i, theta_i, val_i] += center_responses.copy()

                            num_stimuli += 1

                            if not animation:
                                self._print_progress(num_stimuli)
                                self._update_bar_sums(self, new_bar, center_responses)
                            
                            else:
                                if cumulate_mode == 'weighted':
                                    self._weighted_cumulate(new_bar, bar_sum, unit, center_responses[unit])
                                elif cumulate_mode == 'threshold':
                                    self._threshold_cumulate(new_bar, bar_sum, unit, center_responses[unit])
                                elif cumulate_mode == 'center_only':
                                    self._center_only_cumulate(bar_sum, unit, center_responses[unit])
                                else:
                                    raise ValueError(f"cumulate_mode: {cumulate_mode} is not supported.")
                                yield bar_sum[unit], center_responses[unit], num_stimuli


    def map(self):
        self._map(animation=False)
        return self.all_responses

    def animate(self, unit, cumulate_mode='weighted'):
        bar_sum = np.zeros((self.num_units, self.yn, self.xn))
        return self._map(animation=True, unit=unit, cumulate_mode=cumulate_mode, bar_sum=bar_sum)

    def plot_one_unit(self, cumulate_mode, unit):
        if cumulate_mode == 'weighted':
            bar_sum = self.weighted_bar_sum
        elif cumulate_mode == 'threshold':
            bar_sum = self.threshold_bar_sum
        elif cumulate_mode == 'center_only':
            bar_sum = self.center_only_bar_sum

        plt.figure(figsize=(25, 5))
        plt.suptitle(f"RF mapping with bars no.{unit}", fontsize=20)
        
        plt.subplot(1, 5, 1)
        plt.imshow(bar_sum[unit, :, :], cmap='gray')
        plt.title("Cumulated bar maps")
        boundary = 10
        plt.xlim([self.box[1] - boundary, self.box[3] + boundary])
        plt.ylim([self.box[0] - boundary, self.box[2] + boundary])
        rect = make_box(self.box, linewidth=2)
        ax = plt.gca()
        ax.add_patch(rect)
        ax.invert_yaxis()
        
        plt.subplot(1, 5, 2)
        blen_tuning = np.mean(self.all_responses[unit,...], axis=(1,2,3))
        blen_std = np.mean(self.all_responses[unit,...], axis=(1,2,3))/math.sqrt(self.num_units)
        plt.errorbar(self.rf_blen_ratios, blen_tuning, yerr=blen_std)
        plt.title("Bar length tuning")
        plt.xlabel("blen/RF ratio")
        plt.ylabel("avg response")
        plt.grid()
        
        plt.subplot(1, 5, 3)
        bwid_tuning = np.mean(self.all_responses[unit,...], axis=(0,2,3))
        bwid_std = np.mean(self.all_responses[unit,...], axis=(0,2,3))/math.sqrt(self.num_units)
        plt.errorbar(self.aspect_ratios, bwid_tuning, yerr=bwid_std)
        plt.title("Bar width tuning")
        plt.xlabel("aspect ratio")
        plt.ylabel("avg response")
        plt.grid()

        plt.subplot(1, 5, 4)
        theta_tuning = np.mean(self.all_responses[unit,...], axis=(0,1,3))
        theta_std = np.mean(self.all_responses[unit,...], axis=(0,1,3))/math.sqrt(self.num_units)
        plt.errorbar(self.thetas, theta_tuning, yerr=theta_std)
        plt.title("Theta tuning")
        plt.xlabel("theta")
        plt.ylabel("avg response")
        plt.grid()
        
        plt.subplot(1, 5, 5)
        val_tuning = np.mean(self.all_responses[unit,...], axis=(0,1,2))
        val_std = np.mean(self.all_responses[unit,...], axis=(0,1,2))/math.sqrt(self.num_units)
        plt.bar(['white on black', 'black on white'], val_tuning, yerr=val_std, width=0.4)
        plt.title("Contrast tuning")
        plt.ylabel("avg response")
        plt.grid()
    
    def make_pdf(self, pdf_path, cumulate_mode, show=False):
        with PdfPages(pdf_path) as pdf:
            for unit in range(self.num_units):
                self.plot_one_unit(cumulate_mode, unit)
                if show: plt.show()
                pdf.savefig()
                plt.close()


"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
%matplotlib ipympl

model = models.alexnet(pretrained=True)
bm = BarRfMapperP4a(model, 1, (227, 227))
bm.set_debug(True)
a = bm.animate(2)

fig, ax = plt.subplots()

def init_func():
    img = np.zeros((227, 227))
    plt.imshow(img, cmap='gray')

def animate_func(frame):
    plt.imshow(frame[0], cmap='gray')
    plt.title(f"frame {frame[2]}, response = {frame[1]:.2f}")

ani = animation.FuncAnimation(
    fig, animate_func, init_func=init_func, frames=a, interval=300, save_count=0, cache_frame_data=False, repeat=False)

plt.show()
"""
