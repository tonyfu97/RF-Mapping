"""Border-ownership shapes generator.

    Lists of shapes
    ---------------
            standard test (single square)
            overlapping figure test
            C-figures
            size invariance

This module does not contain any neural networks and can only generate stimulus 
arrays. The module can be broken down into 3 sets of functions, separated by 
lines of pound signs #: 
    1. Functions that Draw a Single Stimulus: 
            The base functions that draw a single rectangle, c-shape, or two 
            overlapping rectangles. 
    2. Functions that Make a Stimulus Set:
            They use the base functions to draw a standard set, overlapping set,
            c-figure set, or size-variant set.
    3. Functions that Make Stimulus Sets for a Whole Layer:
            They use the functions above to draw multiple sets of stimuli 
            according to the CNN units' receptive fields. 
           
Reference: 
H. Zhou, H. S. Friedman and R. von der Heydt, "Coding of border ownership in 
    monkey visual cortex," The Journal of neuroscience: the official journal 
    of the Society for Neuroscience, vol. 20, no. 17, pp. 6594-661, 2000. 
"""

# Adapted by Tony Fu from Dr. Wyeth Bair's sinusoidal grating program and Danny
# Burnham's border-ownership project.
# Univesity of Washington Bioengineering, Class of 2022.
# May, 2021

VERSION = 2.0

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm 
from numba import jit 


################# Functions that Draw a Single Stimulus ####################


def draw_rectangle(x0, y0, lx=120, ly=120, angle=math.pi/4, shape_rgb=(0.75, 0.75, 0.75), 
                   xn=227, yn=227, background_rgb=(0.35, 0.35, 0.35)):
    """Draws a stimulus with a rectangle of the given specifications.

    Parameters
    ----------
    x0 : int
        The horizontal coordinate of the center of the rectangle.
    y0 : int
        The vertical coordinate of the center of the rectangle.
    lx : int, default=120
        The horizontal length of the rectangle
    ly : int, default=120
        The vertical length of the rectangle
    angle : float, default=math.pi/4 radians
        The orientation of the rectangle. The angle is in radians and is defined
        counterclockwise, from 0 to pi.
    shape_rgb : tuple of 3 floats, default=(0.75, 0.75, 0.75)
        The color intensity of the rectangle, from 0.0 to 1.0.
    xn : int, default=227
        The horizontal width of the stimulus image
    yn : int, default=227
        The vertical length of the stimulus image
    background_rgb : tuple of 3 floats, default=(0.35, 0.35, 0.35)
        The background color intensity, from 0.0 to 1.0.
    
    Returns
    -------
    A 3D numpy array of dimension 3 x <yn> x <xn>. The area of the rectangle
    will have the corresponding shape_rgb values. The remaining area will have 
    the background_rgb values. 

    Note
    ----
    For the images/stimuli, the x-coordinate points to the right, and the
          y-coordinate points downward. The origin is at the top-left corner. 
    """
    shape_bool = _get_rectangle_bool(x0, y0, lx, ly, angle, xn, yn)
    stimulus = np.empty((3, yn, xn), dtype=float)
    for i in range(3):
        stimulus[i, :, :] = background_rgb[i]
        stimulus[i, shape_bool] = shape_rgb[i]

    return stimulus


def draw_overlap(x0_front, y0_front, x0_back , y0_back, 
                 lx_front=120, ly_front=120, lx_back =200, ly_back =150, 
                 angle_front=math.pi/4, angle_back =math.pi/4, 
                 front_shape_rgb=(0.75, 0.75, 0.75), back_shape_rgb=(0.35, 0.35, 0.35),
                 overlap_misalignment=40, overlap_distance=60, 
                 xn=227, yn=227, background_rgb=(0.1, 0.1, 0.1)):
    """Draws a stimulus with a rectangle of the given specifications.

    Parameters
    ----------
    x0_front : int
        The horizontal coordinate of the center of the front rectangle.
    y0_front : int
        The vertical coordinate of the center of the front rectangle.
    x0_back : int
        The horizontal coordinate of the center of the occluded rectangle.
    y0_back : int
        The vertical coordinate of the center of the occluded rectangle.
    lx_front : int, default=120
        The horizontal length of the front rectangle
    ly_front : int, default=120
        The vertical length of the front rectangle
    lx_back : int, default=200
        The horizontal length of the occluded rectangle
    ly_back : int, default=150
        The vertical length of the occluded rectangle
    angle_front : float, default=math.pi/4
        The counterclockwise rotation of the front rectangle in radians, 
        from 0 to pi.
    angle_back : float, default=math.pi/4
        The counterclockwise rotation of the occluded rectangle in radians, 
        from 0 to pi.
    front_shape_rgb : tuple of 3 floats, default=(0.75, 0.75, 0.75)
        The color intensity of the rectangle, from 0.0 to 1.0.
    back_shape_rgb : tuple of 3 floats, default=(0.35, 0.35, 0.35)
        The color intensity of the occluded rectangle for first stimulus, 
        from 0.0 to 1.0.
    overlap_misalignment : int, default=40
        The distance between the centers of the two rectangles along their 
        vertical axes. Can be thought of as how 'misaligned' the two are.
    overlap_distance : int, default=60
        The distance between the centers of the two rectangles along their 
        horizontal axes. Can be thought of as the degree of overlapping.
    xn : int, default=227
        The horizontal width of the stimulus image
    yn : int, default=227
        The vertical length of the stimulus image
    background_rgb : tuple of 3 floats, default=(0.1, 0.1, 0.1)
        The background color intensity, from 0.0 to 1.0.
    
    Returns
    -------
    A 3D numpy array of dimension 3 x <yn> x <xn>. The area of the front 
    rectangle will have the corresponding front_shape_rgb values. Same goes
    for the occluded rectnagle. The remaining area will have the
    background_rgb values. 

    Note
    ----
    For the images/stimuli, the x-coordinate points to the right, and the
          y-coordinate points downward. The origin is at the top-left corner. 
    """
    stimulus = np.empty((3, yn, xn), dtype=float)

    back_shape_bool  = _get_rectangle_bool(x0_back, y0_back, lx_back, ly_back, 
                                           angle_back, xn, yn)
    front_shape_bool = _get_rectangle_bool(x0_front, y0_front, lx_front, ly_front, 
                                           angle_front, xn, yn)

    for i in range(3):
        stimulus[i, :, :] = background_rgb[i]
        stimulus[i, back_shape_bool ] = back_shape_rgb[i]
        stimulus[i, front_shape_bool] = front_shape_rgb[i]

    return stimulus


def draw_c_figure(x0, y0, lx=120, ly=240, l_inside=120, angle=math.pi/4, 
                  shape_rgb=(0.75, 0.75, 0.75), side="left", xn=227, yn=227, 
                  background_rgb=(0.35, 0.35, 0.35)):
    """Draws a stimulus with a c-figure of the given specifications.

    Parameters
    ----------
    x0 : int
        The horizontal coordinate of the center of the c-figure.
    y0 : int
        The vertical coordinate of the center of the c-figure.
    lx : int, default=120
        The maximum horizontal width of the c-figure
    ly : int, default=240
        The vertical length of the c-figure
    l_inside : int, default=120
        The vertical length of the area to be carved out to create a c-figure out
        of a larger rectangle. Throws ValueError if l_inside >= ly. 
    angle : float, default=math.pi/4 radians
        The orientation of the c-figure. The angle is in radians and is defined
        counterclockwise, from 0 to pi.
    shape_rgb : tuple of 3 floats, default=(0.75, 0.75, 0.75)
        The color intensity of the c-figure, from 0.0 to 1.0.
    side : str, default="left"
        The side of the recangle to carve out in order to create a c-figure.
    xn : int, default=227
        The horizontal width of the stimulus image
    yn : int, default=227
        The vertical length of the stimulus image
    background_rgb : tuple of 3 floats, default=(0.35, 0.35, 0.35)
        The background color intensity, from 0.0 to 1.0.
    
    Returns
    -------
    A 3D numpy array of dimension 3 x <yn> x <xn>. The area of the c-figure
    will have the corresponding shape_rgb values. The remaining area will have 
    the background_rgb values. 

    Note
    ----
    For the images/stimuli, the x-coordinate points to the right, and the
          y-coordinate points downward. The origin is at the top-left corner. 
    """
    shape_bool = _get_c_figure_bool(x0, y0, lx, ly, angle, l_inside, side, xn, yn)
    stimulus = np.empty((3, yn, xn), dtype=float)

    for i in range(3):
        stimulus[i, :, :] = background_rgb[i]
        stimulus[i, shape_bool] = shape_rgb[i]

    return stimulus


def plot_one_stimulus(stimulus, grid=False):
    """Plot the stimulus numpy array.

    Parameters
    ----------
    stimulus : numpy.array
        The 3 x <yn> x <xn> array containing the stimulus rgb values. 
    grid : boolean, default=False
        Whether to display the grid or not.
    """
    plt.figure()
    img2display = stimulus.transpose((1,2,0))
    plt.imshow(img2display, cmap='gray', vmin=0, vmax=1)
    if (grid):
        ax = plt.gca();
        ax.grid(color='r', linestyle='--', linewidth=1)
    plt.show()


@jit(nopython=True)  # Numba accelerator
def _get_rectangle_bool(x0, y0, lx, ly, angle, xn=227, yn=227):
    """ Internal Function. Returns a boolean map in which the area of the rectangle 
    is True, and the area of the remaining area is False."""

    if (0 > angle > math.pi):
        raise ValueError("Angle must be from 0 to pi.")

    rectangle_bool = np.full((yn, xn), False)

    for i in range(0, xn):
        for j in range(0, yn):
            if (angle == 0):
                border1 = (j > (y0 - (ly/2))) # top border
                border2 = (j < (y0 + (ly/2))) # bottom border
                border3 = (i > (x0 - (lx/2))) # left border
                border4 = (i < (x0 + (lx/2))) # right border
            elif (angle == math.pi/2):
                border1 = (i > (x0 - (ly/2))) # left (after pi/2 rotation)
                border2 = (i < (x0 + (ly/2))) # right
                border4 = (j < (y0 + (lx/2))) # bottom
                border3 = (j > (y0 - (lx/2))) # top
            elif (angle == math.pi):
                border1 = (j < (y0 + (ly/2))) # bottom (after pi rotation)
                border2 = (j > (y0 - (ly/2))) # top
                border3 = (i > (x0 - (lx/2))) # left
                border4 = (i < (x0 + (lx/2))) # right
            elif (0 < angle < math.pi/2):
                border1 = (j > (-(1/math.tan(math.pi/2 - angle))*i 
                                + x0*(1/math.tan(math.pi/2 - angle)) 
                                + y0 - ly/(2*math.cos(angle)))) # top->left
                border2 = (j < (-(1/math.tan(math.pi/2 - angle))*i 
                                + x0*(1/math.tan(math.pi/2 - angle)) 
                                + y0 + ly/(2*math.cos(angle)))) # bottom->right
                border3 = (j < (math.tan(math.pi/2 - angle)*i 
                                - x0*math.tan(math.pi/2 - angle) 
                                + y0 + lx/(2*math.sin(angle)))) # left->bottom
                border4 = (j > (math.tan(math.pi/2 - angle)*i
                                - x0*math.tan(math.pi/2 - angle)
                                + y0 - lx/(2*math.sin(angle)))) # right->top
            elif (math.pi/2 < angle < math.pi):
                angle_new = math.pi - angle # (after >pi/2 rotation)
                border1 = (j > (1/math.tan(math.pi/2 - angle_new)*i 
                                - x0*(1/math.tan(math.pi/2 - angle_new)) 
                                + y0 - ly/(2*math.cos(angle_new)))) # left->bottom
                border2 = (j < (1/math.tan(math.pi/2 - angle_new)*i 
                                - x0*(1/math.tan(math.pi/2 - angle_new)) 
                                + y0 + ly/(2*math.cos(angle_new)))) # right->top
                border3 = (j < (-math.tan(math.pi/2 - angle_new)*i 
                                + x0*math.tan(math.pi/2 - angle_new) 
                                + y0 + lx/(2*math.sin(angle_new)))) # bottom->right
                border4 = (j > (-math.tan(math.pi/2 - angle_new)*i
                                + x0*math.tan(math.pi/2 - angle_new)
                                + y0 - lx/(2*math.sin(angle_new)))) # top->left
            if border1 and border2 and border3 and border4:
                rectangle_bool[j, i] = True

    return rectangle_bool


def _get_c_figure_bool(x0, y0, lx, ly, angle, l_inside, side, xn=227, yn=227):
    """Internal function. Returns a boolean map in which the area of the c-
    figure is True, and the area of the remaining area is False."""
    if (l_inside >= ly):
        raise ValueError("l_inside exceeds ly. This will result in a rectangle instead of a c-figure.")

    c_figure_bool = _get_rectangle_bool(x0, y0, lx, ly, angle, xn, yn)

    if (side == "left"):
        sign = 1;
    else: # right
        sign = -1;

    if (0 < angle < math.pi/2) or (math.pi/2 < angle < math.pi):
        inner_bool = _get_rectangle_bool(x0 - lx*math.cos(angle)/2*sign, 
                                         y0 + lx*math.sin(angle)/2*sign, 
                                         lx, l_inside, angle, xn, yn)
    elif (angle == 0 or angle == math.pi):
        if (angle == math.pi):
            sign = -sign
        inner_bool = _get_rectangle_bool(x0 - lx*sign/2, y0, lx, l_inside, angle, xn, yn)
    elif (angle == math.pi/2):
        inner_bool = _get_rectangle_bool(x0, y0 + lx*sign/2, l_inside, lx, 0, xn, yn)
    
    # Carve an area out of the rectangle to create a c-figure.
    c_figure_bool[inner_bool] = False
    return c_figure_bool      


# Test
def _test_drawings_functions():
    # Set the parameters:
    x0 = 50
    y0 = 100
    x0_front = 50
    y0_front = 100
    x0_back = 120
    y0_back = 80
    xn = 227
    yn = 224

    # 1. draw_rectangle()
    rectangle_stim = draw_rectangle(x0, y0, xn=xn, yn=yn)
    plot_one_stimulus(rectangle_stim, grid=True)

    # 2. draw_overlap()
    overlap_stim = draw_overlap(x0_front, y0_front, x0_back , y0_back, 
                                xn=xn, yn=yn)
    plot_one_stimulus(overlap_stim, grid=True)

    # 3. draw_c_figure()
    c_figure_stim = draw_c_figure(x0, y0, xn=xn, yn=yn)
    plot_one_stimulus(c_figure_stim, grid=True)

if __name__ == '__main__':
    _test_drawings_functions()

################### Functions that Make a Stimulus Set #####################


def make_standard_set(RF_x, RF_y, RF_r, lx=120, ly=120, angle=math.pi/4, 
                      shape_rgb=(0.75, 0.75, 0.75), xn=227, yn=227, 
                      background_rgb=(0.35, 0.35, 0.35)):
    """Make a set of 4 overlap figures containing rectangles for standard
    test as shown in Figure 2 in Zhou et al.'s (2000) paper.

    Parameters
    ----------
    RF_x : int
        The horizontal coordinate of the receptive field (RF) on the image 
        (the center is at the top-left).
    RF_y : int
        The vertical coordinate of the receptive field (RF) on the image 
        (the center is at the top-left).
    RF_r : int
        The size of the receptive field (RF).
    lx : int, default=120
        The horizontal length of the rectangle
    ly : int, default=120
        The vertical length of the rectangle
    angle : float, default=math.pi/4 radians
        The orientation of the rectangle. The angle is in radians and is 
        defined counterclockwise, from 0 to pi.
    shape_rgb : tuple of 3 floats, default=(0.75, 0.75, 0.75)
        The color intensity of the rectangle, from 0.0 to 1.0.
    xn : int, default=227
        The horizontal width of the stimulus image
    yn : int, default=227
        The vertical length of the stimulus image
    background_rgb : tuple of 3 floats, default=(0.35, 0.35, 0.35)
        The background color intensity, from 0.0 to 1.
    
    Returns
    -------
    A 4D numpy array of dimension 4 x 3 x <yn> x <xn>. The area of the 
    rectangle will have the corresponding shape_rgb values. The remaining 
    area will have the background_rgb values. There are four conditions per
    standard set. 

    Notes
    -----
    1. For the images/stimuli, the x-coordinate points to the right, and 
       the y-coordinate points downward. The origin is at the top-left 
       corner. 
    2. The "shape_rgb" is actually the background RGB color for 
       conditions B and C. Similarly, the "background_rgb" is actually 
       the rectangle's RGB values for conditions B and C. 
    """
    vertical_shift, horizontal_shift = _find_shift_away_RF(lx, ly, angle)
        
    # Stimulus A and C (shift down and left)
    x0_AC = RF_x - horizontal_shift
    y0_AC = RF_y + vertical_shift
    stim_A = draw_rectangle(x0_AC, y0_AC, lx=lx, ly=ly, angle=angle, 
                            shape_rgb=shape_rgb, xn=xn, yn=yn, 
                            background_rgb=background_rgb)
    stim_C = draw_rectangle(x0_AC, y0_AC, lx=lx, ly=ly, angle=angle, 
                            shape_rgb=background_rgb, xn=xn, yn=yn, 
                            background_rgb=shape_rgb)

    # Stimulus B and D (shift up and right)
    x0_BD = RF_x + horizontal_shift
    y0_BD = RF_y - vertical_shift
    stim_B = draw_rectangle(x0_BD, y0_BD, lx=lx, ly=ly, angle=angle, 
                            shape_rgb=background_rgb, xn=xn, yn=yn, 
                            background_rgb=shape_rgb)
    stim_D = draw_rectangle(x0_BD, y0_BD, lx=lx, ly=ly, angle=angle, 
                            shape_rgb=shape_rgb, xn=xn, yn=yn, 
                            background_rgb=background_rgb)

    # Stack all four stimuli such that the final output array has the 
    # dimension of [4, 3, yn, xn].
    return np.stack((stim_A.copy(), stim_C.copy(), stim_B.copy(), stim_D.copy()), axis=0)


def make_overlap_set(RF_x, RF_y, RF_r, lx_front=120, ly_front=120, 
                     lx_back =200, ly_back =150, angle=math.pi/4,
                     front_shape_rgb=(0.75, 0.75, 0.75), 
                     back_shape_rgb=(0.35, 0.35, 0.35),
                     overlap_misalignment=40, overlap_distance=60, 
                     xn=227, yn=227, background_rgb=(0.1, 0.1, 0.1)):
    """Make a set of 4 figures containing overlapping shapes as shown
    in Figure 3 in Zhou et al. (2000). This function makes sure that the 
    back shape is always to the right side of the front shape.


    Parameters
    ----------
    RF_x : int
        The horizontal coordinate of the receptive field (RF) on the image 
        (the center is at the top-left).
    RF_y : int
        The vertical coordinate of the receptive field (RF) on the image 
        (the center is at the top-left).
    RF_r : int
        The size of the receptive field (RF).
    lx_front : int, default=120
        The horizontal length of the front rectangle
    ly_front : int, default=120
        The vertical length of the front rectangle
    lx_back : int, default=200
        The horizontal length of the occluded rectangle
    ly_back : int, default=150
        The vertical length of the occluded rectangle
    angle: float, default=math.pi/4
        The counterclockwise rotation of both rectangles in radians, 
        from 0 to pi.
    front_shape_rgb : tuple of 3 floats, default=(0.75, 0.75, 0.75)
        The color intensity of the rectangle, from 0.0 to 1.0.
    back_shape_rgb : tuple of 3 floats, default=(0.35, 0.35, 0.35)
        The color intensity of the occluded rectangle for first stimulus, 
        from 0.0 to 1.0.
    overlap_misalignment : int, default=40
        The distance between the centers of the two rectangles along their 
        vertical axes. Can be thought of as how 'misaligned' the two are.
    overlap_distance : int, default=60
        The distance between the centers of the two rectangles along their 
        horizontal axes. Can be thought of as the degree of overlapping.
    xn : int, default=227
        The horizontal width of the stimulus image
    yn : int, default=227
        The vertical length of the stimulus image
    background_rgb : tuple of 3 floats, default=(0.1, 0.1, 0.1)
        The background color intensity, from 0.0 to 1.0.
    
    Returns
    -------
    A 4D numpy array of dimension 4 x 3 x <yn> x <xn>. The area of the 
    rectangles will have the corresponding shape_rgb values. The remaining 
    area will have the background_rgb values. There are four conditions per
    overlap set. 

    Notes
    -----
    1. For the images/stimuli, the x-coordinate points to the right, and 
       the y-coordinate points downward. The origin is at the top-left 
       corner.
    2. The "front_shape_rgb" is actually the RGB value of the occluded 
       rectangle for conditions B and C. Similarly, the "back_shape_rgb" is 
       actually the front rectangle's RGB values for conditions B and C. 
    """
    vertical_shift_front, horizontal_shift_front,vertical_shift_back ,horizontal_shift_back = \
                _find_shift_away_RF_overlap(lx_front, ly_front, lx_back, ly_back, angle,
                                            overlap_misalignment, overlap_distance)
            
    # Stimulus A and C (front shape is shifted down and left):
    x0_AC_front = RF_x - horizontal_shift_front
    y0_AC_front = RF_y + vertical_shift_front
    x0_AC_back = RF_x + horizontal_shift_back
    y0_AC_back = RF_y - vertical_shift_back
    stim_A = draw_overlap(x0_AC_front, y0_AC_front, x0_AC_back , y0_AC_back, 
                          lx_front=lx_front, ly_front=ly_front, 
                          lx_back=lx_back, ly_back=ly_back, 
                          angle_front=angle, angle_back=angle, 
                          front_shape_rgb=front_shape_rgb, 
                          back_shape_rgb=back_shape_rgb,
                          overlap_misalignment=overlap_misalignment, 
                          overlap_distance=overlap_distance, 
                          xn=xn, yn=yn, background_rgb=background_rgb)
    stim_C = draw_overlap(x0_AC_front, y0_AC_front, x0_AC_back , y0_AC_back, 
                          lx_front=lx_front, ly_front=ly_front, 
                          lx_back=lx_back, ly_back=ly_back, 
                          angle_front=angle, angle_back=angle, 
                          front_shape_rgb=back_shape_rgb, 
                          back_shape_rgb=front_shape_rgb,
                          overlap_misalignment=overlap_misalignment, 
                          overlap_distance=overlap_distance, 
                          xn=xn, yn=yn, background_rgb=background_rgb)

    # Stimulus B and D (front shape is shifted up and right):
    # Draw the back shape first...
    x0_BD_front = RF_x + horizontal_shift_front
    y0_BD_front = RF_y - vertical_shift_front
    x0_BD_back = RF_x - horizontal_shift_back
    y0_BD_back = RF_y + vertical_shift_back
    stim_B = draw_overlap(x0_BD_front, y0_BD_front, x0_BD_back , y0_BD_back, 
                          lx_front=lx_front, ly_front=ly_front, 
                          lx_back=lx_back, ly_back=ly_back, 
                          angle_front=angle, angle_back=angle, 
                          front_shape_rgb=back_shape_rgb, 
                          back_shape_rgb=front_shape_rgb,
                          overlap_misalignment=overlap_misalignment, 
                          overlap_distance=overlap_distance, 
                          xn=xn, yn=yn, background_rgb=background_rgb)
    stim_D = draw_overlap(x0_BD_front, y0_BD_front, x0_BD_back , y0_BD_back, 
                          lx_front=lx_front, ly_front=ly_front, 
                          lx_back=lx_back, ly_back=ly_back, 
                          angle_front=angle, angle_back=angle, 
                          front_shape_rgb=front_shape_rgb, 
                          back_shape_rgb=back_shape_rgb,
                          overlap_misalignment=overlap_misalignment, 
                          overlap_distance=overlap_distance, 
                          xn=xn, yn=yn, background_rgb=background_rgb)

    # Stack all four stimuli such that the final output array has the 
    # dimension of [4, 3, yn, xn].
    return np.stack((stim_A.copy(), stim_C.copy(), stim_B.copy(), stim_D.copy()), axis=0)


def make_c_figure_set(RF_x, RF_y, RF_r, lx=120, ly=240, l_inside=120, 
                      angle=math.pi/4, shape_rgb=(0.75, 0.75, 0.75), 
                      xn=227, yn=227, background_rgb=(0.35, 0.35, 0.35)):
    """Make a set of 4 c-shaped figures as shown in Figures 23, 24, and 26 
    in Zhou et al. (2000).

    Parameters
    ----------
    RF_x : int
        The horizontal coordinate of the receptive field (RF) on the image 
        (the center is at the top-left).
    RF_y : int
        The vertical coordinate of the receptive field (RF) on the image 
        (the center is at the top-left).
    RF_r : int
        The size of the receptive field (RF).
    lx : int, default=120
        The maximum horizontal width of the c-figure
    ly : int, default=240
        The vertical length of the c-figure
    l_inside : int, default=120
        The vertical length of the area to be carved out to create a c-figure
        out of a larger rectangle. Throws ValueError if l_inside >= ly. 
    angle : float, default=math.pi/4 radians
        The orientation of the c-figure. The angle is in radians and is
        defined counterclockwise, from 0 to pi.
    shape_rgb : tuple of 3 floats, default=(0.75, 0.75, 0.75)
        The color intensity of the c-figure, from 0.0 to 1.0.
    xn : int, default=227
        The horizontal width of the stimulus image
    yn : int, default=227
        The vertical length of the stimulus image
    background_rgb : tuple of 3 floats, default=(0.35, 0.35, 0.35)
        The background color intensity, from 0.0 to 1.0.
    
    Returns
    -------
    A 4D numpy array of dimension 4 x 3 x <yn> x <xn>. The area of the 
    c-figure will have the corresponding shape_rgb values. The remaining 
    area will have the background_rgb values. There are four conditions per
    overlap set. 

    Notes
    -----
    1. For the images/stimuli, the x-coordinate points to the right, and 
       the y-coordinate points downward. The origin is at the top-left 
       corner.
    2. The "shape_rgb" is actually the background RGB color for 
       conditions B and C. Similarly, the "background_rgb" is actually 
       the rectangle's RGB values for conditions B and C. 
    """
    stim_A = draw_c_figure(RF_x, RF_y, lx=lx, ly=ly, l_inside=l_inside, 
                           angle=angle, shape_rgb=background_rgb, side="left",
                           xn=xn, yn=yn, background_rgb=shape_rgb)
    stim_C = draw_c_figure(RF_x, RF_y, lx=lx, ly=ly, l_inside=l_inside, 
                           angle=angle, shape_rgb=shape_rgb, side="left",
                           xn=xn, yn=yn, background_rgb=background_rgb)
    stim_B = draw_c_figure(RF_x, RF_y, lx=lx, ly=ly, l_inside=l_inside, 
                           angle=angle, shape_rgb=shape_rgb, side="right",
                           xn=xn, yn=yn, background_rgb=background_rgb)
    stim_D = draw_c_figure(RF_x, RF_y, lx=lx, ly=ly, l_inside=l_inside, 
                           angle=angle, shape_rgb=background_rgb, side="right",
                           xn=xn, yn=yn, background_rgb=shape_rgb)
    
    # Stack all four stimuli such that the final output array has the 
    # dimension of [4, 3, yn, xn].
    return np.stack((stim_A.copy(), stim_C.copy(), stim_B.copy(), stim_D.copy()), axis=0)


def make_size_set(RF_x, RF_y, RF_r, square_sizes, angle=math.pi/4, 
                  shape_rgb=(0.35, 0.35, 0.35), xn=227, yn=227, 
                  background_rgb=(0.75, 0.75, 0.75)):
    """Make a set of pairs of squares of varying sizes as shown in Figure 9 
    in Zhou et al. (2000).

    Parameters
    ----------
    RF_x : int
        The horizontal coordinate of the receptive field (RF) on the image 
        (the center is at the top-left).
    RF_y : int
        The vertical coordinate of the receptive field (RF) on the image 
        (the center is at the top-left).
    RF_r : int
        The size of the receptive field (RF).
    square_sizes: tuple of int, default=(70, 120, 200)
        The list containing the side lengths of the squares used as stimuli.
    angle : float, default=math.pi/4 radians
        The orientation of the rectangle. The angle is in radians and is 
        defined counterclockwise, from 0 to pi.
    shape_rgb : tuple of 3 floats, default=(0.75, 0.75, 0.75)
        The color intensity of the rectangle, from 0.0 to 1.0.
    xn : int, default=227
        The horizontal width of the stimulus image
    yn : int, default=227
        The vertical length of the stimulus image
    background_rgb : tuple of 3 floats, default=(0.35, 0.35, 0.35)
        The background color intensity, from 0.0 to 1.0.
    
    Returns
    -------
    A 4D numpy array of dimension n x 3 x <yn> x <xn>. The area of the 
    rectangle will have the corresponding shape_rgb values. The remaining 
    area will have the background_rgb values. There are four conditions per
    standard set. 

    Notes
    -----
    1. For the images/stimuli, the x-coordinate points to the right, and 
       the y-coordinate points downward. The origin is at the top-left 
       corner. 
    2. The "shape_rgb" is actually the background RGB color for the second
       condition of each size. Similarly, the "background_rgb" is actually 
       the rectangle's RGB values for the second condition of each size. 
    """
    
    stimulus_set = np.full([len(square_sizes) * 2, 3, yn, xn], 0.0, dtype=float)
                            
    for i, size in enumerate(square_sizes):
        vertical_shift, horizontal_shift = _find_shift_away_RF(size, size, angle)

        x0_1 = RF_x - horizontal_shift
        y0_1 = RF_y + vertical_shift
        stim_1 = draw_rectangle(x0_1, y0_1, lx=size, ly=size, angle=angle,
                                shape_rgb=shape_rgb, xn=xn, yn=yn, 
                                background_rgb=background_rgb)

        x0_2 = RF_x + horizontal_shift
        y0_2 = RF_y - vertical_shift
        stim_2 = draw_rectangle(x0_2, y0_2, lx=size, ly=size, angle=angle,
                                shape_rgb=background_rgb, xn=xn, yn=yn, 
                                background_rgb=shape_rgb)
        
        stimulus_set[i*2  ,:,:,:] = stim_1.copy()
        stimulus_set[i*2+1,:,:,:] = stim_2.copy()

    return stimulus_set


def plot_one_set(stimulus_set, grid=False, figsize=(15, 15), alpha=1, 
                 cmap='gray', vmin=0, vmax=1, square_sizes=(0,0,0), 
                 RF_circ=False, RF_x=0, RF_y=0, RF_r=10):
    """Plot the stimulus array of the set as subplots.
    
    Parameters
    ----------
    stimulus_set : numpy.array
        The 4D array containing 4 conditions (if standard, overlap, or 
        c-figure) or more (if size variance). Each condition has size
        3 x <yn> x <xn>.
    grid : boolean, default=False
        Whether to plot the grid or not.
    figsize : tuple of 2 int, default=(15,15)
        (vertical size, horizontal size) of the figure
    alpha : float, default=1
        The level of transparency. 0 is most transparent; 1 is like regular
        plotting.
    cmap : str, default='cmap'
        Color map style.
    vmin : float, default=0
        The dynamic range's minimum, below which is represented as black.
    vmax : float, default=1.0
        The dynamic range's maximum, above which is represented as white.
    square_sizes : tuple of 3 int, default=(0,0,0)
        The squares' side length for the size-variance set. Will be shown 
        as the titles of the subplots. 
    RF_circ : boolean, default=False
        Whether to draw the receptive field circle or not.
    RF_x : int, default=0
        The horizontal coordinate of the receptive field (RF) on the image 
        (the center is at the top-left).
    RF_y : int, default=0
        The vertical coordinate of the receptive field (RF) on the image 
        (the center is at the top-left).
    RF_r : int
        The size of the receptive field (RF).
    """
    plt.figure(figsize=figsize)
    set2plot = stimulus_set.transpose((0,2,3,1))

    if (square_sizes == (0,0,0)): # if the set is standard, overlap, or c-figures
        for i, plot_str in enumerate(["A", "C", "B", "D"]):
            plt.subplot(2,2,i+1)
            plt.imshow(set2plot[i,:,:,:], cmap=cmap, alpha=alpha, 
                       vmin=vmin, vmax=vmax)
            plt.title(plot_str)
            ax = plt.gca();

            # add grid
            if (grid):
                ax.grid(color='r', linestyle='--', linewidth=1)
            
            # add receptive field
            if (RF_circ):
                circ = Circle((RF_x, RF_y), RF_r, ec='b',fill=False)
                ax.add_patch(circ)
        plt.show()

    else: # if the set is size-variant
        for i, size in enumerate(square_sizes):
            # Plot #1 for this size
            plt.subplot(2, len(square_sizes), i+1)
            plt.imshow(set2plot[2*i,:,:,:], cmap=cmap, alpha=alpha, 
                       vmin=vmin, vmax=vmax)
            plt.title("size: " + str(size))
            ax = plt.gca();

            # add grid
            if (grid):
                ax.grid(color='r', linestyle='--', linewidth=1)
            
            # add receptive field
            if (RF_circ):
                circ = Circle((RF_x, RF_y), RF_r, ec='b',fill=False)
                ax.add_patch(circ)

            # Plot #2 for this size
            plt.subplot(2, len(square_sizes), i+1+len(square_sizes))
            plt.imshow(set2plot[2*i+1,:,:,:], cmap=cmap, alpha=alpha, 
                       vmin=vmin, vmax=vmax)
            ax = plt.gca();
            
            # add grid
            if (grid):
                ax.grid(color='r', linestyle='--', linewidth=1)
            
            # add receptive field
            if (RF_circ):
                circ = Circle((RF_x, RF_y), RF_r, ec='b',fill=False)
                ax.add_patch(circ)
        plt.show()


def _find_shift_away_RF(lx, ly, angle):
    """Internal function. Calculates the center coordinates of the rectangle 
    such that the receptive field (RF) sits at the center of one the border."""
    # for  0   <= angle <= pi/2 , the RF sits on the right border of the left square.
    # for pi/2 <  angle < pi    , the RF sits on the bottom (now right) border.
    # for  angle == pi, the RF sits on the left (now right) border.
    if (angle == 0 or angle == math.pi):
        vertical_shift = 0
        horizontal_shift = lx/2
    elif (angle == math.pi/2):
        vertical_shift = 0
        horizontal_shift = ly/2
    elif (0 < angle < math.pi/2) or (math.pi/2 < angle < math.pi):
        vertical_shift   = lx*math.sin(angle)/2
        horizontal_shift = lx*math.cos(angle)/2
    # elif (math.pi/2 < angle < math.pi):
    #     horizontal_shift = ly*math.sin(math.pi - angle)/2
    #     vertical_shift   = ly*math.cos(math.pi - angle)/2
    return (vertical_shift, horizontal_shift)


def _find_shift_away_RF_overlap(lx_front, ly_front, lx_back, ly_back, angle,
                                overlap_misalignment, overlap_distance):
    """Calculates the center coordinates of the rectangle such that the 
    receptive field (RF) sits at the center of the border. The shifts
    are (almost) always positive."""
    # if (angle >= 0 and angle <= math.pi/2) or (angle == math.pi):
    if (angle >= 0 and angle <= math.pi):
        # The length of the border is usually the side-length of the front
        # shape, but it will start decreasing as the front shape becomes 
        # exceedingly misaligned with the back shape. 
        if (abs(overlap_misalignment) <= (ly_back - ly_front)):
            # when the border is still inside the back shape
            border_length = ly_front
        elif (overlap_misalignment > (ly_back - ly_front)):
            # when the front shape is too far down (positive misalignment)
            border_length = (ly_front + ly_back)/2 - overlap_misalignment
        else:
            # when the front shape is too far up (negative misalignment)
            border_length = (ly_front + ly_back)/2 + overlap_misalignment 

        # for 0 <= angle <= pi/2 , the RF sits on the right border of the left square.
        # for angle == pi, the RF sits on the left (now right) border.
        if (angle == 0 or angle == math.pi):
            vertical_shift_front = ly_front/2 - border_length/2
            horizontal_shift_front = lx_front/2
            vertical_shift_back = ly_back/2 - border_length/2
            horizontal_shift_back = lx_back/2 - overlap_distance
        elif (angle == math.pi/2):
            vertical_shift_front = lx_front/2
            horizontal_shift_front = overlap_misalignment
            vertical_shift_back = lx_back/2 - overlap_distance
            horizontal_shift_back = overlap_misalignment
        elif (0 < angle < math.pi/2) or (math.pi/2 < angle < math.pi):
            # extra diagonal (downward and right) shift of the front shape due
            # to overlapping misalignment
            extra_shift_front = ly_front/2 - border_length/2
            # extra diagonal (upward and left) shift of the back shape due to 
            # overlapping misalignment (not due to overlapping distance)
            extra_shift_back  = ly_back/2 - border_length/2
            
            vertical_shift_front = lx_front*math.sin(angle)/2 \
                                    + extra_shift_front*math.cos(angle)
            horizontal_shift_front = lx_front*math.cos(angle)/2 \
                                        - extra_shift_front*math.sin(angle)
            vertical_shift_back = lx_back*math.sin(angle)/2 \
                                    + extra_shift_back*math.cos(angle) \
                                    - overlap_distance*math.sin(angle)
            horizontal_shift_back = lx_back*math.cos(angle)/2 \
                                    - extra_shift_back*math.sin(angle) \
                                    - overlap_distance*math.cos(angle)

    # # for  pi/2 <  angle < pi, the RF sits on the bottom (now right) border 
    # # basically just like when 0 < angle < pi/2, but lx and ly are swapped, 
    # # and horizontal shift now becomes vertical shift, and vice-versa. This 
    # # makes sure the back shape is always to the right of the front shape.
    # elif (math.pi/2 < angle < math.pi):
    #     if (abs(overlap_misalignment) <= (lx_back - lx_front)):
    #         # when the border is still inside the back shape
    #         border_length = lx_front
    #     elif (overlap_misalignment > (lx_back - lx_front)):
    #         # when the front shape is too far down (positive misalignment)
    #         border_length = (lx_front + lx_back)/2 - overlap_misalignment
    #     else:
    #         # when the front shape is too far up (negative misalignment)
    #         border_length = (lx_front + lx_back)/2 + overlap_misalignment 

    #     new_angle = math.pi - angle
    #     # extra diagonal (downward and right) shift of the front shape due
    #     # to overlapping misalignment
    #     extra_shift_front = ly_front/2 - border_length/2
    #     # extra diagonal (upward and left) shift of the back shape due to 
    #     # overlapping misalignment (not due to overlapping distance)
    #     extra_shift_back  = ly_back/2 - border_length/2

    #     horizontal_shift_front = ly_front*math.sin(new_angle)/2 \
    #                             + extra_shift_front*math.cos(new_angle)
    #     vertical_shift_front = ly_front*math.cos(angle)/2 \
    #                                 - extra_shift_front*math.sin(new_angle)
    #     horizontal_shift_back = ly_back*math.sin(new_angle)/2 \
    #                             + extra_shift_back*math.cos(new_angle) \
    #                             - overlap_distance*math.sin(new_angle)
    #     vertical_shift_back = ly_back*math.cos(new_angle)/2 \
    #                             - extra_shift_back*math.sin(new_angle) \
    #                             - overlap_distance*math.cos(new_angle)
    return (vertical_shift_front, horizontal_shift_front,
            vertical_shift_back , horizontal_shift_back)
    

# Test
def _test_one_set_functions():
    # Set the parameters:
    RF_x = 55
    RF_y = 85
    RF_r = 15
    xn = 227
    yn = 224
    square_sizes=(40, 80, 160)

    # 1. make_standard_set()
    # for angle in np.linspace(0, math.pi-0.1, 4):
    #     standard_set = make_standard_set(RF_x, RF_y, RF_r, xn=xn, yn=yn, angle=angle)
    #     plot_one_set(standard_set, grid=True, RF_circ=True, RF_x=RF_x, RF_y=RF_y, RF_r=RF_r)
    standard_set = make_standard_set(RF_x, RF_y, RF_r, xn=xn, yn=yn)
    plot_one_set(standard_set, grid=True, RF_circ=True, RF_x=RF_x, RF_y=RF_y, RF_r=RF_r)

    # 2. make_overlap_set()
    overlap_set = make_overlap_set(RF_x, RF_y, RF_r, xn=xn, yn=yn)
    plot_one_set(overlap_set, grid=True, RF_circ=True, RF_x=RF_x, RF_y=RF_y, RF_r=RF_r)

    # 3. make_c_figure_set()
    c_figure_set = make_c_figure_set(RF_x, RF_y, RF_r, xn=xn, yn=yn)
    plot_one_set(c_figure_set, grid=True, RF_circ=True, RF_x=RF_x, RF_y=RF_y, RF_r=RF_r)
        
    # 4. make_size_set()
    size_set = make_size_set(RF_x, RF_y, RF_r, square_sizes=square_sizes,
                             xn=xn, yn=yn)
    plot_one_set(size_set, square_sizes=square_sizes, grid=True, RF_circ=True,
                 RF_x=RF_x, RF_y=RF_y, RF_r=RF_r, figsize=(15,10))
    

if __name__ == '__main__':
    _test_one_set_functions()

########### Functions that Make Stimulus Sets for a Whole Layer ############


def shift_coord(xn, yn, x_raw, y_raw, target_origin="top left"):
    """ Converts the coordinates such that the origin is at the top-left 
    corner instead of at the center.
    
    Parameters
    ----------
    xn : int
        The horizontal width of the stimulus image
    yn : int
        The vertical length of the stimulus image
    x_raw : int
        The raw coordinate along horizontal image axis
    y_raw : int
        The raw coordinate along vertical image axis
    target_origin : str, default="top left"
        Where you would like the origin of the transformed coordinate system to 
        be? Options are: "top left" (if there are not any negative indices. 
        True for most 2d arrays.) and "center". Raise NotImplemented Error
        for invalid input.
    
    Returns
    -------
    A tuple containing the transformed coordinates: (transformed_x, transformed_y)
    
    """
    if (target_origin.lower() == "top left"):
        transformed_x = x_raw + math.floor((xn - 1)/2.0)
        transformed_y = y_raw + math.floor((yn - 1)/2.0)
    elif (target_origin.lower() == "center"): 
        transformed_x = x_raw - math.floor((xn - 1)/2.0)
        transformed_y = y_raw - math.floor((yn - 1)/2.0)
    else:
        raise NotImplementedError
    return (transformed_x, transformed_y)


def make_standard_sets(unit_stats, ori_stats, lx=120, ly=120, 
                       shape_rgb=(0.75,0.75,0.75), xn=227, yn=227, 
                       background_rgb=(0.35,0.35,0.35)):
    """Generates standard sets for the unique units that responses stronger
    than the threshold. This function generate the standard sets for the 
    entire convolutional layer given the receptive field (RF) statistics. 
    For each unit, Make a set of 4 overlap figures containing rectangles for 
    standard test as shown in Figure 2 in Zhou et al.'s (2000) paper.

    Parameters
    ----------
    unit_stats : n x 6 numpy array of float
        Each row contains six RF stats (i.e., unit_idx, R_max, CM_y, CM_x, RF_r, 
        and f_nat) about one unit in the convolutional layer. The first dimension
        n is the number of unique units in the convolutional layer.
    ori_stats : n x 1 numpy array of float
        The counterclockwise rotation of the rectangle in radians, from 0 to pi.
    lx : int, default=120
        The horizontal length of the rectangle
    ly : int, default=120
        The vertical length of the rectangle
    shape_rgb : tuple of 3 floats, default=(0.75, 0.75, 0.75)
        The color intensity of the rectangle, from 0.0 to 1.0.
    xn : int, default=227
        The horizontal width of the stimulus image
    yn : int, default=227
        The vertical length of the stimulus image
    background_rgb : tuple of 3 floats, default=(0.35, 0.35, 0.35)
        The background color intensity, from 0.0 to 1.0.

    Returns
    -------
    A dictionary containing the unit_idx as the keys, and the respective 
    4 x 3 x <yn> x <xn> numpy array as the values.

    Example
    -------
    standard_sets_dict = boshape.make_standard_sets(unit_stats_conv2, unit_stats_conv2_ori[:,3])
    boshape.plot_one_set(standard_sets_dict[1]) # plot the second unit

    Notes
    -----
    1. For the images/stimuli, the x-coordinate points to the right, and 
       the y-coordinate points downward. The origin is at the top-left 
       corner. 
    2. The "shape_rgb" is actually the background RGB color for 
       conditions B and C. Similarly, the "background_rgb" is actually 
       the rectangle's RGB values for conditions B and C. 
    """
    num_units = len(unit_stats)
    standard_sets_dict = {} 
    # key = index of the unit; value = a 4D numpy array (4 x 3 x yn x xn)

    with tqdm(total=num_units, position=0, leave=True) as pbar:
        for i in range(num_units):
            unit_idx, R_max, CM_y, CM_x, RF_r, f_nat = unit_stats[i, :]
            angle = ori_stats[i]

            # Define the origin as the top-left corner instead of the center.
            RF_x, RF_y = shift_coord(xn, yn, CM_x, CM_y)

            one_set = make_standard_set(RF_x, RF_y, RF_r, lx=lx, ly=ly, 
                                        angle=angle, shape_rgb=shape_rgb, 
                                        xn=xn, yn=yn, background_rgb=background_rgb)
            standard_sets_dict.update({int(unit_idx + 0.1): one_set}) # add 0.1 to prevent rounding error
            pbar.update()

    return standard_sets_dict


def make_overlap_sets(unit_stats, ori_stats,
                      lx_front=120, ly_front=120, lx_back=200, ly_back=150, 
                      front_shape_rgb=(0.75,0.75,0.75), back_shape_rgb=(0.35,0.35,0.35),
                      overlap_misalignment=40, overlap_distance=60,
                      xn=227, yn=227, background_rgb=(0.1,0.1,0.1)):
    """Generates overlapping sets for the unique units that responses stronger 
    than the threshold. This function generate the overlapping sets for the 
    entire convolutional layer given the receptive field (RF) statistics. 
    For each unit, make a set of 4 figures containing overlapping shapes as 
    shownin Figure 3 in Zhou et al. (2000). This function makes sure that the 
    back shape is always to the right side of the front shape.
    
    Parameters
    ----------
    unit_stats : n x 6 numpy array of float
        Each row contains six RF stats (i.e., unit_idx, R_max, CM_y, CM_x, RF_r, 
        and f_nat) about one unit in the convolutional layer. The first dimension
        n is the number of unique units in the convolutional layer.
    ori_stats : n x 1 numpy array of float
        The counterclockwise rotation of the rectangle in radians, from 0 to pi.
    lx_front : int, default=120
        The horizontal length of the front rectangle
    ly_front : int, default=120
        The vertical length of the front rectangle
    lx_back : int, default=200
        The horizontal length of the occluded rectangle
    ly_back : int, default=150
        The vertical length of the occluded rectangle
    front_shape_rgb : tuple of 3 floats, default=(0.75, 0.75, 0.75)
        The color intensity of the rectangle, from 0.0 to 1.0.
    back_shape_rgb : tuple of 3 floats, default=(0.35, 0.35, 0.35)
        The color intensity of the occluded rectangle for first stimulus, 
        from 0.0 to 1.0.
    overlap_misalignment : int, default=40
        The distance between the centers of the two rectangles along their 
        vertical axes. Can be thought of as how 'misaligned' the two are.
    overlap_distance : int, default=60
        The distance between the centers of the two rectangles along their 
        horizontal axes. Can be thought of as the degree of overlapping.
    xn : int, default=227
        The horizontal width of the stimulus image
    yn : int, default=227
        The vertical length of the stimulus image
    background_rgb : tuple of 3 floats, default=(0.1, 0.1, 0.1)
        The background color intensity, from 0.0 to 1.0.
    
    Returns
    -------
    A dictionary containing the unit_idx as the keys, and the respective 
    4 x 3 x <yn> x <xn> numpy array as the values.

    Example
    -------
    overlap_sets_dict = boshape.make_overlap_sets(unit_stats_conv2, unit_stats_conv2_ori[:,3])
    boshape.plot_one_set(overlap_sets_dict[1]) # plot the second unit

    Notes
    -----
    1. For the images/stimuli, the x-coordinate points to the right, and 
       the y-coordinate points downward. The origin is at the top-left 
       corner.
    2. The "front_shape_rgb" is actually the RGB value of the occluded 
       rectangle for conditions B and C. Similarly, the "back_shape_rgb" is 
       actually the front rectangle's RGB values for conditions B and C. 
    """
    num_units = len(unit_stats)
    overlap_sets_dict = {} 
    # key = index of the unit; value = a 4D numpy array (4 x 3 x yn x xn)

    with tqdm(total=num_units, position=0, leave=True) as pbar:
        for i in range(num_units):
            unit_idx, R_max, CM_y, CM_x, RF_r, f_nat = unit_stats[i, :]
            angle = ori_stats[i]

            # Define the origin as the top-left corner instead of the center.
            RF_x, RF_y = shift_coord(xn, yn, CM_x, CM_y)

            one_set = make_overlap_set(RF_x, RF_y, RF_r, lx_front=lx_front, ly_front=ly_front, 
                                       lx_back=lx_back, ly_back=ly_back, angle=angle,
                                       front_shape_rgb=front_shape_rgb, 
                                       back_shape_rgb=back_shape_rgb,
                                       overlap_misalignment=overlap_misalignment, 
                                       overlap_distance=overlap_distance, 
                                       xn=xn, yn=yn, background_rgb=background_rgb)
            overlap_sets_dict.update({int(unit_idx + 0.1): one_set}) # add 0.1 to prevent rounding error
            pbar.update()

    return overlap_sets_dict


def make_c_figure_sets(unit_stats, ori_stats, lx=120, ly=240,
                       l_inside=120, shape_rgb=(0.75, 0.75, 0.75), 
                       xn=227, yn=227, background_rgb=(0.35, 0.35, 0.35)):
    """Generates c-figure sets for the unique units that responses stronger
    than the threshold. This function generate the c-figure sets for the
    entire convolutional layer given the receptive field (RF) statistics. 
    For each unit, make a set of 4 c-shaped figures as shown in Figures 23, 
    24, and 26 in Zhou et al. (2000).
    
    Parameters
    ----------
    unit_stats : n x 6 numpy array
        Each row contains six RF stats (i.e., unit_idx, R_max, CM_y, CM_x, RF_r, 
        and f_nat) about one unit in the convolutional layer. The first dimension
        n is the number of unique units in the convolutional layer.
    ori_stats : n x 1 numpy array of float
        The counterclockwise rotation of the rectangle in radians, from 0 to pi.
    lx : int, default=120
        The maximum horizontal width of the c-figure
    ly : int, default=240
        The vertical length of the c-figure
    l_inside : int, default=120
        The vertical length of the area to be carved out to create a c-figure
        out of a larger rectangle. Throws ValueError if l_inside >= ly. 
    shape_rgb : tuple of 3 floats, default=(0.75, 0.75, 0.75)
        The color intensity of the c-figure, from 0.0 to 1.0.
    xn : int, default=227
        The horizontal width of the stimulus image
    yn : int, default=227
        The vertical length of the stimulus image
    background_rgb : tuple of 3 floats, default=(0.35, 0.35, 0.35)
        The background color intensity, from 0.0 to 1.0.

    Returns
    -------
    A dictionary containing the unit_idx as the keys, and the respective 
    4 x 3 x <yn> x <xn> numpy array as the values.

    Example
    -------
    c_figure_sets = boshape.make_c_figure_sets(unit_stats_conv2, unit_stats_conv2_ori[:,3])
    boshape.plot_one_set(c_figure_sets[1]) # plot the second unit

    Notes
    -----
    1. For the images/stimuli, the x-coordinate points to the right, and 
       the y-coordinate points downward. The origin is at the top-left 
       corner.
    2. The "shape_rgb" is actually the background RGB color for 
       conditions B and C. Similarly, the "background_rgb" is actually 
       the rectangle's RGB values for conditions B and C. 
    """
    num_units = len(unit_stats)
    c_figure_sets_dict = {}
    # key = index of the unit; value = a 4D numpy array (4 x 3 x yn x xn)

    with tqdm(total=num_units, position=0, leave=True) as pbar:
        for i in range(num_units):
            unit_idx, R_max, CM_y, CM_x, RF_r, f_nat = unit_stats[i, :]
            angle = ori_stats[i]

            # Define the origin as the top-left corner instead of the center.
            RF_x, RF_y = shift_coord(xn, yn, CM_x, CM_y)

            one_set = make_c_figure_set(RF_x, RF_y, RF_r, lx=lx, ly=ly, 
                                        l_inside=l_inside, angle=angle, 
                                        shape_rgb=shape_rgb, 
                                        xn=xn, yn=yn, background_rgb=background_rgb)
            c_figure_sets_dict.update({int(unit_idx + 0.1): one_set}) # add 0.1 to prevent rounding error
            pbar.update()

    return c_figure_sets_dict


def make_size_sets(unit_stats, ori_stats,
                   square_sizes=(70,120,200), shape_rgb=(0.75,0.75,0.75), 
                   xn=227, yn=227, background_rgb=(0.35,0.35,0.35)):
    """Generates size-variance sets for the unique units that responses 
    stronger than the threshold. This function generate the size-variance 
    sets for the entire convolutional layer given the receptive field (RF) 
    statistics. For each unit,  make a set of pairs of squares of varying 
    sizes as shown in Figure 9 in Zhou et al. (2000).
    
    Parameters
    ----------
    unit_stats : n x 6 numpy array
        Each row contains six RF stats (i.e., unit_idx, R_max, CM_y, CM_x, RF_r, 
        and f_nat) about one unit in the convolutional layer. The first dimension
        n is the number of unique units in the convolutional layer.
    ori_stats : n x 1 numpy array of float
        The counterclockwise rotation of the rectangle in radians, from 0 to pi.
    square_sizes: tuple of int, default=(70, 120, 200)
        The list containing the side lengths of the squares used as stimuli.
    shape_rgb : tuple of 3 floats, default=(0.75, 0.75, 0.75)
        The color intensity of the rectangle, from 0.0 to 1.0.
    xn : int, default=227
        The horizontal width of the stimulus image
    yn : int, default=227
        The vertical length of the stimulus image
    background_rgb : tuple of 3 floats, default=(0.35, 0.35, 0.35)
        The background color intensity, from 0.0 to 1.0.

    Returns
    -------
    A dictionary containing the unit_idx as the keys, and the respective 
    n x 3 x <yn> x <xn> numpy array as the values.

    Example
    -------
    size_sets_dict = boshape.make_size_sets(unit_stats_conv2, unit_stats_conv2_ori[:,3])
    boshape.plot_one_set(size_sets_dict[1]square_sizes=(70,120,200)) # plot the second unit

    Notes
    -----
    1. For the images/stimuli, the x-coordinate points to the right, and 
       the y-coordinate points downward. The origin is at the top-left 
       corner. 
    2. The "shape_rgb" is actually the background RGB color for the second
       condition of each size. Similarly, the "background_rgb" is actually 
       the rectangle's RGB values for the second condition of each size. 
    """
    num_units = len(unit_stats)
    size_sets_dict = {}
    # key = index of the unit; value = a 4D numpy array (2*len(square_sizes) x 3 x yn x xn)

    with tqdm(total=num_units, position=0, leave=True) as pbar:
        for i in range(num_units):
            unit_idx, R_max, CM_y, CM_x, RF_r, f_nat = unit_stats[i, :]
            angle = ori_stats[i]

            # Define the origin as the top-left corner instead of the center.
            RF_x, RF_y = shift_coord(xn, yn, CM_x, CM_y)

            one_set = make_size_set(RF_x, RF_y, RF_r, square_sizes=square_sizes, 
                                    angle=angle, shape_rgb=shape_rgb, 
                                    xn=xn, yn=yn, background_rgb=background_rgb)
            size_sets_dict.update({int(unit_idx + 0.1): one_set}) # add 0.1 to prevent rounding error
            pbar.update()

    return size_sets_dict


# Test
def _test_one_layer_functions():
    # Set some fake stats:
    unit_stats_conv2 = np.array([[0, 23.9, 5.0,  -8.2, 21.9, 0.73],
                                 [1, 42.0, 0.6,     0, 16.3, 0.92],
                                 [2, 22.5, 4.2, -19.8, 21.6, 0.91]])
    unit_stats_conv2_ori = np.array([[0, 1, 2, math.pi/2],
                                     [1, 1, 2, 0.5],
                                     [2, 2, 2, 0]])

    # 1. make_standard_sets()
    standard_sets_dict = make_standard_sets(unit_stats_conv2, unit_stats_conv2_ori[:,3])
    plot_one_set(standard_sets_dict[1]) # plot the second unit

    # 2. make_overlap_sets()
    overlap_sets_dict = make_overlap_sets(unit_stats_conv2, unit_stats_conv2_ori[:,3])
    plot_one_set(overlap_sets_dict[1]) # plot the second unit

    # 3. make_c_figure_sets()
    c_figure_sets_dict = make_c_figure_sets(unit_stats_conv2, unit_stats_conv2_ori[:,3])
    plot_one_set(c_figure_sets_dict[1]) # plot the second unit
        
    # 4. make_size_sets()
    size_sets_dict = make_size_sets(unit_stats_conv2, unit_stats_conv2_ori[:,3])
    plot_one_set(size_sets_dict[1], square_sizes=(70,120,200), figsize=(15,10)) # plot the second unit
    

if __name__ == '__main__':
    _test_one_layer_functions()