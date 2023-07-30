"""
 This demonstrates the generation of the original Pasupathy and Connor (2001)
 shape set.  This demo was built based on coordinates supplied by Anitha
 Pasupathy.  Anitha's MATLAB code was converted to Python by Taekjun Kim.
 Wyeth Bair adapted the coordinates and Taekjun's code to create this demo.

 Taekjun Kim
 Anitha Pasupathy
 Wyeth Bair, April 23, 2020

 IMPORTANT NOTES
   There are 51 unique shapes.
   Each shape is defined by a set of control points, which are (x,y) coords.
   A spline equation is then used to precisely sample the shape boundary.
   *** The shapes are not necessarily centered.
   Shapes are typically shown at up to 8 rotations (45 deg increments)
   Because of symmetry, some shapes are not unique at all 8 rotations.
   Because shapes are not necessarily centered, even non-unique rotations
     can generate unique shape images.
"""
import os
import sys
import math
import warnings

import concurrent.futures
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../..')
from src.rf_mapping.image import make_box
from src.rf_mapping.hook import ConvUnitCounter
from src.rf_mapping.spatial import (xn_to_center_rf,
                                    calculate_center,
                                    get_rf_sizes,)
from src.rf_mapping.files import delete_all_npy_files
from src.rf_mapping.net import get_truncated_model
import src.rf_mapping.constants as c
from src.rf_mapping.stimulus import *


__all__ = ['make_pasu_shape', 'make_pasu_shape_color',
           'pasu_bw_run_01b', 'pasu_color_run_01b']

#
#  In this cell are the control coordinates for all 51 shapes, in addition
#  to some extra information about the number of control coordinates and
#  the number of unique rotations of the shapes.  The extra information is
#  not used in the demo below.

# Number of coordinate values for each shape (2 x Number of x,y pairs)
pasu_shape_n = [
18,18,26,26,26,30,26,18,30,24,24,18,34,26,26,30,38,34,34,42,42,38,46,26,34,34,34,42,42,42,50,34,42,50,50,58,66,26,26,26,30,30,34,50,34,26,38,42,42,30,34]

# Number of rotations for each shape, within standard set of 370
#   Thus, the sum of this array = 370
pasu_shape_nrot = [1,1,8,8,4,8,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,2,8,4,8,8,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8]

# True unique rotations
#    *** Under the assumption that the shape is centered, but shape 4 is
#        not centered.
#    Note, two '8' values have been replaced by '4' relative to the above.
pasu_shape_nrotu = [1,1,8,4,4,8,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,2,8,4,8,8,2,8,8,8,8,8,8,8,4,8,8,8,8,8,8]

# Control coordinates for each of the 51 shapes listed as 1D arrays,
#   in the format:  x0,y0,x1,y1,x2,y2,...
# *** WYETH NOTE:  Apr 2014 - slight change in coords to Pasu_shape_29
pasu_shape = [
[-0.4,0.0,-0.283,0.283,0.0,0.4,0.283,0.283,0.4,0.0,0.283,-0.283,0.0,-0.4,-0.283,-0.283,-0.4,0.0],
[-1.6,0.0,-1.131,1.131,0.0,1.6,1.131,1.131,1.6,0.0,1.131,-1.131,0.0,-1.6,-1.131,-1.131,-1.6,0.0],
[-0.4,0.0,-0.375,0.416,-0.3,0.825,-0.174,1.221,0.0,1.6,0.174,1.221,0.3,0.825,0.375,0.416,0.4,0.0,0.283,-0.283,0.0,-0.4,-0.283,-0.283,-0.4,0.0],
[-0.481,0.215,-0.518,0.6,-0.481,0.983,-0.369,1.354,0.0,1.6,0.369,1.354,0.481,0.983,0.518,0.6,0.481,0.215,0.369,-0.154,0.0,-0.4,-0.369,-0.154,-0.481,0.215],
[-0.373,0.0,-0.266,0.828,-0.069,1.37,0.0,1.6,0.069,1.37,0.266,0.828,0.373,0.0,0.266,-0.828,0.069,-1.37,0.0,-1.6,-0.069,-1.37,-0.266,-0.828,-0.373,0.0],
[-0.438,0.195,-0.277,0.916,-0.182,1.19,-0.102,1.387,0.0,1.6,0.102,1.387,0.182,1.19,0.277,0.916,0.438,0.195,0.477,-0.543,0.393,-1.278,0.0,-1.6,-0.393,-1.278,-0.477,-0.543,-0.438,0.195],
[-0.64,0.0,-0.571,0.689,-0.369,1.354,0.0,1.6,0.369,1.354,0.571,0.689,0.64,0.0,0.571,-0.689,0.369,-1.354,0.0,-1.6,-0.369,-1.354,-0.571,-0.689,-0.64,0.0],
[-0.266,0.82,0.0,1.6,0.122,0.988,0.468,0.468,0.988,0.122,1.6,0.0,0.82,-0.266,0.066,0.066,-0.266,0.82],
[-0.386,0.589,-0.386,1.303,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.303,-0.386,0.589,-0.386,-0.029,-0.029,-0.386,0.589],
[-0.082,0.186,-0.289,0.884,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.483,0.283,1.6,0.0,1.278,-0.393,0.467,-0.294,-0.082,0.186],
[-1.6,0.0,-1.483,0.283,-1.2,0.4,-0.74,0.5,-0.351,0.751,-0.1,1.14,0.0,1.6,0.289,0.884,0.082,0.186,-0.467,-0.294,-1.278,-0.393,-1.6,0.0],
[-0.245,0.781,0.0,1.6,0.075,0.846,0.294,0.122,0.651,-0.546,1.131,-1.131,0.385,-0.733,-0.108,-0.051,-0.245,0.781],
[-0.427,0.573,-0.393,1.278,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.373,0.652,0.487,0.198,0.727,-0.203,1.071,-0.516,1.248,-0.848,1.131,-1.131,0.85,-1.25,0.626,-1.181,0.106,-0.713,-0.256,-0.11,-0.427,0.573],
[-0.123,0.149,-0.167,0.883,0.0,1.6,0.054,0.983,0.257,0.401,0.605,-0.111,1.071,-0.516,1.216,-0.848,1.131,-1.131,0.85,-1.25,0.57,-1.131,0.13,-0.542,-0.123,0.149],
[-0.605,-0.111,-0.257,0.401,-0.054,0.983,0.0,1.6,0.167,0.883,0.123,0.149,-0.13,-0.542,-0.57,-1.131,-0.85,-1.25,-1.131,-1.131,-1.216,-0.848,-1.071,-0.516,-0.605,-0.111],
[-0.533,0.0,-0.397,0.843,-0.176,1.333,0.0,1.6,0.122,0.988,0.468,0.468,0.988,0.122,1.6,0.0,0.988,-0.122,0.468,-0.468,0.122,-0.988,0.0,-1.6,-0.176,-1.333,-0.397,-0.843,-0.533,0.0],
[-0.533,0.0,-0.397,0.843,-0.176,1.333,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.176,-1.333,-0.397,-0.843,-0.533,0.0],
[-0.575,0.172,-0.381,0.923,-0.212,1.273,0.0,1.6,0.122,0.988,0.468,0.468,0.988,0.122,1.6,0.0,1.14,-0.092,0.752,-0.352,0.492,-0.74,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.369,-1.354,-0.571,-0.605,-0.575,0.172],
[-0.571,0.605,-0.369,1.354,0.0,1.6,0.283,1.483,0.4,1.2,0.492,0.74,0.752,0.352,1.14,0.092,1.6,0.0,0.988,-0.122,0.468,-0.468,0.122,-0.988,0.0,-1.6,-0.212,-1.273,-0.381,-0.923,-0.575,-0.172,-0.571,0.605],
[-0.575,0.172,-0.381,0.923,-0.212,1.273,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.369,-1.354,-0.571,-0.605,-0.575,0.172],
[-0.571,0.605,-0.369,1.354,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.212,-1.273,-0.381,-0.923,-0.575,-0.172,-0.571,0.605],
[-0.64,0.0,-0.571,0.689,-0.369,1.354,0.0,1.6,0.283,1.483,0.4,1.2,0.492,0.74,0.752,0.352,1.14,0.092,1.6,0.0,1.14,-0.092,0.752,-0.352,0.492,-0.74,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.369,-1.354,-0.571,-0.689,-0.64,0.0],
[-0.64,0.0,-0.571,0.689,-0.369,1.354,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.369,-1.354,-0.571,-0.689,-0.64,0.0],
[-0.294,0.122,-0.075,0.846,0.0,1.6,0.075,0.846,0.294,0.122,0.651,-0.546,1.131,-1.131,0.612,-0.785,0.0,-0.663,-0.612,-0.785,-1.131,-1.131,-0.651,-0.546,-0.294,0.122],
[-0.467,0.102,-0.35,0.505,-0.393,1.122,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.35,0.505,0.467,-0.102,0.751,-0.688,1.131,-1.131,0.612,-0.785,0.0,-0.663,-0.612,-0.785,-1.131,-1.131,-0.751,-0.688,-0.467,0.102],
[-0.294,0.122,-0.075,0.846,0.0,1.6,0.054,0.983,0.257,0.401,0.605,-0.111,1.071,-0.516,1.248,-0.848,1.131,-1.131,0.85,-1.25,0.57,-1.131,0.179,-0.871,-0.282,-0.78,-0.742,-0.871,-1.131,-1.131,-0.651,-0.546,-0.294,0.122],
[-0.257,0.401,-0.054,0.983,0.0,1.6,0.075,0.846,0.294,0.122,0.651,-0.546,1.131,-1.131,0.742,-0.871,0.282,-0.78,-0.179,-0.871,-0.571,-1.131,-0.85,-1.25,-1.131,-1.131,-1.248,-0.848,-1.071,-0.516,-0.605,-0.111,-0.257,0.401],
[-0.257,0.401,-0.054,0.983,0.0,1.6,0.054,0.983,0.257,0.401,0.605,-0.111,1.071,-0.516,1.248,-0.848,1.131,-1.131,0.85,-1.25,0.57,-1.131,0.308,-0.957,0.0,-0.896,-0.308,-0.957,-0.57,-1.131,-0.85,-1.25,-1.131,-1.131,-1.248,-0.848,-1.071,-0.516,-0.605,-0.111,-0.257,0.401],
[-0.487,0.198,-0.373,0.652,-0.393,1.122,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.35,0.505,0.467,-0.102,0.751,-0.688,1.131,-1.131,0.742,-0.871,0.282,-0.78,-0.179,-0.871,-0.571,-1.131,-0.85,-1.25,-1.131,-1.131,-1.248,-0.848,-1.071,-0.516,-0.727,-0.203,-0.487,0.198],
[-0.467,-0.102,-0.35,0.505,-0.393,1.122,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.373,0.652,0.487,0.198,0.727,-0.203,1.071,-0.516,1.248,-0.848,1.131,-1.131,0.85,-1.25,0.571,-1.131,0.179,-0.871,-0.282,-0.78,-0.742,-0.871,-1.131,-1.131,-0.751,-0.688,-0.467,-0.102],
[-0.487,0.198,-0.373,0.652,-0.393,1.122,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.373,0.652,0.487,0.198,0.727,-0.203,1.071,-0.516,1.248,-0.848,1.131,-1.131,0.85,-1.25,0.57,-1.131,0.308,-0.957,0.0,-0.896,-0.308,-0.957,-0.57,-1.131,-0.85,-1.25,-1.131,-1.131,-1.248,-0.848,-1.071,-0.516,-0.727,-0.203,-0.487,0.198],
[-1.6,0.0,-0.988,0.122,-0.468,0.468,-0.122,0.988,0.0,1.6,0.122,0.988,0.468,0.468,0.988,0.122,1.6,0.0,0.988,-0.122,0.468,-0.468,0.122,-0.988,0.0,-1.6,-0.122,-0.988,-0.468,-0.468,-0.988,-0.122,-1.6,0.0],
[-1.6,0.0,-0.988,0.122,-0.468,0.468,-0.122,0.988,0.0,1.6,0.122,0.988,0.468,0.468,0.988,0.122,1.6,0.0,1.14,-0.1,0.751,-0.351,0.5,-0.74,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.283,-1.483,-0.4,-1.2,-0.5,-0.74,-0.751,-0.351,-1.14,-0.1,-1.6,0.0],
[-1.6,0.0,-1.483,0.283,-1.2,0.4,-0.74,0.5,-0.351,0.751,-0.1,1.14,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.1,-1.14,-0.351,-0.751,-0.74,-0.5,-1.2,-0.4,-1.483,-0.283,-1.6,0.0],
[-1.6,0.0,-0.988,0.122,-0.468,0.468,-0.122,0.988,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.283,-1.483,-0.4,-1.2,-0.5,-0.74,-0.751,-0.351,-1.14,-0.1,-1.6,0.0],
[-1.6,0.0,-1.483,0.283,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,-0.283,1.483,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.1,-1.14,-0.351,-0.751,-0.74,-0.5,-1.2,-0.4,-1.483,-0.283,-1.6,0.0],
[-1.6,0.0,-1.483,0.283,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,-0.283,1.483,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.283,-1.483,-0.4,-1.2,-0.461,-0.894,-0.635,-0.635,-0.894,-0.461,-1.2,-0.4,-1.483,-0.283,-1.6,0.0],
[-0.571,0.605,0.0,1.6,1.131,1.131,1.6,0.0,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.212,-1.273,-0.381,-0.923,-0.575,-0.172,-0.571,0.605],
[-0.575,0.172,-0.381,0.923,-0.212,1.273,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.6,0.0,1.131,-1.131,0.0,-1.6,-0.571,-0.605,-0.575,0.172],
[-0.257,0.401,-0.054,0.983,0.0,1.6,0.054,0.983,0.257,0.401,0.605,-0.111,1.071,-0.516,1.131,-1.131,0.0,-1.6,-1.131,-1.131,-1.071,-0.516,-0.605,-0.111,-0.257,0.401],
[-0.64,0.0,-0.571,0.689,0.0,1.6,1.131,1.131,1.6,0.0,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.369,-1.354,-0.571,-0.689,-0.64,0.0],
[-0.64,0.0,-0.571,0.689,-0.369,1.354,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.6,0.0,1.131,-1.131,0.0,-1.6,-0.571,-0.689,-0.64,0.0],
[-0.487,0.198,-0.373,0.652,-0.393,1.122,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.373,0.652,0.487,0.198,0.727,-0.203,1.071,-0.516,1.131,-1.131,0.0,-1.6,-1.131,-1.131,-1.071,-0.516,-0.727,-0.203,-0.487,0.198],
[-1.6,0.0,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,-0.283,1.483,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.0,-1.6,-1.131,-1.131,-1.6,0.0],
[-1.6,0.0,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,0.0,1.6,1.131,1.131,1.6,0.0,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.0,-1.6,-1.131,-1.131,-1.6,0.0],
[-1.6,0.0,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,0.0,1.6,1.131,1.131,1.6,0.0,1.131,-1.131,0.0,-1.6,-1.131,-1.131,-1.6,0.0],
[-1.6,0.0,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,-0.283,1.483,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.6,0.0,1.131,-1.131,0.0,-1.6,-1.131,-1.131,-1.6,0.0],
[-1.6,0.0,-1.131,1.131,0.0,1.6,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.1,-1.14,-0.351,-0.751,-0.74,-0.5,-1.2,-0.4,-1.6,0.0],
[-1.6,0.0,-1.483,0.283,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,0.0,1.6,1.131,1.131,1.6,0.0,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.1,-1.14,-0.351,-0.751,-0.74,-0.5,-1.2,-0.4,-1.483,-0.283,-1.6,0.0],
[-1.6,0.0,-1.131,1.131,0.0,1.6,1.131,1.131,1.6,0.0,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.1,-1.14,-0.351,-0.751,-0.74,-0.5,-1.2,-0.4,-1.6,0.0],
[-1.6,0.0,-0.988,0.122,-0.468,0.468,-0.122,0.988,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.6,0.0,1.131,-1.131,0.0,-1.6,-0.4,-1.2,-0.5,-0.74,-0.751,-0.351,-1.14,-0.1,-1.6,0.0]]

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
#                                    FVMAX                                    #
#                                                                             #
###############################################################################
def fvmax(invec, sample=50):
    """
    Routine from Taekjun Kim that takes the control points 'invec' and produces
    a more finely sampled set of points as 'outvec'.
    Here, there is a 50 times increase in sampling by default.

    A spline equation is used to interpolate between the input points.
    """
    num_pts = np.shape(invec)[0]
    inshft = np.vstack((invec[num_pts-2,:],invec,invec[1,:]))
    ip = np.arange(0,sample,1)/sample
    
    vtx = np.empty((1,(num_pts-1)*sample+1))
    vty = np.empty((1,(num_pts-1)*sample+1))
    #dvtx = np.empty((1,num*50-49))
    #dvty = np.empty((1,num*50-49))
    
    for i in range(0,num_pts-1):
        bufvrt = inshft[i:i+4,:]
        
        # Spline equation
        incr = np.empty((4,len(ip)))
        incr[0,:] =   -ip*ip*ip + 3*ip*ip -3*ip + 1
        incr[1,:] =  3*ip*ip*ip - 6*ip*ip +4
        incr[2,:] = -3*ip*ip*ip + 3*ip*ip +3*ip + 1
        incr[3,:] =    ip*ip*ip
        
        # dincr = np.empty((4,len(ip)));
        # dincr[0,:] = -3*ip*ip +  6*ip - 3
        # dincr[1,:] =  9*ip*ip - 12*ip
        # dincr[2,:] = -9*ip*ip +  6*ip + 3
        # dincr[3,:] =  3*ip*ip
        vtx[0,i*sample:(i+1)*sample] = np.sum(np.tile(bufvrt[:,0].reshape(4,1),(1,sample))*incr,axis=0)/6.0
        vty[0,i*sample:(i+1)*sample] = np.sum(np.tile(bufvrt[:,1].reshape(4,1),(1,sample))*incr,axis=0)/6.0
        
        #  Wyeth says: These appear to not be used here
        #  Anitha says: These are derivates for computing actual curvature values, if needed
        # dvtx[0,i*50:(i+1)*50] = np.sum(np.tile(bufvrt[:,0].reshape(4,1),(1,50))*dincr,axis=0)/6.0
        # dvty[0,i*50:(i+1)*50] = np.sum(np.tile(bufvrt[:,1].reshape(4,1),(1,50))*dincr,axis=0)/6.0
    
    vtx[0,(num_pts-1)*sample] = vtx[0,0]
    vty[0,(num_pts-1)*sample] = vty[0,0]
    
    outvec = np.transpose(np.vstack((vtx, vty)))
    return outvec


#######################################.#######################################
#                                                                             #
#                               DRAE_BOOL_CONTOUR                             #
#                                                                             #
###############################################################################
@jit
def draw_bool_contour(x, y, shape):
    """
    Returns a boolean map of the size 'shape' with False everywhere except for
    the coordinates provided by x and y.
    """
    bool_contour = np.full((shape), False)
    for x, y in zip(x, y):
        x_trans = int((x + 2)/4 * shape[1])
        x_trans = clip(x_trans, 0, shape[1]-1)
        y_trans = int((y + 2)/4 * shape[0])
        y_trans = clip(y_trans, 0, shape[0]-1)
        bool_contour[y_trans, x_trans] = True
    return bool_contour


#######################################.#######################################
#                                                                             #
#                                FILL_CONTOUR                                 #
#                                                                             #
###############################################################################
def fill_contour(bool_contour, fgval=1.0, bgval=-1.0):
    """A line scan filling algorithm for Pasupathy shapes."""
    filled_contour = np.full((bool_contour.shape), bgval)
    filled_contour[bool_contour] = fgval
    prev_row = np.full(bool_contour[0].shape, bgval)
    for row_i, bool_row in enumerate(bool_contour):
        curr_row = fill_row(bool_row, fgval, bgval, prev_row, row_i)
        filled_contour[row_i, :] = curr_row
        prev_row = curr_row
    return filled_contour


#######################################.#######################################
#                                                                             #
#                                  FILL_ROW                                   #
#                                                                             #
###############################################################################
def fill_row(bool_row, fgval, bgval, prev_row, row_i):
    """
    Return an array of the same length as bool_row but with 'fgval' values at
    the interior of the contours and 'bgval' values at the exterior.
    
    Tony Fu, September 6th, 2022
    """
    max_thickness = max(bool_row.shape[0] // 12, 1)
    is_fgval = np.isclose(prev_row, fgval)

    # Initialize the output array.
    filled_row = np.full(bool_row.shape, bgval)
    
    # Get the indicies of the contours.
    contour_indices = np.where(bool_row)[0]
    
    # Handle 'edge' cases (get it?):
    if contour_indices.shape[0] == 0:
        return filled_row
    if contour_indices.shape[0] == 1:
        filled_row[contour_indices[0]] = fgval
        return filled_row

    # Because contour can be multiple pixels in thickness, we have to group
    # the start and end of each contour into a tuple such that the list becomes
    # [(start1, end1), (start2, end2), ...], where 'start1' and 'end1' are the
    # starting and ending indices of contour no.1, respectively.
    contour_start_end_pairs = []
    start = end = contour_indices[0]
    for idx in contour_indices[1:]:
        # If the current idx is adjacent to end, make it the new end.
        if end == idx - 1:
            end = idx
        # Otherwise, record the previous (start, end) pair, and start tracking
        # the new pair.
        else:
            contour_start_end_pairs.append((start, end))
            start = idx
            end = idx
    # Don't forget to append the last pair.
    if len(contour_start_end_pairs) == 0 or contour_start_end_pairs[-1] != (start,end):
        contour_start_end_pairs.append((start,end))

    # print(row_i, contour_start_end_pairs)

    # Case: 1 contour
    if len(contour_start_end_pairs) == 1:
        start, end = contour_start_end_pairs[0]
        filled_row[start:end+1] = fgval
        return filled_row
    
    # Case: 2 contours
    # TODO: 'underline' artifacts seen in 3 shapes when:
    #       1. output shape is lower than 70 AND
    #       2. the shape is symmetrical about the vertical axis AND
    #       3. it has two vertices pointing down. 
    if len(contour_start_end_pairs) == 2:
        l_start, l_end = contour_start_end_pairs[0]
        r_start, r_end = contour_start_end_pairs[1]
        total_sum = np.sum(is_fgval)
        lr_sum = np.sum(is_fgval[l_end:r_start])
        filled_row[l_start:l_end+1] = fgval
        filled_row[r_start:r_end+1] = fgval
        if total_sum != 0 and lr_sum > 1:
            # if the previous row is most filled between r_start and l_end.
            if lr_sum / (r_start -  l_end) > 0.4:
                filled_row[l_start:r_end+1] = fgval

    # Case: 3 contours
    if len(contour_start_end_pairs) == 3:
        start_0, end_0 = contour_start_end_pairs[0]
        start_1, end_1 = contour_start_end_pairs[1]
        start_2, end_2 = contour_start_end_pairs[2]
        # The contour can actually be vertices, so we need to check the
        # previous row to make sure.
        sum01 = np.sum(is_fgval[end_0:start_1])
        sum12 = np.sum(is_fgval[end_1:start_2])
        total_sum = np.sum(is_fgval)
        if total_sum == 0:
            filled_row[start_0:end_0] = fgval
            filled_row[start_1:end_1] = fgval
        else:
            if sum01 > max_thickness:
                filled_row[start_0:end_1] = fgval
            if sum12 > max_thickness:
                filled_row[start_1:end_2] = fgval
        filled_row[start_0:end_0+1] = fgval
        filled_row[start_1:end_1+1] = fgval
        filled_row[start_2:end_2+1] = fgval
    
    # Case: 4 contours
    if len(contour_start_end_pairs) == 4:
        start_0, end_0 = contour_start_end_pairs[0]
        start_1, end_1 = contour_start_end_pairs[1]
        start_2, end_2 = contour_start_end_pairs[2]
        start_3, end_3 = contour_start_end_pairs[3]
        # The contour can actually be vertices, so we need to check the
        # previous row to make sure.
        sum01 = np.sum(is_fgval[end_0:start_1+1])
        sum12 = np.sum(is_fgval[end_1:start_2+1])
        sum23 = np.sum(is_fgval[end_2:start_3+1])
        total_sum = np.sum(is_fgval)
        middle_sum = np.sum(is_fgval[end_1:start_2])
        if total_sum == 0:
            filled_row[start_0:end_1] = fgval
            filled_row[start_2:end_3] = fgval
        else:
            if sum01 + sum23 > sum12 or middle_sum/(start_2-end_1) < 0.7:
                filled_row[start_0:end_1+1] = fgval
                filled_row[start_2:end_3+1] = fgval
            else:
                filled_row[start_1:end_2+1] = fgval
        filled_row[start_0:end_0+1] = fgval
        filled_row[start_1:end_1+1] = fgval
        filled_row[start_2:end_2+1] = fgval
        filled_row[start_3:end_3+1] = fgval

    return filled_row


#######################################.#######################################
#                                                                             #
#                            PUT_SHAPE_INTO_FULL_XN                           #
#                                                                             #
###############################################################################
@jit
def put_shape_into_full_xn(filled_contour, xn, yn, x0, y0, bgval):
    """Put the filled_contour in the appropriate location."""
    output = np.full((yn, xn), bgval)
    
    # Because x0 and y0 are zero centered, we need to correct them.
    x0 = round(x0 + xn/2)
    y0 = round(y0 + yn/2)
    
    # Calculate the four corners of the filled_contour. These will be used to
    # index the output array.
    size = filled_contour.shape[0]
    vx_min = x0 - (size-1)//2
    vx_max = x0 + size//2
    hx_min = y0 - (size-1)//2
    hx_max = y0 + size//2

    # The filled_contour will be truncated if it is output of the output's
    # extent. 'fc' here standards for 'filled contour' because these will be
    # used to index the filled_contour array.
    fc_vx_min = max(-vx_min, 0)
    fc_vx_max = size - max(vx_max - xn + 1, 0)
    fc_hx_min = max(-hx_min, 0)
    fc_hx_max = size - max(hx_max - yn + 1, 0)
    
    # Make sure the corners don't go over bound.
    vx_min = clip(vx_min, 0, yn-1)
    vx_max = clip(vx_max, 0, yn-1)
    hx_min = clip(hx_min, 0, xn-1)
    hx_max = clip(hx_max, 0, xn-1)
    
    # Put the filled contour at the right position.
    output[vx_min:vx_max+1, hx_min:hx_max+1] = filled_contour[fc_vx_min:fc_vx_max,
                                                                fc_hx_min:fc_hx_max]

    return output


#######################################.#######################################
#                                                                             #
#                               MAKE_PASU_SHAPE                               #
#                                                                             #
###############################################################################
def make_pasu_shape(xn,yn,x0,y0,si,ri,fgval,bgval,size,plot=False):
    """
    Make the original Pasupathy and Connor (2001) shape set. There are 51
    unique shapes (indexed by 'si') and at most 8 unique angles (indexed by
    'ri').
    
    Tony Fu, Sep 6, 2022

    Parameters
    ----------
    xn & yn : int
        The horizontal and vertical dimensions of the output.
    x0 & y0 : int
        The horizontal and vertical positions of the shape. 0 is center, and
        y-axis is pointing downward.
    si : int
        Index of the Pasupathy shape. From 0 to 51.
    ri : int
        Index of the angle. The angle is ri * 45 degrees.
    fgval : float
        Foreground value.
    bgval : float
        Background value.
    size : int, optional
        The size of the output array (always square).
    plot : bool, optional
        Whether to plot the shape or not, by default False

    Returns
    -------
    filled_contour: numpy.ndarray
        The filled shape.
    """
    na = np.asarray(pasu_shape[si]) # A 1D list of control points as np array
    invec = na.reshape(int(len(na)/2),2)
    outvec = fvmax(invec, sample=size//2) # Get a more finely sampled set of points
    
    # Create a set of rotated coordinates
    rot = ri*45.0*np.pi/180.0    # Rotation in radians
    figX =  np.cos(rot)*outvec[:,0] + np.sin(rot)*outvec[:,1]
    figY = -np.sin(rot)*outvec[:,0] + np.cos(rot)*outvec[:,1]

    bool_contour = draw_bool_contour(figX, figY, (size, size))
    filled_contour = fill_contour(bool_contour, fgval, bgval)
    
    # Put the filled_contour in the appropriate location.
    output = put_shape_into_full_xn(filled_contour, xn, yn, x0, y0, bgval)
    
    if plot:
        plt.imshow(output, cmap='gray')
        plt.title(f"{int(rot*180/np.pi)}°")
        # ax = plt.gca()
        # ax.invert_yaxis()
        # plt.axis('off')
        # plt.grid()
    
    return output


if __name__ == "__main__":
    make_pasu_shape(100,100,0,0,48,1,fgval=1.0,bgval=-1.0,size=80,plot=True)
    plt.show()
    
    # for si in range(51):
    #     num_angles = pasu_shape_nrotu[si]
    #     plt.figure(figsize=(num_angles*3, 4))
    #     plt.suptitle(f"{si + 1}")
    #     for ri in range(num_angles): 
    #         plt.subplot(1,num_angles,ri+1)
    #         make_pasu_shape(100,100,0,0,si,ri,fgval=1.0,bgval=-1.0,size=50,plot=True)
    #     plt.show()


#######################################.#######################################
#                                                                             #
#                            MAKE_PASU_SHAPE_COLOR                            #
#                                                                             #
###############################################################################
def make_pasu_shape_color(xn,yn,x0,y0,si,ri,r1,g1,b1,r0,g0,b0,size,plot=False):
    s = np.empty((3, yn, xn), dtype='float32')
    s[0] = make_pasu_shape(xn,yn,x0,y0,si,ri,r1,r0,size,plot=False)
    s[1] = make_pasu_shape(xn,yn,x0,y0,si,ri,g1,g0,size,plot=False)
    s[2] = make_pasu_shape(xn,yn,x0,y0,si,ri,b1,b0,size,plot=False)
    if plot:
        img = (s - s.min()) / (s.max()-s.min())
        plt.imshow(np.transpose(img, (1,2,0)))
        plt.title(f"{int(ri*45.0)}°")
    return s


if __name__ == "__main__":
    make_pasu_shape_color(100,100,-30,35,48,1,1,-1,-1,0,0,0,size=100,plot=True)
    plt.show()


#######################################.#######################################
#                                                                             #
#                             STIMSET_DICT_PASU_BW                            #
#                                                                             #
###############################################################################
def stimset_dict_pasu_bw(xn, max_rf):
    """
    Parameters
    ----------
    xn      - horizontal and vertical image size (pix)\n
    max_rf  - theoretical size of RF (pix).
    """
    splist = []
    yn = xn
    
    fgval =  1.0  # Foreground luminance
    bgval = -1.0  # Background luminance
    
    # Size of the pasupathy shape.
    pasu_sizes = np.array([48/64, 24/64, 12/64]) * max_rf
    
    for pasu_size in pasu_sizes:
        ylist = xlist = stimset_gridx_map(max_rf, pasu_size)
        for i in xlist:
            for j in ylist:
                for si in range(51):
                    num_angles = pasu_shape_nrotu[si]
                    for ri in range(num_angles):
                        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "si":si, "ri":ri,
                             "fgval":fgval, "bgval":bgval, "size":max(round(pasu_size),1)}
                        splist.append(tp)
                        
                        # Now swap 'bgval' and 'fgval' to make opposite contrast
                        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "si":si, "ri":ri,
                             "fgval":bgval, "bgval":fgval, "size":max(round(pasu_size),1)}
                        splist.append(tp)
    return splist


#######################################.#######################################
#                                                                             #
#                            STIMSET_DICT_PASU_RGB7o                          #
#                                                                             #
###############################################################################
def stimset_dict_pasu_rgb7o(xn, max_rf):
    """
    Parameters
    ----------
    xn      - horizontal and vertical image size (pix)\n
    max_rf  - theoretical size of RF (pix).
    """
    splist = []
    yn = xn
    
    a0 = -1.0  # Amplitude low
    a1 =  1.0  # Amplitude high

    # Size of the pasupathy shape.
    pasu_sizes = np.array([48/64, 24/64, 12/64]) * max_rf

    colors = [(a1, a1, a1),
              (a0, a0, a0),
              (a1, a0, a0),
              (a0, a1, a0),
              (a0, a0, a1),
              (a1, a1, a0),
              (a1, a0, a1),
              (a0, a1, a1)]
    
    for pasu_size in pasu_sizes:
        ylist = xlist = stimset_gridx_map(max_rf, pasu_size)
        for i in xlist:
            for j in ylist:
                for si in range(51):
                    num_angles = pasu_shape_nrotu[si]
                    for ri in range(num_angles):
                        for r1, g1, b1 in colors:
                            tp = {"xn":xn, "yn":yn, "x0":i, "y0":j,
                                  "si":si, "ri":ri,
                                  "r1":r1, "g1":g1, "b1":b1,
                                  "r0":0, "g0":0, "b0":0,
                                  "size":max(round(pasu_size),1)}
                            splist.append(tp)
    return splist


if __name__ == "__main__":
    max_rf = 100
    xn = 120
    pasu_size = 30
    splist = stimset_dict_pasu_rgb7o(xn, max_rf)
    print(len(splist))


#######################################.#######################################
#                                                                             #
#                                 PASU_RUN_01b                                #
#                                                                             #
###############################################################################
def pasu_run_01b(splist, truncated_model, num_units, batch_size=100,
                   _debug=False, has_color=False):
    """
    Presents Pasupathy shapes and returns the center responses in array of
    dimension: [num_stim, num_units].

    Parameters
    ----------
    splist     - Pasupathy shape stimulus parameter list.\n
    truncated_model - neural network up to the layer of interest.\n
    num_units  - number of units/channels.\n
    batch_size - how many shapes to present at once.\n
    _debug     - if true, reduce the number of shapes and plot them.\n
    """
    num_stim = len(splist)
    xn = splist[0]['xn']
    yn = splist[0]['yn']
    center_responses = np.zeros((num_stim, num_units))

    shape_i = 0
    while (shape_i < num_stim):
        if _debug and shape_i > 200:
            break
        sys.stdout.write('\r')
        sys.stdout.write(f"Presenting {shape_i}/{num_stim} stimuli...")
        sys.stdout.flush()

        real_batch_size = min(batch_size, num_stim-shape_i)
        shape_batch = np.zeros((real_batch_size, 3, yn, xn))

        # Create a batch of bars.
        for i in range(real_batch_size):
            params = splist[shape_i + i]

            if has_color:
                new_shape = make_pasu_shape_color(params['xn'], params['yn'],
                                                  params['x0'], params['y0'],
                                                  params['si'], params['ri'],
                                                  params['r1'], params['g1'], params['b1'],
                                                  params['r0'], params['g0'], params['b0'],
                                                  params['size'], plot=False)
                shape_batch[i] = new_shape
            else:
                new_shape = make_pasu_shape(params['xn'], params['yn'],
                                            params['x0'], params['y0'],
                                            params['si'], params['ri'],
                                            params['fgval'], params['bgval'],
                                            params['size'], plot=False)
                # Replicate new bar to all color channel.
                shape_batch[i, 0] = new_shape
                shape_batch[i, 1] = new_shape
                shape_batch[i, 2] = new_shape

        # Present the patch of bars to the truncated model.
        with torch.no_grad():  # turn off gradient calculations for speed.
            y = truncated_model(torch.tensor(shape_batch).type('torch.FloatTensor').to(c.DEVICE))
        yc, xc = calculate_center(y.shape[-2:])
        center_responses[shape_i:shape_i+real_batch_size, :] = y[:, :, yc, xc].detach().cpu().numpy()
        shape_i += real_batch_size

    return center_responses


#######################################.#######################################
#                                                                             #
#                                MAKE_SHAPEMAP                                #
#                                                                             #
###############################################################################
def make_shapemaps(splist, center_responses, unit_i, _debug=False, has_color=False, 
                   num_shapes=500, response_thr=0.8, stim_thr=0.2):
    """
    Parameters
    ----------
    splist           - shape stimulus parameter list.\n
    center_responses - responses of center unit in [stim_i, unit_i] format.\n
    unit_i           - unit's index.\n
    response_thr     - shape w/ a reponse below response_thr * rmax will be
                       excluded.\n
    stim_thr         - shape pixels w/ a value below stim_thr will be excluded.\n
    _debug           - if true, print ranking info.\n

    Returns
    -------
    The weighted_max_map, weighted_min_map, non_overlap_max_map, and
    non_overlap_min_map of one unit.
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
    num_weighted_max_shapes = 0
    num_weighted_min_shapes = 0
    num_non_overlap_max_shapes = 0
    num_non_overlap_min_shapes = 0

    if _debug:
        print(f"unit {unit_i}: r_max: {r_max:7.2f}, max bar idx: {isort[::-1][:5]}")

    for max_shape_i in isort[::-1]:
        response = center_responses[max_shape_i, unit_i]
        params = splist[max_shape_i]
        # Note that the background color are set to 0, while the foreground
        # values are always positive.
        if has_color:
            new_shape = make_pasu_shape_color(params['xn'], params['yn'],
                                                  params['x0'], params['y0'],
                                                  params['si'], params['ri'],
                                                  params['r1'], params['g1'], params['b1'],
                                                  params['r0'], params['g0'], params['b0'],
                                                  params['size'], plot=False)
        else:
            new_shape = make_pasu_shape(params['xn'], params['yn'],
                                        params['x0'], params['y0'],
                                        params['si'], params['ri'],
                                        1, 0,
                                        params['size'], plot=False)
        # if (response - r_min) > r_range * response_thr:
        if num_weighted_max_shapes < num_shapes:
            has_included = add_non_overlap_map(new_shape, non_overlap_max_map, stim_thr)
            add_weighted_map(new_shape, weighted_max_map, (response - r_min)/r_range)
            # counts the number of bars in each map
            num_weighted_max_shapes += 1
            if has_included:
                num_non_overlap_max_shapes += 1
        else:
            break

    for min_shape_i in isort:
        response = center_responses[min_shape_i, unit_i]
        params = splist[min_shape_i]
        if has_color:
            new_shape = make_pasu_shape_color(params['xn'], params['yn'],
                                                  params['x0'], params['y0'],
                                                  params['si'], params['ri'],
                                                  params['r1'], params['g1'], params['b1'],
                                                  params['r0'], params['g0'], params['b0'],
                                                  params['size'], plot=False)
        else:
            new_shape = make_pasu_shape(params['xn'], params['yn'],
                                        params['x0'], params['y0'],
                                        params['si'], params['ri'],
                                        1, 0,
                                        params['size'], plot=False)
        # if (response - r_min) < r_range * (1 - response_thr):
        if num_weighted_min_shapes < num_shapes:
            has_included = add_non_overlap_map(new_shape, non_overlap_min_map, stim_thr)
            add_weighted_map(new_shape, weighted_min_map, (r_max - response)/r_range)
            # counts the number of bars in each map
            num_weighted_min_shapes += 1
            if has_included:
                num_non_overlap_min_shapes += 1
        else:
            break

    return weighted_max_map, weighted_min_map,\
           non_overlap_max_map, non_overlap_min_map,\
           num_weighted_max_shapes, num_weighted_min_shapes,\
           num_non_overlap_max_shapes, num_non_overlap_min_shapes


#######################################.#######################################
#                                                                             #
#                               PASU_BW_RUN_01b                               #
#                                                                             #
###############################################################################
def pasu_bw_run_01b(model, model_name, result_dir, _debug=False, batch_size=10,
                    num_shapes=500, conv_i_to_run=None):
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
    tb1_path = os.path.join(result_dir, f"{model_name}_pasu_tb1.txt")
    tb20_path = os.path.join(result_dir, f"{model_name}_pasu_tb20.txt")
    tb100_path = os.path.join(result_dir, f"{model_name}_pasu_tb100.txt")
    weighted_counts_path = os.path.join(result_dir, f"{model_name}_pasu_weighted_counts.txt")
    non_overlap_counts_path = os.path.join(result_dir, f"{model_name}_pasu_non_overlap_counts.txt")
    
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
        splist = stimset_dict_pasu_bw(xn, max_rf)
        
        if max_rf < 30:
            continue

        # Array initializations
        weighted_max_maps = np.zeros((num_units, max_rf, max_rf))
        weighted_min_maps = np.zeros((num_units, max_rf, max_rf))
        non_overlap_max_maps = np.zeros((num_units, max_rf, max_rf))
        non_overlap_min_maps = np.zeros((num_units, max_rf, max_rf))
        padding = (xn - max_rf)//2

        # Present all bars to the model
        truncated_model = get_truncated_model(model, layer_idx)
        center_responses = pasu_run_01b(splist, truncated_model,
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
        unit_batch_size = os.cpu_count() // 3
        unit_i = 0
        while (unit_i < num_units):
            if _debug and unit_i >= 20:
                break
            real_batch_size = min(unit_batch_size, num_units - unit_i)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(make_shapemaps,
                                       [splist for _ in range(real_batch_size)],
                                       [center_responses for _ in range(real_batch_size)],
                                       [i for i in np.arange(unit_i, unit_i + real_batch_size)],
                                       [_debug for _ in range(real_batch_size)],
                                       [False for _ in range(real_batch_size)],
                                       [num_shapes for _ in range(real_batch_size)],
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
        weighte_max_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_max_shapemaps.npy")
        weighted_min_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_min_shapemaps.npy")
        non_overlap_max_maps_path = os.path.join(result_dir, f"{layer_name}_non_overlap_max_shapemaps.npy")
        non_overlap_min_maps_path = os.path.join(result_dir, f"{layer_name}_non_overlap_min_shapemaps.npy")
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
        weighted_pdf_path = os.path.join(result_dir, f"{layer_name}_weighted_shapemaps.pdf")
        make_map_pdf(weighted_max_maps, weighted_min_maps, weighted_pdf_path)
        non_overlap_pdf_path = os.path.join(result_dir, f"{layer_name}_non_overlap_shapemaps.pdf")
        make_map_pdf(non_overlap_max_maps, non_overlap_min_maps, non_overlap_pdf_path)


#######################################.#######################################
#                                                                             #
#                               PASU_COLOR_RUN_01                             #
#                                                                             #
###############################################################################
def pasu_color_run_01(model, model_name, result_dir, _debug=False, batch_size=100):
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
    tb1_path = os.path.join(result_dir, f"{model_name}_pasu_color_tb1.txt")
    tb20_path = os.path.join(result_dir, f"{model_name}_pasu_color_tb20.txt")
    tb100_path = os.path.join(result_dir, f"{model_name}_pasu_color_tb100.txt")
    weighted_counts_path = os.path.join(result_dir, f"{model_name}_pasu_color_weighted_counts.txt")
    non_overlap_counts_path = os.path.join(result_dir, f"{model_name}_pasu_color_non_overlap_counts.txt")

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
        if conv_i < 8: continue
        layer_name = f"conv{conv_i + 1}"
        print(f"\n{layer_name}\n")
        # Get layer-specific info
        xn = xn_list[conv_i]
        layer_idx = layer_indices[conv_i]
        num_units = nums_units[conv_i]
        max_rf = max_rfs[conv_i][0]
        splist = stimset_dict_pasu_rgb7o(xn, max_rf)
        
        if max_rf < 30:
            continue

        # Array initializations
        weighted_max_maps = np.zeros((num_units, 3, max_rf, max_rf))
        weighted_min_maps = np.zeros((num_units, 3, max_rf, max_rf))
        non_overlap_max_maps = np.zeros((num_units, 3, max_rf, max_rf))
        non_overlap_min_maps = np.zeros((num_units, 3, max_rf, max_rf))
        padding = (xn - max_rf)//2

        # Present all bars to the model
        truncated_model = get_truncated_model(model, layer_idx)
        center_responses = pasu_run_01b(splist, truncated_model,
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
                results = executor.map(make_shapemaps,
                                       [splist for _ in range(real_batch_size)],
                                       [center_responses for _ in range(real_batch_size)],
                                       [i for i in np.arange(unit_i, unit_i + real_batch_size)],
                                       [_debug for _ in range(real_batch_size)],
                                       [True for _ in range(real_batch_size)],
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
        weighte_max_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_max_shapemaps.npy")
        weighted_min_maps_path = os.path.join(result_dir, f"{layer_name}_weighted_min_shapemaps.npy")
        non_overlap_max_maps_path = os.path.join(result_dir, f"{layer_name}_non_overlap_max_shapemaps.npy")
        non_overlap_min_maps_path = os.path.join(result_dir, f"{layer_name}_non_overlap_min_shapemaps.npy")
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
        weighted_pdf_path = os.path.join(result_dir, f"{layer_name}_weighted_shapemaps.pdf")
        make_map_pdf(np.transpose(weighted_max_maps, (0,2,3,1)),
                     np.transpose(weighted_min_maps, (0,2,3,1)),
                     weighted_pdf_path)
        non_overlap_pdf_path = os.path.join(result_dir, f"{layer_name}_non_overlap_shapemaps.pdf")
        make_map_pdf(np.transpose(non_overlap_max_maps, (0,2,3,1)),
                     np.transpose(non_overlap_min_maps, (0,2,3,1)),
                     non_overlap_pdf_path)


#######################################.#######################################
#                                                                             #
#                              MAKE_PASU_GRID_PDF                             #
#                                                                             #
###############################################################################
def make_pasu_grid_pdf(pdf_path, model):
    xn_list = xn_to_center_rf(model, image_size=(999,999))  # Get the xn just big enough.
    layer_indices, max_rfs = get_rf_sizes(model, (999, 999), layer_type=nn.Conv2d)
    num_layers = len(max_rfs)

    # Array of shape sizes
    pasu_size_ratios = np.array([48/64, 24/64, 12/64])
    pasu_size_ratio_strs = ['48/64', '24/64', '12/64']
    
    # Decide what shape to plot
    si = 16
    ri = 0

    with PdfPages(pdf_path) as pdf:
        for size_i, pasu_size_ratio in enumerate(pasu_size_ratios):
            plt.figure(figsize=(4*num_layers, 5))
            plt.suptitle(f"Shape No.{si}, Shape Size = {pasu_size_ratio_strs[size_i]}", fontsize=24)
            for conv_i, max_rf in enumerate(max_rfs):
                layer_name = f"conv{conv_i + 1}"
                layer_index = layer_indices[conv_i]
                # Get layer-specific info
                xn = xn_list[conv_i]
                max_rf = max_rf[0]

                # Set bar parameters
                pasu_size = max(round(pasu_size_ratio * max_rf),1)
                xlist = stimset_gridx_map(max_rf, pasu_size)

                # Plot the bar
                shape = make_pasu_shape(xn,xn,0,0,si,ri,1,0.5,pasu_size,plot=False)
                plt.subplot(1, num_layers, conv_i+1)
                plt.imshow(shape, cmap='gray', vmin=0, vmax=1)
                plt.title(f"{layer_name}\n(idx={layer_index}, maxRF={max_rf}, xn={xn})")

                # Plot the bar centers (i.e., the "grids").
                for y0 in xlist:
                    for x0 in xlist:
                        plt.plot(y0+xn/2, x0+xn/2, 'k.')

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
# if __name__ == "__main__":
#     model = models.alexnet()
#     model_name = 'alexnet'
#     model = models.vgg16()
#     model_name = 'vgg16'
#     model = models.resnet18()
#     model_name = 'resnet18'
#     pdf_path = os.path.join(c.RESULTS_DIR,'pasu','mapping', model_name,
#                             f'{model_name}_test_grid.pdf')
#     make_pasu_grid_pdf(pdf_path, model)


#######################################.#######################################
#                                                                             #
#                               MAKE_PASU_SET_PDF                             #
#                                                                             #
###############################################################################
def make_pasu_set_pdf(size):
    pdf_path = os.path.join(c.RESULTS_DIR,'pasu', f'pasu_shapes_size{size}.pdf')
    with PdfPages(pdf_path) as pdf:
        for si in range(51):
            num_angles = pasu_shape_nrotu[si]
            plt.figure(figsize=(num_angles*3, 4))
            plt.suptitle(f"{si + 1}")
            for ri in range(num_angles): 
                plt.subplot(1,num_angles,ri+1)
                make_pasu_shape(size,size,0,0,si,ri,fgval=1.0,bgval=-1.0,size=size,plot=True)
            pdf.savefig()
            plt.close()


# if __name__ == "__main__":
#     for size in [25, 50, 75, 100, 150, 200]:
#         make_pasu_set_pdf(size)
