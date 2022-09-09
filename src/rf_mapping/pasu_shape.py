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
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['make_pasu_shape']

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


def clip(val, min_val, max_val):
    val = max(min_val, val)
    val = min(max_val, val)
    return val


def draw_bool_contour(x, y, shape):
    bool_contour = np.full((shape), False)
    for x, y in zip(x, y):
        x_trans = int((x + 2)/4 * shape[1])
        x_trans = clip(x_trans, 0, shape[1]-1)
        y_trans = int((y + 2)/4 * shape[0])
        y_trans = clip(y_trans, 0, shape[0]-1)
        bool_contour[y_trans, x_trans] = True
    return bool_contour


def fill_contour(bool_contour, fgval=1.0, bgval=-1.0):
    filled_contour = np.full((bool_contour.shape), bgval)
    filled_contour[bool_contour] = fgval
    prev_row = np.full(bool_contour[0].shape, bgval)
    for row_i, bool_row in enumerate(bool_contour):
        curr_row = fill_row(bool_row, fgval, bgval, prev_row)
        filled_contour[row_i, :] = curr_row
        prev_row = curr_row
    return filled_contour


def fill_row(bool_row, fgval, bgval, prev_row):
    """
    Return an array of the same length as bool_row but with 'fgval' values at
    the interior of the contours and 'bgval' values at the exterior.
    
    Tony Fu, September 6th, 2022
    """
    max_thickness = max(bool_row.shape[0] // 15, 1)
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


    # Case: 1 contour
    if len(contour_start_end_pairs) == 1:
        start, end = contour_start_end_pairs[0]
        filled_row[start:end+1] = fgval
        return filled_row
    
    # Case: 2 contours
    if len(contour_start_end_pairs) == 2:
        l_start, l_end = contour_start_end_pairs[0]
        r_start, r_end = contour_start_end_pairs[1]
        total_sum = np.sum(is_fgval)
        lr_sum = np.sum(is_fgval[l_end:r_start])
        filled_row[l_start:l_end+1] = fgval
        filled_row[r_start:r_end+1] = fgval
        if total_sum != 0 and lr_sum > 1:
            if lr_sum / (r_start -  l_end) > 0.2:
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
        if total_sum == 0:
            filled_row[start_0:end_1] = fgval
            filled_row[start_2:end_3] = fgval
        else:
            if sum01 + sum23 > sum12:
                filled_row[start_0:end_1] = fgval
                filled_row[start_2:end_3] = fgval
            else:
                filled_row[start_1:end_2] = fgval
        filled_row[start_0:end_0+1] = fgval
        filled_row[start_1:end_1+1] = fgval
        filled_row[start_2:end_2+1] = fgval
        filled_row[start_3:end_3+1] = fgval

    return filled_row


def make_pasu_shape(si, ri, output_size=100, plot=False):
    """
    Make the original Pasupathy and Connor (2001) shape set. There are 51
    unique shapes (indexed by 'si') and at most 8 unique angles (indexed by
    'ri').
    
    Tony Fu, Sep 6, 2022

    Parameters
    ----------
    si : int
        Index of the Pasupathy shape. From 0 to 51.
    ri : int
        Index of the angle. The angle is ri * 45 degrees.
    output_size : int, optional
        The size of the output array (always square), by default 100
    plot : bool, optional
        Whether to plot the shape or not, by default False

    Returns
    -------
    filled_contour: numpy.ndarray
        The filled shape.
    """
    na = np.asarray(pasu_shape[si]) # A 1D list of control points as np array
    invec = na.reshape(int(len(na)/2),2)
    outvec = fvmax(invec, sample=output_size//2) # Get a more finely sampled set of points
    
    # Create a set of rotated coordinates
    rot = ri*45.0*np.pi/180.0    # Rotation in radians
    figX =  np.cos(rot)*outvec[:,0] + np.sin(rot)*outvec[:,1]
    figY = -np.sin(rot)*outvec[:,0] + np.cos(rot)*outvec[:,1]

    bool_contour = draw_bool_contour(figX, figY, (output_size, output_size))
    filled_contour = fill_contour(bool_contour)
    
    if plot:
        plt.imshow(filled_contour, cmap='gray')
        plt.title(f"{int(rot*180/np.pi)}°")
        # ax = plt.gca()
        # ax.invert_yaxis()
        # plt.axis('off')
        # plt.grid()
    
    return filled_contour


if __name__ == "__main__":
    # make_pasu_shape(34, 1, output_size=100, plot=True)
    for si in range(51):
        num_angles = pasu_shape_nrotu[si]
        plt.figure(figsize=(num_angles*3, 4))
        plt.suptitle(f"{si + 1}")
        for ri in range(num_angles): 
            plt.subplot(1,num_angles,ri+1)
            make_pasu_shape(si, ri, output_size=100, plot=True)
        plt.show()
