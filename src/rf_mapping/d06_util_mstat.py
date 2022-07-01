import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import pickle

#######################################.#######################################
#                                                                             #
#                                MSTAT_READ_SPAR                              #
#                                                                             #
#  Read the list of stimulus parameter dictionaries.                          #
#                                                                             #
###############################################################################
def mstat_read_spar(infile_pre):
  
  dpath  = "map_spr/"     # Path where data files are stored
  infile = dpath + infile_pre + '_sp.pickle'
  
  with open(infile, 'rb') as f:   #  Read list of parameter dictionaries
    splist = pickle.load(f)
  
  print("  Length of stimulus parameter list:",len(splist))
  return splist

#######################################.#######################################
#                                                                             #
#                              MSTAT_READ_RESP                                #
#                                                                             #
#  Read the 2D array [units][stimuli] of response values for all units in     #
#  a layer that has been mapped.                                              #
#                                                                             #
###############################################################################
def mstat_read_resp(infile_pre):
  
  dpath  = "map_spr/"     # Path where data files are stored
  infile = dpath + infile_pre + '_sr.npy'
  
  rsp = np.load(infile)    # Read 2D numpy array or responses
  
  print("  Length of model response list:  ",len(rsp[0]))
  print("  Number of units:  ",len(rsp))
  return rsp

#######################################.#######################################
#                                                                             #
#                               MSTAT_HIST_SHOW                               #
#                                                                             #
#  2021 Jun 17                                                                #
#  This routine makes and shows response histograms for all units.            #
#                                                                             #
###############################################################################
def mstat_hist_show(infile_pre, logflag):
  
  rsp = mstat_read_resp(infile_pre)    # Read 2D numpy array or responses
  zn = len(rsp)

  rmin = rsp.min()
  rmax = rsp.min()
  rrng = rmax - rmin
  
  for zi in range(zn):
    # Build a string to make an informative title for the plot
    smin = "%.2f" % rsp[zi].min()
    smax = "%.2f" % rsp[zi].max()
    title = 'Unit ' + str(zi) + '  [' + smin + ', ' + smax + ']'

    #plt.xlim(xmin = -100, xmax = 100)
    if logflag == 1:
      plt.yscale('log', nonposy='clip')
    plt.hist(rsp[zi], bins=100)
    plt.title(title)
    plt.show()

#######################################.#######################################
#                                                                             #
#                             MSTAT_RESP_MIN_MAX                              #
#                                                                             #
#  2021 Jun 17                                                                #
#  This routine makes and shows response histograms for all units.            #
#                                                                             #
###############################################################################
def mstat_hist_show(infile_pre, logflag):
  
  rsp = mstat_read_resp(infile_pre)    # Read 2D numpy array or responses
  zn = len(rsp)

  rmin = rsp.min()
  rmax = rsp.min()
  rrng = rmax - rmin
  
  for zi in range(zn):
    # Build a string to make an informative title for the plot
    smin = "%.2f" % rsp[zi].min()
    smax = "%.2f" % rsp[zi].max()
    title = 'Unit ' + str(zi) + '  [' + smin + ', ' + smax + ']'

    #plt.xlim(xmin = -100, xmax = 100)
    if logflag == 1:
      plt.yscale('log', nonposy='clip')
    plt.hist(rsp[zi], bins=100)
    plt.title(title)
    plt.show()
