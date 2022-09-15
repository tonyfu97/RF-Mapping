import math
import numpy as np

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
  #
  #  splist  - stimulus parameter list - APPEND TO THIS LIST
  #  xn      - horizontal and vertical image size
  #  xlist   - list of x-coordinates (pix)
  #  orilist - list of orientation values (degr)
  #  blen    - Length of bar (pix)
  #  bwid    - Width of bar (pix)
  #  aa      - Anti-aliasing space constant (pix)
  #
  
  yn = xn        # Assuming image is square
  ylist = xlist  # Use same coordinates for y grid locations
  #print(xlist)   # Testing
  
  fgval =  1.0  # Foreground luminance
  bgval = -1.0  # Background luminance
  
  nstim = len(xlist) * len(ylist) * len(orilist) * 2
  print("  Creating ", nstim, " stimulus dictionary entries.")
  
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
#                            STIM_DAPP_BAR_XYO_RGB7O                          #
#                                                                             #
#  For each x, y and orientation in the lists,                                #
#  creates dictionary entries for 14 color/BW conditions, 7 pairs in both     #
#  ordering of foreground an background.                                      #
#                                                                             #
###############################################################################
def stim_dapp_bar_xyo_rgb7o(splist,xn,xlist,orilist,blen,bwid,aa):
  #
  #  splist  - stimulus parameter list - APPEND TO THIS LIST
  #  xn      - horizontal and vertical image size
  #  xlist   - list of x-coordinates (pix)
  #  orilist - list of orientation values (degr)
  #  blen    - Length of bar (pix)
  #  bwid    - Width of bar (pix)
  #  aa      - Anti-aliasing space constant (pix)
  #
  
  yn = xn        # Assuming image is square
  ylist = xlist  # Use same coordinates for y grid locations
  #print(xlist)   # Testing
  
  a0 = -1.0  # Amplitude low
  a1 =  1.0  # Amplitude high
  
  nstim = len(xlist) * len(ylist) * len(orilist) * 2
  print("  Creating ", nstim, " stimulus dictionary entries.")
  
  for i in xlist:
    for j in ylist:
      for o in orilist:
        # 111 v 000          B&W
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a1, "g1":a1, "b1":a1,"r0":a0, "g0":a0, "b0":a0}
        splist.append(tp)
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a0, "g1":a0, "b1":a0,"r0":a1, "g0":a1, "b0":a1}
        splist.append(tp)
        
        # 100 v 010      red-green
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a1, "g1":a0, "b1":a0,"r0":a0, "g0":a1, "b0":a0}
        splist.append(tp)
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a0, "g1":a1, "b1":a0,"r0":a1, "g0":a0, "b0":a0}
        splist.append(tp)
        
        # 100 v 001      red-blue
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a1, "g1":a0, "b1":a0,"r0":a0, "g0":a0, "b0":a1}
        splist.append(tp)
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a0, "g1":a0, "b1":a1,"r0":a1, "g0":a0, "b0":a0}
        splist.append(tp)
        
        # 010 v 001      green-blue
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a0, "g1":a1, "b1":a0,"r0":a0, "g0":a0, "b0":a1}
        splist.append(tp)
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a0, "g1":a0, "b1":a1,"r0":a0, "g0":a1, "b0":a0}
        splist.append(tp)
        
        # 110 v 001     yellow-blue
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a1, "g1":a1, "b1":a0,"r0":a0, "g0":a0, "b0":a1}
        splist.append(tp)
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a0, "g1":a0, "b1":a1,"r0":a1, "g0":a1, "b0":a0}
        splist.append(tp)
        
        # 101 v 010    purple-green
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a1, "g1":a0, "b1":a1,"r0":a0, "g0":a1, "b0":a0}
        splist.append(tp)
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a0, "g1":a1, "b1":a0,"r0":a1, "g0":a0, "b0":a1}
        splist.append(tp)
        
        # 011 v 100    cyan-red
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a0, "g1":a1, "b1":a1,"r0":a1, "g0":a0, "b0":a0}
        splist.append(tp)
        tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen, "aa":aa,
              "wid":bwid, "r1":a1, "g1":a0, "b1":a0,"r0":a0, "g0":a1, "b0":a1}
        splist.append(tp)


#######################################.#######################################
#                                                                             #
#                             STIM_DAPP_SIN_XYO_BW                            #
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
def stim_dapp_sin_xyo_bw(splist,xn,xlist,orilist,diam,sf):
  #
  #  splist  - stimulus parameter list - APPEND TO THIS LIST
  #  xn      - horizontal and vertical image size
  #  xlist   - list of x-coordinates (pix)
  #  orilist - list of orientation values (degr)
  #  diam    - grating diameter / size (pix)
  #  sf      - spatial frequency
  #
  
  yn = xn        # Assuming image is square
  ylist = xlist  # Use same coordinates for y grid locations
  #print(xlist)   # Testing
  
  con   =  1.0  # Contrast
  bgval =  0.0  # Background luminance
  
  phase = np.array([ 0.0, 90.0, 180.0, 270.0 ])
  
  nstim = len(xlist) * len(ylist) * len(orilist) * 2
  print("  Creating ", nstim, " stimulus dictionary entries.")
  
  for i in xlist:
    for j in ylist:
      for o in orilist:
        for ph in phase:
          tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "size":diam,
                "sf":sf, "ph":ph, "contrast":con, "bgval":bgval}
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
  #
  #  max_rf - maximum RF size (pix)
  #  blen   - bar length (pix)
  #
  
  dx = blen / 2.0                       # Grid spacing is 1/2 of bar length
  xmax = round((max_rf/dx) / 2.0) * dx  # Max offset of grid point from center
  xlist = np.arange(-xmax,xmax+1,dx)
  
  return xlist


#######################################.#######################################
#                                                                             #
#                            STIMSET_DICT_RFMP_4A                             #
#                                                                             #
#  Return the stimulus parameter dictionary with the appropriate entries      #
#  for the entire stimulus set for RF mapping paradigm "4a".                  #
#                                                                             #
###############################################################################
def stimset_dict_rfmp_4a(xn,max_rf):
  #
  #  xn     - stimulus image size (pix)
  #  max_rf - maximum RF size (pix)
  #
  
  splist = []  # List of dictionary entries, one per stimulus image
  
  #  There are 4 bar lengths
  barlen = np.array([48/64 * max_rf,    #  Array of bar lengths
                     24/64 * max_rf,
                     12/64 * max_rf,
                      6/64 * max_rf])
  
  #  There are 3 aspect ratios
  arat = np.array([1/2, 1/5, 1/10])   # Array of apsect ratios
  
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
#s = stimset_dict_rfmp_4a(79,51)
# stimset_show_stim_p4a(s[0]):

#######################################.#######################################
#                                                                             #
#                            STIMSET_DICT_RFMP_4C7O                           #
#                                                                             #
#  Return the stimulus parameter dictionary with the appropriate entries      #
#  for the entire stimulus set for RF mapping paradigm "4c7o".                #
#                                                                             #
###############################################################################
def stimset_dict_rfmp_4c7o(xn,max_rf):
  #
  #  xn     - stimulus image size (pix)
  #  max_rf - maximum RF size (pix)
  #
  
  splist = []  # List of dictionary entries, one per stimulus image
  
  #  There are 4 bar lengths
  barlen = np.array([48/64 * max_rf,    #  Array of bar lengths
                     24/64 * max_rf,
                     12/64 * max_rf,
                      6/64 * max_rf])
  
  #  There are 3 aspect ratios
  arat = np.array([1/2, 1/5, 1/10])   # Array of apsect ratios
  
  #  There are 16 orientations, even spaced around 360 deg starting at 0 deg
  orilist = np.arange(0.0, 180.0, 22.5)
  
  #  This constant sets how much blurring occurs at the edges of the bars
  aa =  0.5      # Antialias distance (pix)
  
  for bl in barlen:
    xlist = stimset_gridx_barmap(max_rf,bl)
    for ar in arat:
      stim_dapp_bar_xyo_rgb7o(splist,xn,xlist,orilist,bl,ar*bl,aa)
  
  print("  Length of stimulus parameter list:",len(splist))
  
  return splist

#
#  HERE IS AN EXAMPLE OF HOW TO CALL THE CODE ABOVE:
#
#s = stimset_dict_rfmp_4c7o(79,51)
# stimset_show_stim_p4c7o(s[0])

#######################################.#######################################
#                                                                             #
#                            STIMSET_DICT_RFMP_SIN1                           #
#                                                                             #
#  Return the stimulus parameter dictionary with the appropriate entries      #
#  for the entire stimulus set for RF mapping paradigm "sin1".                #
#                                                                             #
###############################################################################
def stimset_dict_rfmp_sin1(xn,max_rf):
  #
  #  xn     - stimulus image size (pix)
  #  max_rf - maximum RF size (pix)
  #
  
  splist = []  # List of dictionary entries, one per stimulus image
  
  #  There are 4 diams
  diam = np.array([48/64 * max_rf,    #  Array of grating diameters
                   24/64 * max_rf,
                   12/64 * max_rf,
                    6/64 * max_rf])

  sf = np.array([ 0.02, 0.04, 0.08, 0.16, 0.32 ])
  
  #  There are 16 orientations, even spaced around 360 deg starting at 0 deg
  orilist = np.arange(0.0, 180.0, 22.5)
  
  for d in diam:
    xlist = stimset_gridx_barmap(max_rf,d)
    for s in sf:
      stim_dapp_sin_xyo_bw(splist,xn,xlist,orilist,d,s)
  
  print("  Length of stimulus parameter list:",len(splist))
  
  return splist

#
#  HERE IS AN EXAMPLE OF HOW TO CALL THE CODE ABOVE:
#
#s = stimset_dict_rfmp_sin1(79,51)
#stimset_show_stim_sin(s[0])
