"""
Code to generate stimulus bars.

Wyeth Bair, date unknown

Modified slightly by Tony Fu, July 1, 2022
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from numba import njit

#######################################.#######################################
#                                                                             #
#                                 STIMFR_LINE                                 #
#                                                                             #
###############################################################################
# njit slow it down by 6x.
def stimfr_line(xn,yn,xi,yi,fgval,bgval):
    #
    #  xn    - (int) length along 1st dimension of returned array
    #  yn    - (int) length along 2nd dimension of returned array
    #  xi    - (int) pixel index along 1st dimension
    #  yi    - (int) pixel index along 2nd deminsion
    #  fgval - (float) pixel luminance [0..1]
    #  bgval - (float) background luminance [0..1]
    #
    #  Return a numpy array that has a vertical or horizontal line
    #
    s = np.full((yn,xn), bgval, dtype='float32')  # Fill w/ BG value

    if (0 <= yi < yn):
        s[yi, :] = fgval
    elif (0 <= xi < xn):
        s[:, xi] = fgval
    return s

#######################################.#######################################
#                                                                             #
#                                 STIMFR_BAR                                  #
#                                                                             #
###############################################################################
@njit  # sped up by 5x
def stimfr_bar(xn,yn,x0,y0,theta,blen,bwid,laa,fgval,bgval):
    #
    #  Return a numpy array that has a bar on a background:
    #
    #  xn    - (int) horizontal width of returned array
    #  yn    - (int) vertical height of returned array
    #  x0    - (float) horizontal offset of center (pix)
    #  y0    - (float) vertical offset of center (pix)
    #  theta - (float) orientation (pix)
    #  blen  - (float) length of bar (pix)
    #  bwid  - (float) width of bar (pix)
    #  laa   - (float) length scale for anti-aliasing (pix)
    #  fgval - (float) bar luminance [0..1]
    #  bgval - (float) background luminance [0..1]
    #
  
    s = np.full((yn,xn), bgval, dtype='float32')  # Fill w/ BG value

    lo2 = blen/2.0
    wo2 = bwid/2.0
    dval = fgval - bgval  # Luminance difference

    while theta >= 180.0:  # Make sure theta is in [0,180]  (DOES THIS MATTER?)
        theta -= 180.0
    while theta < 0.0:
        theta += 180.0

    thetar = theta * math.pi/180.0
    rot_a =  math.cos(thetar); rot_b = math.sin(thetar)  # Forward rot. matrix
    rot_c = -math.sin(thetar); rot_d = math.cos(thetar)

    a =  math.cos(-thetar); b = math.sin(-thetar)   # reverse rotation matrix
    c = -math.sin(-thetar); d = math.cos(-thetar)

    xc = (xn-1.0)/2.0  # Center of stimulus field
    yc = (yn-1.0)/2.0
  
    # Compute maximum possible x- and y-extent of rotated bar corner points in
    #   zero-centered frame:
    dx = 0.5*bwid  # These are the only two quantities needed to represent
    dy = 0.5*blen  #   the first two points of the bar

    dx1 = -rot_a*dx + rot_c*dy  # Forward rotate the first two corners:
    dy1 = -rot_b*dx + rot_d*dy  #  The 1st corner point is (-dx,dy), so we've
    dx2 =  rot_a*dx + rot_c*dy  #    added a '-' on the 'dx' terms.
    dy2 =  rot_b*dx + rot_d*dy  #  The 2nd corner point is (dx,dy)

    maxx = laa + max(abs(dx1),abs(dx2))  # Max of ABS of 1st two x-coords
    maxy = laa + max(abs(dy1),abs(dy2))  # Max of ABS of 1st two y-coords

    ix0 = int(round( xc + x0 - maxx ))     # "xc + x0" is bar x-center in image
    ix1 = int(round( xc + x0 + maxx )) + 1 # Add 1 because of stupid python range
    iy0 = int(round( yc + y0 - maxy ))     # "yc + y0" is bar y-center in image
    iy1 = int(round( yc + y0 + maxy )) + 1 # Add 1 because of stupid python range

    if (ix1 > xn):  ix1 = xn
    if (ix0 <  0):  ix0 = 0
    if (iy1 > yn):  iy1 = yn
    if (iy0 <  0):  iy0 = 0
  
    for i in range(ix0,ix1):    # for i in range(0,xn):
        xx = (i-xc) - x0     # relative to bar center
        for j in range (iy0,iy1):    # for j in range (0,yn):
            yy = j-yc - y0     # relative to bar center
            
            x = a*xx + c*yy    # rotate back
            y = b*xx + d*yy
            
            # Compute distance from bar edge, 'db'
            if x > 0.0:
                dbx = wo2 - x   # +/- indicates inside/outside
            else:
                dbx = x + wo2
            
            if y > 0.0:
                dby = lo2 - y    # +/- indicates inside/outside
            else:
                dby = y + lo2
            
            if dbx < 0.0:      # x outside
                if dby < 0.0:
                    db = -math.sqrt(dbx*dbx + dby*dby)  # Both outside
                else:
                    db = dbx
            else:              # x inside
                if dby < 0.0:    #   y outside
                    db = dby
                else:            #   Both inside - take the smallest distance
                    if dby < dbx:
                        db = dby
                    else:
                        db = dbx
            
            if laa > 0.0:
                if db > laa:
                    f = 1.0     # This point is far inside the bar
                elif db < -laa:
                    f = 0.0     # This point is far outside the bar
                else:         # Use sinusoidal sigmoid
                    f = 0.5 + 0.5*math.sin(db/laa * 0.25*math.pi)
            else:
                if db >= 0.0:
                    f = 1.0   # inside
                else:
                    f = 0.0   # outside
            
            s[yn-1-j,i] += f * dval  #  add a fraction 'f' of the 'dval'
            #if (f < 0.01):
            #  s[yn-1-j,i] += 0.5   ### FOR TESTING THE mininmal box coords

    return s

#######################################.#######################################
#                                                                             #
#                            STIMFR_BAR_THRESH_LIST                           #
#                                                                             #
###############################################################################
@njit  # Sped up by 115x
def stimfr_bar_thresh_list(xn,yn,x0,y0,theta,blen,bwid,laa,thr):
    #
    #  Return a list of coordinates that are above threshold for this
    #    stimulus.
    #
    #  *** CODE MUST MATCH EXACTLY TO THAT OF:  'stimfr_bar'
    #

    lo2 = blen/2.0
    wo2 = bwid/2.0

    while theta >= 180.0:
        theta -= 180.0
    while theta < 0.0:
        theta += 180.0

    thetar = -theta * math.pi/180.0
    a =  math.cos(thetar); b = math.sin(thetar)   # rotation matrix
    c = -math.sin(thetar); d = math.cos(thetar)

    xc = (xn-1.0)/2.0  # Center of stimulus field
    yc = (yn-1.0)/2.0
  
    # Compute maximum box coordinates
    # *** WYETH - this is a crude estimate, and can be refined to use
    #     the width and length properly
    maxx = int(blen * abs(math.sin(thetar)) + bwid + 2.0*laa + 1)
    maxy = int(blen * abs(math.cos(thetar)) + bwid + 2.0*laa + 1)
    ix0 = int(xc - maxx/2.0 + x0)
    ix1 = int(ix0 + maxx + 1)
    if (ix1 > xn):  ix1 = xn
    if (ix0 <  0):  ix0 = 0
    iy0 = int(yc - maxy/2.0 + y0)
    iy1 = int(iy0 + maxy + 1)
    if (iy1 > yn):  iy1 = yn
    if (iy0 <  0):  iy0 = 0
    #print("X:  ", ix0, ix1)
    #print("Y:  ", iy0, iy1)
  
    clist = []  # Start with empty list of coordinates
    for i in range(ix0,ix1):    #for i in range(0,xn):
        xx = (i-xc) - x0     # relative to bar center
        for j in range (iy0,iy1):    #for j in range (0,yn):
            yy = j-yc - y0     # relative to bar center
            
            x = a*xx + c*yy    # rotate back
            y = b*xx + d*yy
            
            # Compute distance from bar edge, 'db'
            if x > 0.0:
                dbx = wo2 - x   # +/- indicates inside/outside
            else:
                dbx = x + wo2
            
            if y > 0.0:
                dby = lo2 - y    # +/- indicates inside/outside
            else:
                dby = y + lo2
            
            if dbx < 0.0:      # x outside
                if dby < 0.0:
                    db = -math.sqrt(dbx*dbx + dby*dby)  # Both outside
                else:
                    db = dbx
            else:              # x inside
                if dby < 0.0:    #   y outside
                    db = dby
                else:            #   Both inside - take the smallest distance
                    if dby < dbx:
                        db = dby
                    else:
                        db = dbx
        
            if laa > 0.0:
                if db > laa:
                    f = 1.0     # This point is far inside the bar
                elif db < -laa:
                    f = 0.0     # This point is far outside the bar
                else:         # Use sinusoidal sigmoid
                    f = 0.5 + 0.5*math.sin(db/laa * 0.25*math.pi)
            else:
                if db >= 0.0:
                    f = 1.0   # inside
                else:
                    f = 0.0   # outside
            
            if (f > thr):
                #clist.append([i,j])  OLD WAY
                clist.append([yn-1-j,i])

    return clist

#######################################.#######################################
#                                                                             #
#                               STIMFR_BAR_COLOR                              #
#                                                                             #
###############################################################################
# njit slowed down by 4x
def stimfr_bar_color(xn,yn,x0,y0,theta,blen,bwid,laa,r1,g1,b1,r0,g0,b0):
    #
    #  Return a numpy array that has a bar on a background:
    #
    #  xn    - (int) horizontal width of returned array
    #  yn    - (int) vertical height of returned array
    #  x0    - (float) horizontal offset of center (pix)
    #  y0    - (float) vertical offset of center (pix)
    #  theta - (float) orientation (pix)
    #  blen  - (float) length of bar (pix)
    #  bwid  - (float) width of bar (pix)
    #  laa   - (float) length scale for anti-aliasing (pix)
    #  r1,g1,b1 - bar color
    #  r0,g0,b0 - background color
    #
  
    sr = np.full((yn,xn), r0, dtype='float32')  # Fill w/ BG value
    sg = np.full((yn,xn), g0, dtype='float32')  # Fill w/ BG value
    sb = np.full((yn,xn), b0, dtype='float32')  # Fill w/ BG value

    lo2 = blen/2.0
    wo2 = bwid/2.0
    dvalr = r1-r0  # difference
    dvalg = g1-g0  # difference
    dvalb = b1-b0  # difference

    while theta >= 180.0:
        theta -= 180.0
    while theta < 0.0:
        theta += 180.0
  
    thetar = -theta * math.pi/180.0
    a =  math.cos(thetar); b = math.sin(thetar)   # rotation matrix
    c = -math.sin(thetar); d = math.cos(thetar)

    xc = (xn-1.0)/2.0  # Center of stimulus field
    yc = (yn-1.0)/2.0

    # Compute maximum box coordinates
    # *** WYETH - this is a crude estimate, and can be refined to use
    #     the width and length properly
    maxx = int(blen * abs(math.sin(thetar)) + bwid + 2.0*laa + 1)
    maxy = int(blen * abs(math.cos(thetar)) + bwid + 2.0*laa + 1)
    ix0 = int(xc - maxx/2.0 + x0)
    ix1 = int(ix0 + maxx + 1)
    if (ix1 > xn):  ix1 = xn
    if (ix0 <  0):  ix0 = 0
    iy0 = int(yc - maxy/2.0 + y0)
    iy1 = int(iy0 + maxy + 1)
    if (iy1 > yn):  iy1 = yn
    if (iy0 <  0):  iy0 = 0
    #print("X:  ", ix0, ix1)
    #print("Y:  ", iy0, iy1)
  
    for i in range(ix0,ix1):    #for i in range(0,xn):
        xx = (i-xc) - x0     # relative to bar center
        for j in range (iy0,iy1):    #for j in range (0,yn):
            yy = j-yc - y0     # relative to bar center
            
            x = a*xx + c*yy    # rotate back
            y = b*xx + d*yy
            
            # Compute distance from bar edge, 'db'
            if x > 0.0:
                dbx = wo2 - x   # +/- indicates inside/outside
            else:
                dbx = x + wo2
            
            if y > 0.0:
                dby = lo2 - y    # +/- indicates inside/outside
            else:
                dby = y + lo2
            
            if dbx < 0.0:      # x outside
                if dby < 0.0:
                    db = -math.sqrt(dbx*dbx + dby*dby)  # Both outside
                else:
                    db = dbx
            else:              # x inside
                if dby < 0.0:    #   y outside
                    db = dby
                else:            #   Both inside - take the smallest distance
                    if dby < dbx:
                        db = dby
                    else:
                        db = dbx
            
            if laa > 0.0:
                if db > laa:
                    f = 1.0     # This point is far inside the bar
                elif db < -laa:
                    f = 0.0     # This point is far outside the bar
                else:         # Use sinusoidal sigmoid
                    f = 0.5 + 0.5*math.sin(db/laa * 0.25*math.pi)
            else:
                if db >= 0.0:
                    f = 1.0   # inside
                else:
                    f = 0.0   # outside
            
            sr[yn-1-j,i] += f * dvalr  #  add a fraction 'f' of the 'dvalr'
            sg[yn-1-j,i] += f * dvalg  #  add a fraction 'f' of the 'dvalg'
            sb[yn-1-j,i] += f * dvalb  #  add a fraction 'f' of the 'dvalb'
  
    return sr, sg, sb


@njit  # Sped up by 35x
def im5k_draw_box(d,i0,j0,w):
    #      d  - numpy array [3][xn][xn] to over-write
    # (i0,j0) - initial point (pix)
    #      w  - size of box (pix)
    xn = len(d[0])
    for i in range(i0,i0+w):
        if (i >= 0) & (i <  xn):
            if (i == i0) | (i == i0+w-1):
                for j in range(j0,j0+w):
                    if (j >= 0) & (j <  xn):
                        d[0][i][j] = 1.0
                        d[1][i][j] = 0.0
                        d[2][i][j] = 0.0
            else:
                if (j0 >= 0) & (j0 <  xn):
                    d[0][i][j0] = 1.0
                    d[1][i][j0] = 0.0
                    d[2][i][j0] = 0.0
                if (j0+w-1 >= 0) & (j0+w-1 <  xn):
                    d[0][i][j0+w-1] = 1.0
                    d[1][i][j0+w-1] = 0.0
                    d[2][i][j0+w-1] = 0.0

#######################################.#######################################
#                                                                             #
#                                 STIMSET_1PIX                                #
#                                                                             #
###############################################################################
# njit slowed it down by 2x
def stimset_1pix(xn,yn,i0,i1,fgval,bgval):
    #
    #  Return a numpy array with one RGB stimulus with one pixel set
    #
    #  xn    - (int) horizontal width of returned array
    #  yn    - (int) vertical height of returned array
    #  i0    - (int) pixel index along 1st dimension
    #  i0    - (int) pixel index along 2nd deminsion
    #  fgval - (float) pixel luminance [0..1]
    #  bgval - (float) background luminance [0..1]
    #

    s = np.full((yn,xn), bgval, dtype='float32')  # Fill w/ BG value
    s[i0][i1] = fgval

    nstim = 1
    d = np.empty((nstim,3,xn,yn), dtype='float32')

    k = 0
    d[k][0] = s
    d[k][1] = s
    d[k][2] = s

    return d

#######################################.#######################################
#                                                                             #
#                                STIMSET_LINES                                #
#                                                                             #
###############################################################################
# Cannot use njit
def stimset_lines(n0,n1,fgval,bgval):
    #
    #  Return a full set of 1-pixel vertical and horizontal lines
    #
    #  n0    - (int) length along 1st dimension of returned array
    #  n1    - (int) length along 2nd dimension of returned array
    #  fgval - (float) pixel luminance [0..1]
    #  bgval - (float) background luminance [0..1]
    #

    nstim = n0 + n1
    d = np.empty((nstim,3,n0,n1), dtype='float32')

    k = 0
    for i in range(n1):
        s = stimfr_line(n0,n1,-1,i,fgval,bgval)
        d[k][0] = s
        d[k][1] = s
        d[k][2] = s
        k += 1

    for i in range(n0):
        s = stimfr_line(n0,n1,i,-1,fgval,bgval)
        d[k][0] = s
        d[k][1] = s
        d[k][2] = s
        k += 1

    return d

#######################################.#######################################
#                                                                             #
#                           STIMSET_BARMAP_SHOWRANGE                          #
#                                                                             #
#  Create an image to show the range of locations being mapped.               #
#                                                                             #
###############################################################################
def stimset_barmap_showrange(xn,rf,xmin,xmax,dx,ori0,dori,blen,bwid,aa):
    #
    #  xn     - horizontal and vertical image size
    #  rf     - max rf size
    #  xmin   - Minimum x-coord (pix)
    #  xmax   - Maximum x-coord (pix)
    #  dx     - Delta-x, position step size (pix)
    #  ori0   - Initial orientation (degr)
    #  dori   - Delta-orientation, step size (degr)
    #  blen   - Length of bar (pix)
    #  bwid   - Width of bar (pix)
    #  aa     - Anti-aliasing space constant (pix)
    #

    yn = xn  # Assuming image is square
    xlist = ylist = np.arange(xmin,xmax+1,dx)
    orilist = np.arange(ori0, 180.0, dori)

    #  This will be a 4D array with the following dimensions:
    #    [nstim, 3, xn, yn]
    #
    fgval =  1.0  # White
    bgval =  0.5  # Black

    n = len(xlist)
    print("  Stimulus array is",n,"X",n)

    d = np.empty((3,xn,yn), dtype='float32')

    i = int(n/2)
    o = orilist[0]
    s = stimfr_bar(xn,yn,xlist[i],ylist[i],o,blen,bwid,aa,fgval,bgval)

    d[0] = s
    d[1] = s
    d[2] = s
  
    im5k_draw_box(d,int(xn/2 - rf/2),int(xn/2 - rf/2),rf)
  
    for i in xlist:
        ii = int(round(i + xn/2))
        for j in ylist:
            jj = int(round(j + xn/2))
            d[0][ii][jj] = 0.0
            d[1][ii][jj] = 0.0
            d[2][ii][jj] = 0.0
  
    return d

#######################################.#######################################
#                                                                             #
#                            STIMSET_MAKE_BARMAP_BW                           #
#                                                                             #
#  Create a stimulus set for black and white bars, with three major           #
#  dimensions of variation:  x- and y-position, and orientation.              #
#                                                                             #
#  The input parameters specify the range of x-values to use (and these       #
#  are replicated to use for the y-range as well, and the delta-orientation   #
#  'dori' to use for varying orientation.                                     #
#                                                                             #
#  The other bar parameters are held fixed across the set:  length, width,    #
#  and anti-aliasing.                                                         #
#                                                                             #
###############################################################################
# Cannot use njit.
def stimset_make_barmap_bw(splist,xn,xmin,xmax,dx,ori0,dori,blen,bwid,aa):
    #
    #  splist - stimulus parameter list - APPEND TO THIS LIST
    #  xn     - horizontal and vertical image size
    #  xmin   - Minimum x-coord (pix)
    #  xmax   - Maximum x-coord (pix)
    #  dx     - Delta-x, position step size (pix)
    #  ori0   - Starting value for ori (degr)
    #  dori   - Delta-orientation, step size (degr)
    #  blen   - Length of bar (pix)
    #  bwid   - Width of bar (pix)
    #  aa     - Anti-aliasing space constant (pix)
    #

    yn = xn  # Assuming image is square
    xlist = ylist = np.arange(xmin,xmax+1,dx)
    orilist = np.arange(ori0, 180.0, dori)

    #  This will be a 4D array with the following dimensions:
    #    [nstim, 3, xn, yn]
    #
    fgval =  1.0  # Foreground luminance
    bgval = -1.0  # Background luminance

    nstim = len(xlist) * len(ylist) * len(orilist) * 2
    print("  Creating ", nstim, " stimuli.")

    d = np.empty((nstim,3,xn,yn), dtype='float32')

    k = 0
    for i in xlist:
        #print(i)
        for j in ylist:
            for o in orilist:
                s = stimfr_bar(xn,yn,i,j,o,blen,bwid,aa,fgval,bgval)
                d[k][0] = s
                d[k][1] = s
                d[k][2] = s
                k += 1
                tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen,
                    "wid":bwid, "aa":aa, "fgval":fgval, "bgval":bgval}
                splist.append(tp)
                
                # Now swap 'bgval' and 'fgval' to make opposite contrast
                s = stimfr_bar(xn,yn,i,j,o,blen,bwid,aa,bgval,fgval)
                d[k][0] = s
                d[k][1] = s
                d[k][2] = s
                k += 1
                
                tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen,
                    "wid":bwid, "aa":aa, "fgval":bgval, "bgval":fgval}
                splist.append(tp)
  
    #print("    done.")
    return d

#######################################.#######################################
#                                                                             #
#                            STIMSET_MAKE_BARMAP_COL                          #
#                                                                             #
#  Create a stimulus set for color bars, with three major                     #
#  dimensions of variation:  x- and y-position, and orientation.              #
#                                                                             #
#  The input parameters specify the range of x-values to use (and these       #
#  are replicated to use for the y-range as well, and the delta-orientation   #
#  'dori' to use for varying orientation.                                     #
#                                                                             #
#  The other bar parameters are held fixed across the set:  length, width,    #
#  and anti-aliasing.                                                         #
#                                                                             #
###############################################################################
def stimset_make_barmap_col(splist,xn,xmin,xmax,dx,ori0,dori,blen,bwid,aa,
                            r1,g1,b1,r0,g0,b0):
    #
    #  splist - stimulus parameter list - APPEND TO THIS LIST
    #  xn     - horizontal and vertical image size
    #  xmin   - Minimum x-coord (pix)
    #  xmax   - Maximum x-coord (pix)
    #  dx     - Delta-x, position step size (pix)
    #  ori0   - Starting orientation (degr)
    #  dori   - Delta-orientation, step size (degr)
    #  blen   - Length of bar (pix)
    #  bwid   - Width of bar (pix)
    #  aa     - Anti-aliasing space constant (pix)
    #  r1,g1,b1 - bar color
    #  r0,g0,b0 - background color

    yn = xn  # Assuming image is square
    xlist = ylist = np.arange(xmin,xmax+1,dx)
    orilist = np.arange(ori0, 180.0, dori)
    #orilist = np.arange(0, 180.0, dori)

    #  This will be a 4D array with the following dimensions:
    #    [nstim, 3, xn, yn]

    nstim = len(xlist) * len(ylist) * len(orilist) * 2
    print("  Creating ", nstim, " stimuli.")

    d = np.empty((nstim,3,xn,yn), dtype='float32')

    k = 0
    for i in xlist:
        #print(i)
        for j in ylist:
            for o in orilist:
                d[k][0], d[k][1], d[k][2] = \
                stimfr_bar_color(xn,yn,i,j,o,blen,bwid,aa,r1,g1,b1,r0,g0,b0)
                k += 1
                tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen,
                    "wid":bwid, "aa":aa, "r1":r1, "g1":g1, "b1":b1,
                    "r0":r0, "g0":g0, "b0":b0}
                splist.append(tp)
                
                # Now swap bar and background colors
                d[k][0], d[k][1], d[k][2] = \
                stimfr_bar_color(xn,yn,i,j,o,blen,bwid,aa,r0,g0,b0,r1,g1,b1)
                k += 1
                tp = {"xn":xn, "yn":yn, "x0":i, "y0":j, "theta":o, "len":blen,
                    "wid":bwid, "aa":aa, "r1":r0, "g1":g0, "b1":b0,
                    "r0":r1, "g0":g1, "b0":b1}
                splist.append(tp)
  
    return d

#######################################.#######################################
#                                                                             #
#                                 BARMAP_RUN4                                 #
#                                                                             #
#  Run 4 sets of bars at multi-scale grid.                                    #
#                                                                             #
###############################################################################
def barmap_run4(splist,srlist,m,xn,max_rf,arat):
    #
    #  splist - stimulus parameter list
    #  srlist - response list
    #  m      - model
    #  xn     - stimulus image size
    #  max_rf - maximum RF size (pix)
    #  arat   - aspect ratio of bar, e.g., 1/8
    #
    n = 4
    blen = np.array([48/64 * max_rf,     #  Array of bar lengths
                    24/64 * max_rf,
                    12/64 * max_rf,
                    6/64 * max_rf])

    dx = blen / 2.0                     #  Array of grid spacing

    xmax = np.empty(n)
    for i in range(n):
        xmax[i] = round((max_rf/dx[i]) / 2.0) * dx[i]

        aa = 0.5       # Antialias distance (pix)

        o0 =  0.0      # Initial orientation (deg)
        do = 22.5      # orientation step size (deg)

    #xxn = int(round(1.5*xn))
    for i in range(n):  ### SHOW THE GRID
        t = stimset_barmap_showrange(xn,max_rf,-xmax[i],xmax[i],dx[i],o0,do,
                                        blen[i],arat*blen[i],aa)
        plt.imshow(np.transpose(t, (1, 2, 0)))  # Make RGB dimension be last
        plt.show()

    for i in range(n):
        d = stimset_make_barmap_bw(splist,xn,-xmax[i],xmax[i],dx[i],o0,do,blen[i],
                                    arat*blen[i],aa)
        mrfmap_run_01(srlist,d,m)   # Show stimuli 'd' to model 'm', update 'srlist'

#######################################.#######################################
#                                                                             #
#                              BARMAP_RUN4_AS_7                               #
#                                                                             #
#  Run 4 sets of bars at multi-scale grid, but for the fourth set, divide     #
#  it into 4 groups, for a total of 7 stimulus sets.                          #
#                                                                             #
###############################################################################
def barmap_run4_as_7(splist,srlist,m,xn,max_rf,arat,showflag):
    #
    #  splist - stimulus parameter list
    #  srlist - response list
    #  m      - model
    #  xn     - stimulus image size
    #  max_rf - maximum RF size (pix)
    #  arat   - aspect ratio of bar, e.g., 1/8
    #  showflag - 1-show the grid, 0-do not show
    #
    n = 7
    blen = np.array([48/64 * max_rf,     #  Array of bar lengths
                    24/64 * max_rf,
                    12/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf])

    dori = np.array([22.5, 22.5, 22.5, 90.0, 90.0, 90.0, 90.0]) # Ori increment
    ori0 = np.array([ 0.0,  0.0,  0.0,  0.0, 22.5, 45.0, 67.5]) # Initial ori.

    dx = blen / 2.0                     #  Array of grid spacing

    xmax = np.empty(n)
    for i in range(n):
        xmax[i] = round((max_rf/dx[i]) / 2.0) * dx[i]

    aa = 0.5       # Antialias distance (pix)

    xxn = int(round(1.5*xn))
    if (showflag == 1):
        for i in range(n):  ### SHOW THE GRID
            o0 = ori0[i]   # Initial orientation (deg)
            do = dori[i]   # orientation step size (deg)
            #t = stimset_barmap_showrange(xxn,max_rf,-xmax[i],xmax[i],dx[i],o0,do,
            #                             blen[i],arat*blen[i],aa)
            plt.imshow(np.transpose(t, (1, 2, 0)))  # Make RGB dimension be last
            plt.show()
  
    for i in range(n):
        o0 = ori0[i]   # Initial orientation (deg)
        do = dori[i]   # orientation step size (deg)
        d = stimset_make_barmap_bw(splist,xn,-xmax[i],xmax[i],dx[i],o0,do,blen[i],
                                    arat*blen[i],aa)
        mrfmap_run_01(srlist,d,m)   # Show stimuli 'd' to model 'm', update 'srlist'

#######################################.#######################################
#                                                                             #
#                              BARMAP_RUN4_AS_12                              #
#                                                                             #
#  Run 4 sets of bars at multi-scale grid, but for the fourth set, divide     #
#  it into 4 groups, for a total of 7 stimulus sets.                          #
#                                                                             #
###############################################################################
def barmap_run4_as_12(splist,srlist,m,xn,max_rf,arat,showflag):
    #
    #  splist - stimulus parameter list
    #  srlist - response list
    #  m      - model
    #  xn     - stimulus image size
    #  max_rf - maximum RF size (pix)
    #  arat   - aspect ratio of bar, e.g., 1/8
    #  showflag - 1-show the grid, 0-do not show
    #
    n = 12
    blen = np.array([48/64 * max_rf,     #  Array of bar lengths
                    24/64 * max_rf,
                    12/64 * max_rf,
                    12/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf])

    # Ori increment
    dori = np.array([22.5, 22.5, 45.0, 45.0, 180.0, 180.0, 180.0, 180.0,
                    180.0, 180.0, 180.0, 180.0])
    ori0 = np.array([ 0.0,  0.0,  0.0, 22.5, 0.0, 22.5, 45.0, 67.5,
                    90.0, 112.5, 135.0, 157.5]) # Initial ori.

    dx = blen / 2.0                     #  Array of grid spacing

    xmax = np.empty(n)
    for i in range(n):
        xmax[i] = round((max_rf/dx[i]) / 2.0) * dx[i]
  
    aa = 0.5       # Antialias distance (pix)
  
    #xxn = int(round(1.5*xn))
    if (showflag == 1):
        for i in range(n):  ### SHOW THE GRID
            o0 = ori0[i]   # Initial orientation (deg)
            do = dori[i]   # orientation step size (deg)
            t = stimset_barmap_showrange(xxn,max_rf,-xmax[i],xmax[i],dx[i],o0,do,
                                        blen[i],arat*blen[i],aa)
            plt.imshow(np.transpose(t, (1, 2, 0)))  # Make RGB dimension be last
            plt.show()

    for i in range(n):
        o0 = ori0[i]   # Initial orientation (deg)
        do = dori[i]   # orientation step size (deg)
        d = stimset_make_barmap_bw(splist,xn,-xmax[i],xmax[i],dx[i],o0,do,blen[i],
                                    arat*blen[i],aa)
        mrfmap_run_01(srlist,d,m)   # Show stimuli 'd' to model 'm', update 'srlist'

#######################################.#######################################
#                                                                             #
#                             BARMAP_RUN4_AS_7C                               #
#                                                                             #
#  Like 'barmap_run4_as_7' except here we need to run multiple color          #
#  combinations.                                                              #
#                                                                             #
###############################################################################
def barmap_run4_as_7c(splist,srlist,m,xn,max_rf,arat,showflag,r1,g1,b1,
                      r0,g0,b0):
    #
    #  splist   - stimulus parameter list
    #  srlist   - response list
    #  m        - model
    #  xn       - stimulus image size
    #  max_rf   - maximum RF size (pix)
    #  arat     - aspect ratio of bar, e.g., 1/8
    #  showflag - 1-show the grid, 0-do not show
    #  r1,g1,b1 - bar color
    #  r0,g0,b0 - background color
    #

    print("  Color: ",r1,g1,b1," on: ",r0,g0,b0)

    n = 7
    blen = np.array([48/64 * max_rf,     #  Array of bar lengths
                    24/64 * max_rf,
                    12/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf])

    dori = np.array([22.5, 22.5, 22.5, 90.0, 90.0, 90.0, 90.0]) # Ori increment
    ori0 = np.array([ 0.0,  0.0,  0.0,  0.0, 22.5, 45.0, 67.5]) # Initial ori.

    dx = blen / 2.0                     #  Array of grid spacing

    xmax = np.empty(n)
    for i in range(n):
        xmax[i] = round((max_rf/dx[i]) / 2.0) * dx[i]

    aa = 0.5       # Antialias distance (pix)

    #xxn = int(round(1.5*xn))
    if (showflag == 1):
        for i in range(n):  ### SHOW THE GRID
            o0 = ori0[i]   # Initial orientation (deg)
            do = dori[i]   # orientation step size (deg)
            t = stimset_barmap_showrange(xxn,max_rf,-xmax[i],xmax[i],dx[i],o0,do,
                                        blen[i],arat*blen[i],aa)
            plt.imshow(np.transpose(t, (1, 2, 0)))  # Make RGB dimension be last
            plt.show()
    
    for i in range(n):
        o0 = ori0[i]   # Initial orientation (deg)
        do = dori[i]   # orientation step size (deg)
        d = stimset_make_barmap_col(splist,xn,-xmax[i],xmax[i],dx[i],o0,do,blen[i],
                                    arat*blen[i],aa,r1,g1,b1,r0,g0,b0)
    
        #d = stimset_make_barmap_bw(splist,xn,-xmax[i],xmax[i],dx[i],o0,do,blen[i],
        #                           arat*blen[i],aa)
        
        mrfmap_run_01(srlist,d,m)  # Show stimuli 'd' to model 'm', update 'srlist'

#######################################.#######################################
#                                                                             #
#                             BARMAP_RUN4_AS_7C                               #
#                                                                             #
#  Like 'barmap_run4_as_7' except here we need to run multiple color          #
#  combinations.                                                              #
#                                                                             #
###############################################################################
def barmap_run4_as_12c(splist,srlist,m,xn,max_rf,arat,showflag,r1,g1,b1,
                       r0,g0,b0):
    #
    #  splist   - stimulus parameter list
    #  srlist   - response list
    #  m        - model
    #  xn       - stimulus image size
    #  max_rf   - maximum RF size (pix)
    #  arat     - aspect ratio of bar, e.g., 1/8
    #  showflag - 1-show the grid, 0-do not show
    #  r1,g1,b1 - bar color
    #  r0,g0,b0 - background color
    #

    print("  Color: ",r1,g1,b1," on: ",r0,g0,b0)

    n = 12
    blen = np.array([48/64 * max_rf,     #  Array of bar lengths
                    24/64 * max_rf,
                    12/64 * max_rf,
                    12/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf,
                    6/64 * max_rf])
  
    dori = np.array([22.5, 22.5, 45.0, 45.0, 180.0, 180.0, 180.0, 180.0,
                    180.0, 180.0, 180.0, 180.0]) # Ori increment
    ori0 = np.array([ 0.0,  0.0,  0.0,  22.5, 0.0, 22.5, 45.0, 67.5,
                    90.0, 112.5, 135.0, 157.5]) # Initial ori.

    dx = blen / 2.0                     #  Array of grid spacing

    xmax = np.empty(n)
    for i in range(n):
        xmax[i] = round((max_rf/dx[i]) / 2.0) * dx[i]
  
    aa = 0.5       # Antialias distance (pix)
  
    #xxn = int(round(1.5*xn))
    if (showflag == 1):
        for i in range(n):  ### SHOW THE GRID
            o0 = ori0[i]   # Initial orientation (deg)
            do = dori[i]   # orientation step size (deg)
            t = stimset_barmap_showrange(xxn,max_rf,-xmax[i],xmax[i],dx[i],o0,do,
                                        blen[i],arat*blen[i],aa)
            plt.imshow(np.transpose(t, (1, 2, 0)))  # Make RGB dimension be last
            plt.show()
  
    for i in range(n):
        o0 = ori0[i]   # Initial orientation (deg)
        do = dori[i]   # orientation step size (deg)
        d = stimset_make_barmap_col(splist,xn,-xmax[i],xmax[i],dx[i],o0,do,blen[i],
                                    arat*blen[i],aa,r1,g1,b1,r0,g0,b0)

        #d = stimset_make_barmap_bw(splist,xn,-xmax[i],xmax[i],dx[i],o0,do,blen[i],
        #                           arat*blen[i],aa)

        mrfmap_run_01(srlist,d,m)  # Show stimuli 'd' to model 'm', update 'srlist'

#######################################.#######################################
#                                                                             #
#                                 STIMSET_PLOT                                #
#                                                                             #
#  After running the 'stimset_make...' routine above, this can be used to     #
#    plot the k-th stimulus in the set 'd'.                                   #
#                                                                             #
###############################################################################
def stimset_plot(d,k):
    # Plot the 'k' stimulus of the set
    npimg = np.empty_like(d[k])
    npimg[:,:] = d[k]
    print(npimg.max())
    if (npimg.min() < 0.0):
        npimg *= 0.5
        npimg += 0.5
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Make RGB dimension be last
    plt.show()

#
#  Plot the stimulus for the k-th set of parameters in 'splist'
#
def splist_plot(splist,k):
    d = splist[k]      # Parameter dictionary for this stimulus
    s = stimfr_bar(d['xn'],d['yn'],d['x0'],d['y0'],d['theta'],
                   d['len'],d['wid'],d['aa'],d['fgval'],d['bgval'])
    #plt.imshow(np.transpose(s))  # Make RGB dimension be last
    plt.imshow(s)  # Make RGB dimension be last
    plt.show()

#
#  Run the stimulus and update response list
#
def mrfmap_run_01(srlist,dstim,m):
    #
    #  srlist - response list to append
    #   stimd - stimulus data as numpy array
    #

    nstim = len(dstim)                # Number of stimuli
    ttstim = torch.tensor(dstim)      # Convert numpy to torch.tensor
    r = m.forward(ttstim)             # r[stim, zfeature, x, y] Run the model
    #print(r.shape)

    mi = int((len(r[0,0,0]) - 1)/2)   # index for middle of spatial array
    zn = len(r[0])                    # Number of unique units
    td = r[:,:,mi,mi].cpu().detach()  # Get responses for middle units only
    nd = td.numpy()                   # Convert back to numpy

    #  Store the responses for each unit, for each stimulus in 'srlist',
    #     appending to this list.
    if len(srlist) == 0:  # If 'srlist' empty, create empty lists for each 'zn'
        for i in range(zn):
            srlist.append([])

    for i in range(zn):         # For each unique unit (feature index)
        for j in range(nstim):
            srlist[i].append(nd[j,i])


#######################################.#######################################
#                                                                             #
#                               MRFMAP_MAKE_MAP_1                             #
#                                                                             #
#   *** These comments need to be organized...                                #
#                                                                             #
#  Create a sorted index list 'isort' that goes from the largest to
#     smallest response in 'srlist[z]'
#     Take the largest response:
#       rmax = srlist[z][ isort[0] ]
#     and define a threshold response to be some fraction 'frac' of this:
#       rmin = r_thr * rmax   # threhold response level
#
#     Set our MRF map 'mrf[xn][yn]'  to all zeros (assuming images are
#       'xn' by 'yn' pixels.
#
#   ...
#
###############################################################################
def mrfmap_make_map_1(xn,srlist,splist,zi,r_thr,s_thr):
    #  xn     - size of map (pix)
    #  srlist - responses
    #  splist - stimulus parameters
    #  zi     - index of unit within response list
    #  r_thr  - threshold fraction of response
    #  s_thr  - threshold fraction of stimulus

    tr = srlist[zi]              # a pointer to response array for 'zi' unit
    isort = np.argsort(tr)       # Get indices that sort the list (least 1st)
    isort = np.flip(isort)       # Flip array so that largest comes first
    rmax = tr[isort[0]]          # Maximum response in list
    rmin = r_thr * rmax          # threshold response value
    mrfmap = np.full((xn,xn),0)  # MRF map starts with all zeros

    #print("  Unit",zi," Max response: ",rmax)

    #splist_plot(splist,isort[0])   # Plot best stimulus

    # For i = 0,1,2,... and while the srlist[z][i] > rmin
    #   i.  Generate a stimulus image 'img' from the parameters in 'splist[i]'
    #  ii.  For each pixel (x,y) in 'img' that differs from the background,
    #       if *all* such pixels are zero in 'mrf[x][y]', then set all of these
    #       pixels in mrf[x][y] to 1.

    nstim = len(splist)
    for si in range(nstim):      # for each stimulus
        if tr[isort[si]] < rmin:   # Stop loop at first subthreshold response
            break
        
        pd = splist[isort[si]]     # Parameter dictionary for this stimulus
        clist = stimfr_bar_thresh_list(xn,xn,pd['x0'],pd['y0'],pd['theta'],
                                    pd['len'],pd['wid'],pd['aa'],s_thr)
        cnt = 0
        for c in clist:
            cnt += mrfmap[c[0],c[1]]  # Add values in current mrfmap where stim > thr
        
        if (cnt == 0):  # If this stimulus does not touch any part of the MRF
            for c in clist:
                mrfmap[c[0],c[1]] = 1   # Set all stimulated positions to '1'
            #print("  Added stimlus ",isort[si])
    
    return mrfmap, rmax

#######################################.#######################################
#                                                                             #
#                                MAPSTAT_COMR_1                               #
#                                                                             #
#  For a 2D array 'map', compute the (x,y) center of mass and the radius      #
#  that contains a fraction 'f' of the area of the map.                       #
#                                                                             #
###############################################################################
def mapstat_comr_1(map,f):
    #
    #
    xn = len(map)
    list0 = np.sum(map,1)  # Sum along axis 1, to ultimately get COM on axis 0
    list1 = np.sum(map,0)  # Sum along axis 0, to ultimately get COM on axis 1
    total = np.sum(list0)  # Overall total weight of entire map
    xvals = np.arange(xn)  # X-values (pix)
    prod0 = xvals*list0
    prod1 = xvals*list1
    if (total > 0.0):
        com0 = np.sum(prod0)/total
        com1 = np.sum(prod1)/total
    else:
        com0 = com1 = -1
    
    #print("  Center of mass: ",com0,com1)
    
    dist2 = []  # empty list to hold squared distances from COM
    magn = []   # empty list to hold magnitude
    for i in range(xn):
        di2 = (i-com0)*(i-com0)
        for j in range(xn):
            dj2 = (j-com1)*(j-com1)
        if (map[i,j] > 0.0):
            dist2.append(di2 + dj2)
            magn.append(map[i,j])
    
    isort = np.argsort(dist2)   # Get list of indices that sort list (least 1st)
    
    if (com0 == -1):
        return -1, -1, -1
    
    n = len(dist2)
    
    # Go down the sorted list, adding up the magnitudes, until the fractional
    #  criterion is exceeded.  Compute the radius for the final point added.
    #
    tot = 0.0
    for i in range(n):  # for each non-zero position in the map
        k = isort[i]      # Get index 'k' of the next point closest to the COM
        tot += magn[k]
        if (tot/total >= f):
            break
    
    radius = np.sqrt(dist2[k])
    #print("  Radius: ",radius)
    return com0, com1, radius

#######################################.#######################################
#                                                                             #
#                                  RFMAP_TOP_4A                               #
#                                                                             #
###############################################################################
def rfmap_top_4a(m,xn,max_rf,netid,layname,showflag):
    #
    #  m        - model to run
    #  xn       - size (pix) of stimulus image
    #  max_rf   - maximum RF size (pix) for units being mapped
    #  netid    - e.g. 'n01'
    #  layname  - e.g. 'conv2'
    #  showflag - 1-show RF maps, 0-do not show
    #
    
    splist = []  # splist[i] - list of parameters that generated stimulus 'i'
    srlist = []  # srlist[z][i] - responses for unit 'z' for each stimulus 'i'
    
    barmap_run4_as_7(splist,srlist,m,xn,max_rf,1/2,0)  # 3 apsect ratios
    barmap_run4_as_7(splist,srlist,m,xn,max_rf,1/5,0)
    barmap_run4_as_7(splist,srlist,m,xn,max_rf,1/10,0)
    
    print("  Length of stimulus parameter list:",len(splist))
    print("  Length of model response list:  ",len(srlist[0]))
    # Note, there are 33,984 stimuli
    
    statfile = netid + '_stat_' + layname + '_mrf_4a.txt'
    pdffile = netid + '_' + layname + '_p4a.pdf'
    
    mid = (xn-1)/2   # Middle pixel of the map
    r_thr = 0.10
    s_thr = 0.20
    zn = len(srlist)
    with PdfPages(pdffile) as pdf:
        with open(statfile, 'a') as outfile:
            for zi in range(zn):
                mrfmap, rmax = mrfmap_make_map_1(xn,srlist,splist,zi,r_thr,s_thr)
                com0, com1, radius = mapstat_comr_1(mrfmap,0.9)
                xp = "%3d %8.4f %5.1f %5.1f %5.1f" % (zi,rmax,com0-mid,com1-mid,radius)
                print(xp)
                wn = outfile.write(xp + "\n")  # 'wn' is number of bytes written?
                #
                plt.imshow(mrfmap)   # 2D array as image
                plt.title(netid + "  " + layname + "   Unit " + str(zi))
                pdf.savefig()
                if (showflag == 1):
                    plt.show()

#######################################.#######################################
#                                                                             #
#                                   RUN_6opp                                  #
#                                                                             #
#  Run four bar sizes (and 3 aspect ratios) but at six color opponent         #
#  configurations.                                                            #
#                                                                             #
###############################################################################
def run_6opp(splist,srlist,m,xn,max_rf,arat,bwflag,showflag):
    #
    #  splist - stimulus parameter list
    #  srlist - response list
    #  m      - model
    #  xn     - stimulus image size
    #  max_rf - maximum RF size (pix)
    #  arat   - aspect ratio of bar, e.g., 1/8
    #  bwflag - 1-use BW bars in addition to the six color opponent cases
    #  showflag - 1-show the grid, 0-do not show
    #
    print("  ASPECT RATIO", arat)  
    #                                                           BAR   Background
    #                                                          r g b   r g b
    barmap_run4_as_7c(splist,srlist,m,xn,max_rf,arat,showflag, 1,0,0,  0,1,0)
    barmap_run4_as_7c(splist,srlist,m,xn,max_rf,arat,showflag, 1,0,0,  0,0,1)
    barmap_run4_as_7c(splist,srlist,m,xn,max_rf,arat,showflag, 0,1,0,  0,0,1)
    barmap_run4_as_7c(splist,srlist,m,xn,max_rf,arat,showflag, 1,1,0,  0,0,1)
    barmap_run4_as_7c(splist,srlist,m,xn,max_rf,arat,showflag, 1,0,1,  0,1,0)
    barmap_run4_as_7c(splist,srlist,m,xn,max_rf,arat,showflag, 0,1,1,  1,0,0)
    
    if (bwflag == 1):
        print("  Black and white")
        barmap_run4_as_7(splist,srlist,m,xn,max_rf,arat,showflag)

#######################################.#######################################
#                                                                             #
#                                 RUN_6opp12                                  #
#                                                                             #
#  Run four bar sizes (and 3 aspect ratios) but at six color opponent         #
#  configurations.                                                            #
#                                                                             #
###############################################################################
def run_6opp12(splist,srlist,m,xn,max_rf,arat,bwflag,showflag):
    #
    #  splist - stimulus parameter list
    #  srlist - response list
    #  m      - model
    #  xn     - stimulus image size
    #  max_rf - maximum RF size (pix)
    #  arat   - aspect ratio of bar, e.g., 1/8
    #  bwflag - 1-use BW bars in addition to the six color opponent cases
    #  showflag - 1-show the grid, 0-do not show
    #
    print("  ASPECT RATIO", arat)  
    #                                                           BAR   Background
    #                                                           r g b   r g b
    barmap_run4_as_12c(splist,srlist,m,xn,max_rf,arat,showflag, 1,0,0,  0,1,0)
    barmap_run4_as_12c(splist,srlist,m,xn,max_rf,arat,showflag, 1,0,0,  0,0,1)
    barmap_run4_as_12c(splist,srlist,m,xn,max_rf,arat,showflag, 0,1,0,  0,0,1)
    barmap_run4_as_12c(splist,srlist,m,xn,max_rf,arat,showflag, 1,1,0,  0,0,1)
    barmap_run4_as_12c(splist,srlist,m,xn,max_rf,arat,showflag, 1,0,1,  0,1,0)
    barmap_run4_as_12c(splist,srlist,m,xn,max_rf,arat,showflag, 0,1,1,  1,0,0)
    
    if (bwflag == 1):
        print("  Black and white")
        barmap_run4_as_12(splist,srlist,m,xn,max_rf,arat,showflag)

#######################################.#######################################
#                                                                             #
#                                RFMAP_TOP_4C6O                               #
#                                                                             #
#  Run four bar sizes (and 3 aspect ratios) but at six color opponent         #
#  configurations.                                                            #
#                                                                             #
###############################################################################
def rfmap_top_4c6o(m,xn,max_rf,netid,layname,showflag):
    #
    #  m        - model to run
    #  xn       - size (pix) of stimulus image
    #  max_rf   - maximum RF size (pix) for units being mapped
    #  netid    - e.g. 'n01'
    #  layname  - e.g. 'conv2'
    #  showflag - 1-show RF maps, 0-do not show
    #
    
    splist = []  # splist[i] - list of parameters that generated stimulus 'i'
    srlist = []  # srlist[z][i] - responses for unit 'z' for each stimulus 'i'
    
    #run_6opp(splist,srlist,m,xn,max_rf,1/2, 1,0)
    #run_6opp(splist,srlist,m,xn,max_rf,1/5, 1,0)
    #run_6opp(splist,srlist,m,xn,max_rf,1/10,1,0)
    run_6opp12(splist,srlist,m,xn,max_rf,1/2, 1,0)
    run_6opp12(splist,srlist,m,xn,max_rf,1/5, 1,0)
    run_6opp12(splist,srlist,m,xn,max_rf,1/10,1,0)
    
    print("  Length of stimulus parameter list:",len(splist))
    print("  Length of model response list:  ",len(srlist[0]))
    # Note, there are 237,888 stimuli
    
    statfile = netid + '_stat_' + layname + '_mrf_4c6o.txt'
    pdffile = netid + '_' + layname + '_p4c6o.pdf'
    
    mid = (xn-1)/2   # Middle pixel of the map
    r_thr = 0.10
    s_thr = 0.20
    zn = len(srlist)
    with PdfPages(pdffile) as pdf:
        with open(statfile, 'a') as outfile:
            for zi in range(zn):
                mrfmap, rmax = mrfmap_make_map_1(xn,srlist,splist,zi,r_thr,s_thr)
                com0, com1, radius = mapstat_comr_1(mrfmap,0.9)
                xp = "%3d %8.4f %5.1f %5.1f %5.1f" % (zi,rmax,com0-mid,com1-mid,radius)
                print(xp)
                wn = outfile.write(xp + "\n")  # 'wn' is number of bytes written?
                #
                plt.imshow(mrfmap)   # 2D array as image
                plt.title(netid + "  " + layname + "   Unit " + str(zi))
                pdf.savefig()
                if (showflag == 1):
                    plt.show()
