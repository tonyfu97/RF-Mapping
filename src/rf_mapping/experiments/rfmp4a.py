import math

import numpy as np
from numba import njit
import torch
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def clip(val, vmin, vmax):
    """Limit value to be vmin <= val <= vmax."""
    val = min(val, vmax)
    val = max(val, vmin)
    return val

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
    dval = fgval - bgval  # Luminance difference
    thetar = theta * math.pi/180.0

    # while theta >= 180.0:  # Make sure theta is in [0,180]  (DOES THIS MATTER?)
    #     theta -= 180.0
    # while theta < 0.0:
    #     theta += 180.0

    # Forward rot. matrix
    rot_a =  math.cos(thetar); rot_b = math.sin(thetar)
    rot_c = -math.sin(thetar); rot_d = math.cos(thetar)

    # Reverse rotation matrix
    a =  math.cos(-thetar); b = math.sin(-thetar)
    c = -math.sin(-thetar); d = math.cos(-thetar)
  
    # Maximum extent of unrotated bar corners in zero-centered frame:
    dx = 0.5*bwid
    dy = 0.5*blen

    # Rotate top-left corner from (-dx, dy) to (dx1, dy1)
    dx1 = -rot_a*dx + rot_c*dy
    dy1 = -rot_b*dx + rot_d*dy
    
    # Rotate top-right corner from (dx, dy) to (dx2, dy2)
    dx2 =  rot_a*dx + rot_c*dy
    dy2 =  rot_b*dx + rot_d*dy

    # Center of stimulus field
    xc = (xn-1.0)/2.0
    yc = (yn-1.0)/2.0

    # Maximum extent of rotated bar corners in zero-centered frame:
    maxx = laa + max(abs(dx1),abs(dx2))
    maxy = laa + max(abs(dy1),abs(dy2))

    # 
    ix0 = int(round( xc + x0 - maxx ))     # "xc + x0" is bar x-center in image
    ix1 = int(round( xc + x0 + maxx )) + 1 # Add 1 because of stupid python range
    iy0 = int(round( yc + y0 - maxy ))     # "yc + y0" is bar y-center in image
    iy1 = int(round( yc + y0 + maxy )) + 1 # Add 1 because of stupid python range

    ix0 = clip(ix0, 0, xn)
    ix1 = clip(ix1, 0, xn)
    iy0 = clip(iy0, 0, yn)
    iy1 = clip(iy1, 0, yn)
  
    for i in range(ix0,ix1):    # for i in range(0,xn):
        xx = (i-xc) - x0     # relative to bar center
        for j in range (iy0,iy1):    # for j in range (0,yn):
            yy = (j-yc) - y0     # relative to bar center
            
            x = a*xx + c*yy    # rotate back
            y = b*xx + d*yy
            
            # Compute distance from bar edge, 'db'
            if x > 0.0:
                dbx = bwid/2 - x   # +/- indicates inside/outside
            else:
                dbx = x + bwid/2
            
            if y > 0.0:
                dby = blen/2 - y    # +/- indicates inside/outside
            else:
                dby = y + blen/2
            
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
# TODO: refactor as:
# dgval = fgval - bgval
# return np.argwhere(s > (thres * dgval) + fgval)


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
