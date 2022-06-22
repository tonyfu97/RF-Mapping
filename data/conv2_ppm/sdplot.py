def mytest(x, y):
   print(x)
   
   print(y)




# Can create a pdf containing a response map for each unit (mean, sd, min, or max).
# Produces a file containing the unit name, x center of mass (COM) for top user-defined
# percent of response data and its corresponding y COM, x COM for the remaining bottom
# (100-percent)% of data and corresponding y COM. Cutoff discrimination: high split
# includes cutoff percentage values if needed (ex: 10% of 17x17 matrix = 28.9, cutoff for
# top 10% high split is size 29). Populates data array from input files in a manner
# appropriate to plotting with matplotlib.
#
#
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import scipy.optimize as opt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import copy
from sklearn import mixture
from numpy.polynomial.polynomial import polyfit
import scipy.stats


#
# Change 'axflag' to 0 for x-axis and 1 for y-axis.
#
def newcalcCOM(data, axis, degrees, axflag):
compress = (np.sum(data, axis=axflag))
if axflag == 1:
  compress = np.flip(compress)
valsum = np.sum(compress)
overlay = axis*compress
if valsum != 0:
  deg = (np.sum(overlay))/valsum
  dot = deg*2.0 + degrees*2
else:
  dot = 0
  deg = 0

return dot, deg
#
#
# Draw plot
#
#
#
#
def DrawMap(unit, data, ptitle, dimensions, x_axis, degrees, fname, x_center, y_center, x_periph, y_periph, top):
cwd = os.getcwd()
x_dot, xdeg = newcalcCOM(data, x_axis, degrees, 0)
y_dot, ydeg = newcalcCOM(data, x_axis, degrees, 1)
plt.imshow(data, cmap=plt.cm.Blues_r)
plt.xlabel('x position (deg)')
plt.ylabel('y position (deg)')
plt.xticks(np.arange(0, dimensions, 1), x_axis, fontsize=4)
plt.yticks(np.arange(0, dimensions, 1), np.flip(x_axis), fontsize=4)
plt.title(ptitle + ": Unit " + str(i))
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=7)
d = dimensions - 1
# Plotting dots for comparing centers of mass (before Gaussian fitting)
#plt.plot(x_center, d - y_center, 'ok', markersize=4)
#plt.plot(x_center, d - y_center, 'ow', markersize=2)
#plt.plot(x_dot, d - y_dot, 'or', markersize=2)
#plt.plot(x_periph, d - y_periph, 'om', markersize=2)
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(str(cwd) + '/vis/mean_resp/' + str(fname))
plt.close()
return xdeg, ydeg
# Produces top_matrix, where values less than user-defined percent of max becomes 0,
# anything greater than percentcutoff% is retained. Produces bottom_matrix containing
# the compliment bottom (1-percentcutoff)%. Cutoff discrimination: high split
# includes cutoff percentage values if needed (ex: 10% of 17x17 matrix = 28.9, cutoff for
# top 10% high split is size 29).
def SplitMatrix(matrix, percentcutoff, dimensions):
# intialize matrices to be split
bottom_matrix = np.array(matrix)
top_matrix = np.array(matrix)
# dimensions of 'matrix':
dim = dimensions*dimensions
# number of elements cutoff by 'percentcutoff' boundary
cutoff = int(round(dim*percentcutoff/100))
# find indices of the largest numbers in the matrix
high_vals = np.dstack(np.unravel_index(np.argsort(bottom_matrix.ravel())[-cutoff:], (dimensions, dimensions)))
# Low Values (bottom (100-percent)%): set largest numbers in matrix = 0
for i in range(cutoff):
bottom_matrix[high_vals[0,i,0],high_vals[0,i,1]] = 0
# find indices of the smallest numbers in the matrix
low_vals = np.dstack(np.unravel_index(np.argsort(top_matrix.ravel())[:(dim-cutoff)], (dimensions, dimensions)))
# High Values (top percent%): set smallest numbers in matrix = 0
for i in range (dim-cutoff):
top_matrix[low_vals[0,i,0], low_vals[0,i,1]] = 0
return top_matrix, bottom_matrix
#
def calcDistance(x1, y1, x2, y2):
distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
return distance
#
# Give it arrays containing max mean response, xCOM, and yCOM for each unit
def CreateScatterplot(max_resp, xCOM_deg_list, yCOM_deg_list):
max_norm2_resp = np.max(max_resp)
norm_max = max_resp / max_norm2_resp
scatter = plt.scatter(xCOM_deg_list, yCOM_deg_list, c=norm_max)
cbar = plt.colorbar(scatter)
cbar.set_label('Max Response', rotation=90)
plt.xlabel('x COM (deg)')
plt.ylabel('y COM (deg)')
ptitle = 'Mean Response COM'
plt.title(ptitle)
cwd = os.getcwd()
fname = 'COM_scatterplot.pdf'
plt.savefig(str(cwd) + '/vis/' + fname)
plt.close()

def TotalMeanScatter(max_resp, xCOM_deg_list, yCOM_deg_list):
max_norm2_resp = np.max(max_resp)
norm_max = max_resp / max_norm2_resp
scatter = plt.scatter(xCOM_deg_list, yCOM_deg_list, c=norm_max)
cbar = plt.colorbar(scatter)
cbar.set_label('Max Response', rotation=90)
plt.xlabel('x COM (deg)')
plt.ylabel('y COM (deg)')
ptitle = 'Mean Response COM'
plt.title(ptitle)
cwd = os.getcwd()
fname = 'Total_COM_scatterplot.pdf'
plt.savefig(str(cwd) + '/vis/' + fname)
plt.close()
#
#define model function and pass independant variables x and y as a list
# tup: data, amplitude of gauss, yo is center; sigma_x, sigma_y are the x and y spreads of the blob; parameters
# a, b, c are for rotation by a clockwise angle (theta)
def twoD_Gaussian(tup, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
x = tup[0, :]
y = tup[1, :]
xo = float(xo)
yo = float(yo)
a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
return g.ravel()
#
def Map2DGaussian(data, dimensions, degrees, increment, max_mean, gauss_file_2d):
x1 = np.linspace(-degrees, degrees, num = dimensions)
x2 = x1
x1, x2 = np.meshgrid(x1, x2)
#create data for an initial guess
radius = 5
top_xCOM_mu = 0
top_yCOM_mu = 0
# how well does this function's values (twoD_Gaussian) predicted for this x,y (gaussian fxn applied to (x, y)-> some values) match data?
# popt = (1D array) optimal values for the parameters so that the sum of the squared redisduals of f(xdata, *popt) - ydata is minimized
# pcov = (2D array) estimated covariance of popt. diagonals provide variance of the parameter estimate.
xdata = np.vstack((x1.ravel(), x2.ravel()))
# xdata is x and y dimension axes. has 2 rows and 289 columns
# y data should be 1D array of length n*m (in this case 17x17 = 289)
# has values associated with
ydata = data.ravel()
tup = [xdata, ydata]
initial_guess = (max_mean, 0, 0, radius, radius, 0, 0)
popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, ydata, p0=initial_guess)
# data_fitted is just the perfect fit based off of popt's estimation of best parameters
# popt creates ideal params: amplitude, xo, yo, sigma_x, sigma_y, theta, offset
data_fitted = twoD_Gaussian(xdata, *popt)


data2 = data.flatten()
abs_err = data_fitted - data2
squared_err = np.square(abs_err)
MSE = np.mean(squared_err)

max_val = popt[0]
x_deg = popt[1]
y_deg = popt[2]
sigma_x = popt[3]
sigma_y = popt[4]
theta = popt[5]
offset = popt[6]

fig, ax = plt.subplots(1, 1)
im = ax.imshow(data, cmap=plt.cm.jet, origin='bottom', extent=(x1.min(), x1.max(), x2.min(), x2.max()))
cbar = ax.figure.colorbar(im)
cbar.ax.set_ylabel('Mean Response', rotation=90)
plt.xlabel('x position (deg)')
plt.ylabel('y position (deg)')
title = 'Topographic Map 2D Gaussian'
plt.title(title)
# ax.contour
# x dimensions of meshgrid, y dimensions of meshgrid, height values over which contour drawn
ax.contour(x1, x2, data_fitted.reshape((dimensions, dimensions)), 8, colors='w')
cwd = os.getcwd()
plt.savefig(str(cwd) + '/vis/gauss/' + str(gauss_file_2d))
plt.close()

# 3D Visualization

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#z = data_fitted.reshape((dimensions,dimensions))
#ax.plot_surface(x1, x2, z, cmap='plasma')
#ax.set_zlim(0,np.max(z)+2)
#plt.show()
#plt.close()
# xo, yo is center; sigma_x, sigma_y are the x and y spreads of the blob; parameters
# a, b, c are for rotation by a clockwise angle (theta)
return max_val, x_deg, y_deg, sigma_x, sigma_y, theta, offset, MSE
def Bimodal_twoD_Gaussian(tup, amp1, xo1, yo1, sigma_x1, sigma_y1, theta1, offset1, amp2, xo2, yo2, sigma_x2, sigma_y2, theta2, offset2):
bimod = twoD_Gaussian(tup, amp1, xo1, yo1, sigma_x1, sigma_y1, theta1, offset1) + twoD_Gaussian(tup, amp2, xo2, yo2, sigma_x2, sigma_y2, theta2, offset2)
return bimod
def MapBimodalGauss(data, dimensions, degrees, increment, max_mean, bimod_gauss_outf):
x1 = np.linspace(-degrees, degrees, num = dimensions)
x2 = x1
x1, x2 = np.meshgrid(x1, x2)
#create data for an initial guess
radius = 1
radius = 1
top_xCOM_mu = 0
top_yCOM_mu = 0
# how well does this function's values (twoD_Gaussian) predicted for this x,y (gaussian fxn applied to (x, y)-> some values) match data?
# popt = (1D array) optimal values for the parameters so that the sum of the squared redisduals of f(xdata, *popt) - ydata is minimized
# pcov = (2D array) estimated covariance of popt. diagonals provide variance of the parameter estimate.
xdata = np.vstack((x1.ravel(), x2.ravel()))
# xdata is x and y dimension axes. has 2 rows and 289 columns
# y data should be 1D array of length n*m (in this case 17x17 = 289)
# has values associated with
ydata = data.ravel()
tup = [xdata, ydata]
right_center = np.array([degrees/2, 0])
left_center = np.array([-degrees/2, 0])
initial_guess = (max_mean, left_center[0], left_center[1], radius, radius, 0, 0, max_mean, right_center[0], right_center[1], radius, radius, 0, 0)
popt, pcov = opt.curve_fit(Bimodal_twoD_Gaussian, xdata, ydata, p0=initial_guess)
# data_fitted is just the perfect fit based off of popt's estimation of best parameters
# popt returns ideal parameters: amp1, xo1, yo1, sigma_x1, sigma_y1, theta1, offset1,
# amp2, xo2, yo2, sigma_x2, sigma_y2, theta2, offset2. popt = fitted parameters
data_fitted = Bimodal_twoD_Gaussian(xdata, *popt)

data2 = data.flatten()
abs_err = data_fitted - data2
squared_err = np.square(abs_err)
MSE = np.mean(squared_err)

max_val1 = popt[0]
x1_deg = popt[1]
y1_deg = popt[2]
sigma_x1 = popt[3]
sigma_y1 = popt[4]
theta1 = popt[5]
offset1 = popt[6]
max_val2 = popt[7]
x2_deg = popt[8]
y2_deg = popt[9]
sigma_x2 = popt[10]
sigma_y2 = popt[11]
theta2 = popt[12]
offset2 = popt[13]
bimod_distance = calcDistance(x1_deg, y1_deg, x2_deg, y2_deg)


fig, ax = plt.subplots(1, 1)
im = ax.imshow(data, cmap=plt.cm.jet, origin='bottom', extent=(x1.min(), x1.max(), x2.min(), x2.max()))
cbar = ax.figure.colorbar(im)
cbar.ax.set_ylabel('Mean Response', rotation=90)
plt.xlabel('x position (deg)')
plt.ylabel('y position (deg)')
title = 'Topographic Map 2D Gaussian - Bimodality'
plt.title(title)
ax.contour(x1, x2, data_fitted.reshape((dimensions, dimensions)), 8, colors='w')
cwd = os.getcwd()
plt.savefig(str(cwd) + '/vis/b_gauss/' + str(bimod_gauss_outf))
plt.close()
#
# 3D Visualization
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#z = data_fitted.reshape((dimensions,dimensions))
#ax.plot_surface(x1, x2, z, cmap='plasma')
#ax.set_zlim(0,np.max(z)+2)
#plt.show()
#plt.close()
return max_val1, x1_deg, y1_deg, sigma_x1, sigma_y1, theta1, offset1, max_val2, x2_deg, y2_deg, sigma_x2, sigma_y2, theta2, offset2, bimod_distance, MSE
# Whichever method of fit produces the better AIC is the better version. Produce a measurement of AIC for bimodal distribution for every unit.
def AIC_Bimod_Measurement(data):
m1 = mixture.GaussianMixture(1).fit(data)
m2 = mixture.GaussianMixture(2).fit(data)
uni_AIC = m1.aic(data)
bimod_AIC = m2.aic(data)
if uni_AIC < bimod_AIC:
modality = 'unimodal'
elif uni_AIC > bimod_AIC:
modality = 'bimodal'
else:
modality = 'neither'
return bimod_AIC, modality
# I don't do anything with this modality, but I could eventually.
def MapBimodalityIndex(AIC, amp):
scatter = plt.scatter(AIC, amp, alpha = 0.8)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(AIC, amp)
plt.plot(AIC, intercept + slope * AIC, '-')
print('bimodality r squared, intercept, slope' + str(r_value) + " " + str(intercept) + " " + str(slope))
plt.xlabel('Bimodality (AIC)')
plt.ylabel('Mean Response Amplitude')
ptitle = 'Bimodality Index'
plt.title(ptitle)
cwd = os.getcwd()
fname = 'Bimodality_Index.pdf'
plt.ylim(bottom = 5)
plt.savefig(str(cwd) + '/vis/' + fname)
plt.close()
def GaussPeakScatterplot(x_peak, y_peak):
scatter = plt.scatter(x_peak, y_peak)
plt.xlabel('x COM (deg)')
plt.ylabel('y COM (deg)')
ptitle = 'Positions of Gaussian Fit Peaks'
plt.title(ptitle)
cwd = os.getcwd()
fname = 'Gaussian_Peaks_Scat.pdf'
plt.savefig(str(cwd) + '/vis/' + fname)
plt.close()
def producecolormaps(i, infile, outfile, statsfile, increment, top_gauss_outf, bimod_gauss_outf, uni_gauss_statsf, bimod_gauss_statsf):
"""say what function does here for help(function)"""
x_deg_list = list()
y_deg_list = list()
mean_resp_list = list()
sd_resp_list = list()
min_resp_list = list()
max_resp_list = list()
#
cwd = os.getcwd()
with open(str(cwd) + '/statsfiles/' + str(infile), 'r') as inputfile:
for line in inputfile:
# Format of Line: cx -3.0 cy -3.0 0.000000 0.000000 0.000000 0.000000
tlist = line.split() # Temporary list to hold split words
x_deg_list.append(tlist[1])
y_deg_list.append(tlist[3])
mean_resp_list.append(tlist[4])
sd_resp_list.append(tlist[5])
min_resp_list.append(tlist[6])
max_resp_list.append(tlist[7])
#
degrees = (abs(float(x_deg_list[0])))
dimensions = 1
while x_deg_list[dimensions] == x_deg_list[dimensions-1]:
dimensions += 1
#
xmu = np.zeros([dimensions, dimensions])
xsd = np.zeros([dimensions, dimensions])
xmax = np.zeros([dimensions, dimensions])
xmin = np.zeros([dimensions, dimensions])
for j in range(dimensions):
for i in range(dimensions):
xmu[(dimensions-1)-i,j] = mean_resp_list[i+j*dimensions] # Mean
xsd[(dimensions-1)-i,j] = sd_resp_list[i+j*dimensions] # SD
xmax[(dimensions-1)-i,j] = max_resp_list[i+j*dimensions] # Max
xmin[(dimensions-1)-i,j] = min_resp_list[i+j*dimensions] # Min
#
# Prepare for plotting
#
x_axis = np.arange(-degrees, degrees + increment, increment)
fname = outfile
# Preparing to write to a statistics file
# Some information about the bimodality of the map. Change percent cutoff here if wanted
top_mu, bottom_mu = SplitMatrix(xmu, 10, dimensions)
# Find the value of the "top" = above cutoff percentage of SplitMatrix,
top_xCOM_raw, xCOM_deg = newcalcCOM(top_mu, x_axis, degrees, 0)
top_xCOM_raw = round(top_xCOM_raw, 4)
top_xCOM_deg = round(xCOM_deg, 4)
#
top_yCOM_raw, yCOM_deg = newcalcCOM(top_mu, x_axis, degrees, 1)
top_yCOM_deg = round(yCOM_deg, 4)
top_yCOM_raw = round(top_yCOM_raw, 4)
#
bottom_xCOM_raw, xCOM_deg = newcalcCOM(bottom_mu, x_axis, degrees, 0)
bottom_xCOM_raw = round(bottom_xCOM_raw, 4)
bottom_xCOM_deg = round(xCOM_deg, 4)
#
bottom_yCOM_raw, yCOM_deg = newcalcCOM(bottom_mu, x_axis, degrees, 1)
bottom_yCOM_raw = round(bottom_yCOM_raw, 4)
bottom_yCOM_deg = round(yCOM_deg, 4)
#
#xdeg, ydeg = DrawMap(i, xmu, 'Mean Response', dimensions, x_axis, degrees, fname, top_xCOM_raw, top_yCOM_raw, bottom_xCOM_raw, bottom_yCOM_raw, top_mu)
#change
x_dot, xdeg = newcalcCOM(xmu, x_axis, degrees, 0)
y_dot, ydeg = newcalcCOM(xmu, x_axis, degrees, 1)
xdeg = round(xdeg, 4)
ydeg = round(ydeg, 4)
max_mean = round(np.amax(xmu), 4)
#
#Max value of mean response for statsfile
#center_periph_dist = calcDistance(top_x_COM, top_y_COM, bottom_x_COM, bottom_y_COM)
#print("center peripheral COM distance = " + str(center_periph_dist))

max_val, x_deg, y_deg, sigma_x, sigma_y, theta, offset, uni_MSE = Map2DGaussian(xmu, dimensions, degrees, increment, max_mean, top_gauss_outf)
max_val1, x1_deg, y1_deg, sigma_x1, sigma_y1, theta1, offset1, max_val2, x2_deg, y2_deg, sigma_x2, sigma_y2, theta2, offset2, bimod_dist, bimod_MSE = MapBimodalGauss(xmu, dimensions, degrees, increment, max_mean, bimod_gauss_outf)

# did rounding for some and got lazy. Also figured it would be less exact expression of gaussian function.
max_val = round(max_val, 4)
x_deg = round(x_deg, 4)
y_deg = round(y_deg, 4)
sigma_x = round(sigma_x, 4)
sigma_y = round(sigma_y, 4)
theta = round(theta, 4)
offset = round(offset, 4)


with open(statsfile, 'a') as stats_file:
stats_file.write(infile + " " + str(xdeg) + " " + str(ydeg) + " " + str(top_xCOM_deg) + " " + str(top_yCOM_deg) + " " + \
str(bottom_xCOM_deg) + " " + str(bottom_yCOM_deg) + " " + str(max_mean) + " " + str(max_val) + " " + \
str(max_val1) + " " + str(x1_deg) + " " + str(y1_deg) + " " + str(max_val2) + " " + str(x2_deg) + " " + \
str(y2_deg) + " " + str(bimod_dist) + " " + "\n")
with open(uni_gauss_statsf, 'a') as gauss_statsfile:
gauss_statsfile.write(infile + " " + str(max_val) + " " + str(x_deg) + " " + str(y_deg) + " " + str(sigma_x) + " " + str(sigma_y) + \
" " + str(theta) + " " + str(offset) + " " + str(uni_MSE) + "\n")
with open(bimod_gauss_statsf, 'a') as bimod_gauss_statsfile:
bimod_gauss_statsfile.write(infile + " " + str(max_val1) + " " + str(x1_deg) + " " + str(y1_deg) + " " + str(sigma_x1) + " " + \
str(sigma_y1) + " " + str(theta1) + " " + str(offset1) + " " + str(max_val2) + " " + str(x2_deg) + " " + \
str(y2_deg) + " " + str(sigma_x2) + " " + str(sigma_y2) + " " + str(theta2) + " " + str(bimod_MSE) + " " + "\n")

#
#
#Specific to Files with Titles like: "cx33_norm2_6_6_0" or "xy44_norm2_6_6_0"
if "norm2" in infile:
substring=infile[15:]
unit_num=substring
plt.suptitle('Unit Norm2 - %s' %(unit_num))
else:
plt.suptitle(infile)
plt.close()

bimod_AIC, modality = AIC_Bimod_Measurement(xmu)



return xdeg, ydeg, max_mean, top_xCOM_deg, top_yCOM_deg, bimod_AIC, x_deg, y_deg, sigma_x, sigma_y

#
# Further PDF formatting
#
statfile = 'COM_Stats.txt'
if os.path.exists(statfile):
os.remove(statfile)
uni_gauss_statsfile = 'Uni_Gauss_Stats.txt'
if os.path.exists(uni_gauss_statsfile):
os.remove(uni_gauss_statsfile)
bimod_gauss_statsfile = 'Bimod_Gauss_Stats.txt'
if os.path.exists(bimod_gauss_statsfile):
os.remove(bimod_gauss_statsfile)
# GET RID of following?
sourcefiles_loc = '/statsfiles'
destination_loc = '/vis'
#
# initialize
max_mean = np.zeros(256)
top_xCOM_deg = np.zeros(256)
top_yCOM_deg = np.zeros(256)
bimod_AIC = np.zeros(256)
gauss_peak_x = np.zeros(256)
gauss_peak_y = np.zeros(256)
x_spread = np.zeros(256)
y_spread = np.zeros(256)
xdeg = np.zeros(256)
ydeg = np.zeros(256)
for i in range(256):
print(i)
p = str(i)
inf = 'xy44_norm2_6_6_' + p
outf = 'Top_Unit' + p + '.pdf'
top_gauss_outf = 'Topographic_2DGaussian' + p + '.pdf'
bimod_gauss_outf = 'Bimod_Topographic_2DGaussian' + p + '.pdf'
try:
xdeg[i], ydeg[i], max_mean[i], top_xCOM_deg[i], top_yCOM_deg[i], bimod_AIC[i], gauss_peak_x[i], gauss_peak_y[i], x_spread[i], y_spread[i] = producecolormaps(i, inf, outf, statfile, 0.5, top_gauss_outf, bimod_gauss_outf, uni_gauss_statsfile, bimod_gauss_statsfile)
except RuntimeError:
pass
#
print('spreadx' + str(x_spread))
TotalMeanScatter(max_mean, xdeg, ydeg)
#
print(gauss_peak_x)
a = np.where(gauss_peak_x == np.amax(gauss_peak_x))
print('max x deg index ' + str(a))
#

MapBimodalityIndex(bimod_AIC, max_mean)
# Creating a scatterplot works correctly if all 256 units are run
CreateScatterplot(max_mean, xdeg, ydeg)

from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

def makeColors(vals):
colors = np.zeros((len(vals), 3))
norm = Normalize(vmin=vals.min(), vmax=vals.max())
colors = [cm.ScalarMappable(norm=norm, cmap='jet').to_rgba(val) for val in vals]
return colors

def GaussScatter(gauss_peak_x, gauss_peak_y):
# compute kernel density object estimate of distribution of scatter
# one gaussian fit found a very unrealistic value (too far in x)
#cut =
plt.scatter(gauss_peak_x, gauss_peak_y, s=70, alpha = 0.02)
plt.show()
cwd = os.getcwd()
fname = 'Gaussian_Scatterplot_Peaks.pdf'
plt.savefig(str(cwd) + '/vis/' + fname)
plt.close()

GaussScatter(gauss_peak_x, gauss_peak_y)

GaussPeakScatterplot(gauss_peak_x, gauss_peak_y)



#make plot for standard deviation of the one gaussian fit
def plotGaussSD(x_spread, y_spread):
scatter = plt.scatter(x_spread, y_spread, alpha=.8)
#fig, ax = plt.subplots()
#ax.ticklabel_format(useOffset=False, style='plain')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_spread, y_spread)
plt.plot(x_spread, intercept + slope * x_spread, '-')
print('r squared gauss peaks, intercept, slope' + str(r_value) + " " + str(intercept) + " " + str(slope))
plt.xlabel('SDx')
plt.ylabel('SDy')
ptitle = 'Sizes of Receptive Fields'
plt.title(ptitle)
cwd = os.getcwd()
fname = 'Gaussian_Peaks_Position.pdf'
plt.savefig(str(cwd) + '/vis/' + fname)
plt.close()

plotGaussSD(x_spread, y_spread)

