"""
Code to fit a 2D pixel array to a 2D Gaussian.

Tony Fu, June 17, 2022

This code is the modified version of the code found on the following thread:
https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
"""
import os
import sys

from statistics import variance
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('../..')
from src.rf_mapping.files import check_extension
import src.rf_mapping.constants as c


#######################################.#######################################
#                                                                             #
#                                TWOD_GAUSSIAN                                #
#                                                                             #
###############################################################################
def twoD_Gaussian(xycoord, amplitude, mu_x, mu_y, sigma_1, sigma_2,
                  theta, offset):
    """
    The model function of a 2D Gaussian. Intended to be the first input
    argument for scipy.optimize.curve_fit(f, xdata, ydata).

    Parameters
    ----------
    According to the official documentation of scipy.optimize.curve_fit(), the
    first argument of f (this function) should be the indepedent variable(s),
    and the rest of the arguments should be function parameters. The order
    of the parameters must be also specified in the GaussianFitParamFormat
    class (excluding xycoord).

    xycoord: a two-tuple of 2D numpy arrays = (x, y)
        The x and y coordinates, in unit pixels. x and y should be 2D array,
        i.e., the result of meshgrid(x, y).
    amplitude: int or float
        The amplitude of the Gaussian, i.e., the max of the Gaussian if the
        offset is zero. Unitless.
    mu_x and mu_y: int or float
        The center coordiates of the Gaussian, in unit pixels.
    sigma_1 and sigma_2: int or float
        The std. dev. of the two orthogonal axis in unit pixels. sigma_1 should
        be the horizontal axis if theta = 0 degree.
    theta: int or float
        The angle of the sigma_1 away from the positive x-axis, measured
        counterclockwise in unit degrees.
    offset: int or float
        The offset of the Gaussian. Unitless.

    Returns
    -------
    z: 2D numpy array
        The height of the 2D Guassian given coordinates (x, y). This has to be
        an 1D array as per official documentation of
        scipy.optimize.curve_fit().
    """
    x, y = xycoord
    mu_x = float(mu_x)
    mu_y = float(mu_y)
    theta = theta * np.pi / 180

    a = (np.cos(theta)**2)/(2*sigma_1**2) + (np.sin(theta)**2)/(2*sigma_2**2)
    b = -(np.sin(2*theta))/(4*sigma_1**2) + (np.sin(2*theta))/(4*sigma_2**2)
    c = (np.sin(theta)**2)/(2*sigma_1**2) + (np.cos(theta)**2)/(2*sigma_2**2)
    z = offset + amplitude*np.exp(-(a*((x-mu_x)**2) + 2*b*(x-mu_x)*(y-mu_y)
                                  + c*((y-mu_y)**2)))
    return z.flatten()


def test_twoD_Gaussian():
    """Plot a 2D Gaussian. No assert statements. Must use visual inspection."""
    # Create x and y indices.
    x_size, y_size = 140, 200
    x = np.arange(x_size)
    y = np.arange(y_size)
    x, y = np.meshgrid(x, y)

    # Create data.
    data = twoD_Gaussian((x, y), 3, 70, 100, 40, 10, 10, 10)

    # Plot twoD_Gaussian data generated above.
    plt.figure()
    plt.imshow(data.reshape(y_size, x_size))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    test_twoD_Gaussian()


#######################################.#######################################
#                                                                             #
#                                GAUSSIAN_FIT                                 #
#                                                                             #
###############################################################################
def gaussian_fit(image, initial_guess=None, plot=True, show=False, cmap=plt.cm.jet):
    """
    Fit a 2D gaussian to an input image.

    Parameters
    ----------
    image: 2D numpy array
        The image to be fit. The pixel [0, 0] is the top-left corner.
    intial_guess: tuple
        Initial guesses for the parameters (see Returns).
    plot: bool
        Whether to plot the result isocline or not.
    plot: bool
        Whether to show the result plot.

    Returns
    -------
    param_estimate: numpy.array consists of the following values
        amplitude: int or float
            The amplitude of the Gaussian, i.e., the max of the Gaussian if the
            offset is zero. Unitless.
        mu_x & mu_y: int or float
            The center coordiates of the Gaussian, in unit pixels.
        sigma_1 & sigma_2: int or float
            The std. dev. of the two orthogonal axis in unit pixels. sigma_1
            should be the horizontal axis if theta = 0 degree.
        theta: int or float
            The angle of the sigma_1 away from the positive x-axis, measured
            counterclockwise in unit degrees.
        offset: int or float
            The offset of the Gaussian. Unitless.
    param_sem: numpy.array
        The standard errors of the parameter estimates listed above.
    """
    # Create x and y indices.
    y_size, x_size = image.shape
    x = np.arange(x_size)
    y = np.arange(y_size)
    x, y = np.meshgrid(x, y)

    # Initialize intial guess.
    if initial_guess is None:
        amp_guess = image.max() - image.min()
        muy_guess, mux_guess = np.unravel_index(np.argmax(image), image.shape)
        sigma_1_guess = image.shape[0]/4
        sigma_2_guess = image.shape[0]/5
        theta_guess = 90
        offset_guess = image.mean()
        
        initial_guess = (amp_guess, mux_guess, muy_guess,
                         sigma_1_guess, sigma_2_guess,
                         theta_guess, offset_guess)

    # Fit the 2D Gaussian.
    try:
        param_estimate, params_covar = opt.curve_fit(twoD_Gaussian, (x, y),
                                                    image.flatten(),
                                                    p0=initial_guess,
                                                    method='lm',
                                                    check_finite=True)
        param_sem = np.sqrt(np.diag(params_covar))
    except:
        param_estimate = np.full(len(initial_guess), -1)
        param_sem = np.full(len(initial_guess), 999)
        print("bad fit")

    # Plot the original image with the fitted curves.
    if plot:
        image_fitted = twoD_Gaussian((x, y), *param_estimate)
        # fig, ax = plt.subplots(1, 1)
        plt.imshow(image, cmap=cmap)
        # plt.colorbar()
        plt.contour(x, y, image_fitted.reshape(y_size, x_size), 9, colors='w')

    if show:
        plt.show()

    return param_estimate, param_sem


def test_gaussian_fit():
    """Test fit a 2D Gaussian."""
    # Create x and y indices.
    x_size, y_size = 200, 250
    x = np.arange(x_size)
    y = np.arange(y_size)
    x, y = np.meshgrid(x, y)

    # Create data.
    data = twoD_Gaussian((x, y), 3, 80, 180, 40, 70, 45, 10).reshape(y_size,
                                                                     x_size)
    data_noisy = data + 0.2 * np.random.normal(size=data.shape)

    # Test fit.
    param_estimate, param_sem = gaussian_fit(data_noisy, plot=True)
    print(param_estimate)
    print(param_sem)
    assert np.all(param_sem < 1), 'SEM too high.'


if __name__ == '__main__':
    test_gaussian_fit()


#######################################.#######################################
#                                                                             #
#                                   MAKE_PDF                                  #
#                                 (DEPRICATED)                                #
#                                                                             #
###############################################################################
def make_pdf(data_dir, best_file_names, worst_file_names, both_file_names, 
             pdf_dir, pdf_name, plot_title):
    """
    Creates a multi-page pdf file, each page contains the Gaussian fit of the
    (average guided-backpropagation of the) best and the worst image patches
    for each unit.
    
    The parameters and the corresponding errors of each gaussian fit will be
    saved to same file names given as {best/worst/both_file_names} in pdf_dir.

    Parameters
    ----------
    data_dir : str
        The directory to the data files.
    best/worst/both_file_names : list of str
        The file names of the images in data_path.
    pdf_dir : str
        The directory to save the result pdf, does not have to exist in the
        first place.
    pdf_name : str
        The name of the pdf file.
    plot_title : str
        The title string on each pdf page.
    save_params : bool
        If true, the parameters and the corresponding errors of each gaussian
        fit will be saved to same file names given as {best_file_names} and
        {worst_file_names} to pdf_dir.
    """
    pdf_name = check_extension(pdf_name, 'pdf')
    pdf_path = os.path.join(pdf_dir, pdf_name)

    with PdfPages(pdf_path) as pdf:
        page_count = 0
        for best_file_name, worst_file_name, both_file_name in\
                tqdm(zip(best_file_names, worst_file_names, both_file_names)):
            # Load the back-propagation sum of the image patches.
            best_file_name = check_extension(best_file_name, '.npy')
            best_file_path = os.path.join(data_dir, best_file_name)
            best_backprop_sum_np = np.load(best_file_path)

            worst_file_name = check_extension(worst_file_name, '.npy')
            worst_file_path = os.path.join(data_dir, worst_file_name)
            worst_backprop_sum_np = np.load(worst_file_path)

            both_file_name = check_extension(both_file_name, '.npy')
            both_file_path = os.path.join(data_dir, both_file_name)
            both_backprop_sum_np = np.load(both_file_path)

            # Fit 2D Gaussian, and plot them.
            plt.figure(figsize=(30, 10))
            plt.suptitle(f"{plot_title}: unit no.{page_count}", fontsize=20)
            page_count += 1

            plt.subplot(1, 3, 1)
            params, sems = gaussian_fit(best_backprop_sum_np, plot=True, show=False)

            param_file_name = os.path.join(pdf_dir, best_file_name)
            param_arr = np.vstack((params, sems))
            np.save(param_file_name, param_arr)
            # Saved as [[param0, param1, ....],[sem0, sem1, ...]]
                
            subtitle1 = f"A={params[0]:.2f}(err={sems[0]:.2f}), "\
                        f"mu_x={params[1]:.2f}(err={sems[1]:.2f}), "\
                        f"mu_y={params[2]:.2f}(err={sems[2]:.2f}),"
            subtitle2 = f"sigma_1={params[3]:.2f}(err={sems[3]:.2f}), "\
                        f"sigma_2={params[4]:.2f}(err={sems[4]:.2f}),"
            subtitle3 = f"theta={params[5]:.2f}(err={sems[5]:.2f}), "\
                        f"offset={params[6]:.2f}(err={sems[6]:.2f})"
            plt.title(f"Max Image Patches\n({subtitle1}\n{subtitle2}"\
                      f"\n{subtitle3})", fontsize=14)

            plt.subplot(1, 3, 2)
            params, sems = gaussian_fit(worst_backprop_sum_np, plot=True, show=False)
            
            param_file_name = os.path.join(pdf_dir, worst_file_name)
            param_arr = np.vstack((params, sems))
            np.save(param_file_name, param_arr)
            # Saved as [[param0, param1, ....],[sem0, sem1, ...]]

            subtitle1 = f"A={params[0]:.2f}(err={sems[0]:.2f}), "\
                        f"mu_x={params[1]:.2f}(err={sems[1]:.2f}), "\
                        f"mu_y={params[2]:.2f}(err={sems[2]:.2f}),"
            subtitle2 = f"sigma_1={params[3]:.2f}(err={sems[3]:.2f}), "\
                        f"sigma_2={params[4]:.2f}(err={sems[4]:.2f}),"
            subtitle3 = f"theta={params[5]:.2f}(err={sems[5]:.2f}), "\
                        f"offset={params[6]:.2f}(err={sems[6]:.2f})"
            plt.title(f"Min Image Patches\n({subtitle1}\n{subtitle2}"\
                    f"\n{subtitle3})", fontsize=14)
                

            plt.subplot(1, 3, 3)
            params, sems = gaussian_fit(both_backprop_sum_np, plot=True, show=False)

            param_file_name = os.path.join(pdf_dir, both_file_name)
            param_arr = np.vstack((params, sems))
            np.save(param_file_name, param_arr)
            # Saved as [[param0, param1, ....],[sem0, sem1, ...]]

            subtitle1 = f"A={params[0]:.2f}(err={sems[0]:.2f}), "\
                        f"mu_x={params[1]:.2f}(err={sems[1]:.2f}), "\
                        f"mu_y={params[2]:.2f}(err={sems[2]:.2f}),"
            subtitle2 = f"sigma_1={params[3]:.2f}(err={sems[3]:.2f}), "\
                        f"sigma_2={params[4]:.2f}(err={sems[4]:.2f}),"
            subtitle3 = f"theta={params[5]:.2f}(err={sems[5]:.2f}), "\
                        f"offset={params[6]:.2f}(err={sems[6]:.2f})"
            plt.title(f"Max + Min Image Patches\n({subtitle1}\n{subtitle2}"\
                      f"\n{subtitle3})", fontsize=14)

            pdf.savefig()
            plt.close()


if __name__ == '__main__':
    model_name = 'alexnet'
    sum_mode = 'abs'

    backprop_sum_dir = c.RESULTS_DIR + f'/ground_truth/backprop_sum/{model_name}/{sum_mode}'
    pdf_dir = c.RESULTS_DIR + f'/ground_truth/gaussian_fit/{model_name}/test'

    layer_name = "conv5"
    num_units = 5
    best_file_names = [f"max_{layer_name}.{unit_i}.npy" for unit_i in range(num_units)]
    worst_file_names = [f"min_{layer_name}.{unit_i}.npy" for unit_i in range(num_units)]
    both_file_names = [f"both_{layer_name}.{unit_i}.npy" for unit_i in range(num_units)]
    pdf_name = f"{layer_name}.gaussian.pdf"
    plot_title = f"{model_name} {layer_name} (sum mode = {sum_mode})"

    make_pdf(backprop_sum_dir, best_file_names, worst_file_names, both_file_names,
             pdf_dir, pdf_name, plot_title)


#######################################.#######################################
#                                                                             #
#                          GAUSSIAN FIT PARAM FORMAT                          #
#                                                                             #
###############################################################################
class GaussianFitParamFormat:
    """
    Currently, the results of elliptical Gaussian fit is formatted as an array
    of 7 parameters ordered as follows:
        [A, mu_x, mu_y, sigma_1, sigma_2, theta, offset]
    This indices must match the order of the arguments of the twoD_Gaussian()
    function.
    """
    NUM_PARAMS  = 7
    
    A_IDX       = 0
    MU_X_IDX    = 1
    MU_Y_IDX    = 2
    SIGMA_1_IDX = 3
    SIGMA_2_IDX = 4
    THETA_IDX   = 5
    OFFSET_IDX  = 6


#######################################.#######################################
#                                                                             #
#                             CALC_F_EXPLAINED_VAR                            #
#                                                                             #
###############################################################################
def calc_f_explained_var(sum_map, params):
    """
    Calculates the fraction of variance explained by the fit with the formula:
        exp_var = 1 - var(sum_map - fit_map)/var(sum_map)

    Parameters
    ----------
    sum_map : numpy.ndarray
        The map constructed by summing bars.
    params : [floats, ...]
        Paramters of 2D elliptical Gaussian fit. Should comply with
        GaussianFitParamFormat().

    Returns
    -------
    exp_var : float
        The fraction of explained variance.
    """
    # Reconstruct map with fit parameters.
    x_size = sum_map.shape[1]
    y_size = sum_map.shape[0]
    x = np.arange(x_size)
    y = np.arange(y_size)
    x, y = np.meshgrid(x, y)
    fit_map = twoD_Gaussian((x, y),
                            params[GaussianFitParamFormat.A_IDX],
                            params[GaussianFitParamFormat.MU_X_IDX],
                            params[GaussianFitParamFormat.MU_Y_IDX],
                            params[GaussianFitParamFormat.SIGMA_1_IDX],
                            params[GaussianFitParamFormat.SIGMA_2_IDX],
                            params[GaussianFitParamFormat.THETA_IDX],
                            params[GaussianFitParamFormat.OFFSET_IDX])
    # Calcualte variances
    residual_var = variance(fit_map - sum_map.flatten())
    gt_var = variance(sum_map.flatten())
    return 1 - (residual_var/gt_var)


#######################################.#######################################
#                                                                             #
#                               WRAP_ANGLE_180                                #
#                                                                             #
###############################################################################
def wrap_angle_180(angle):
    """Constraints angle to be 0 <= angle < 180."""
    while angle >= 180:
        angle -= 180
    while angle < 0:
        angle += 180
    return angle


#######################################.#######################################
#                                                                             #
#                                THETA_TO_ORI                                 #
#                                                                             #
###############################################################################
def theta_to_ori(sigma_1, sigma_2, theta):
    """
    Translates theta into orientation. Needs this function because theta
    tells us the orientation of sigma_1, which may or may not be the semi-
    major axis, whereas orientation is always about the semi-major axis.
    Therefore, when sigma_2 > sigma_1, our theta is off by 90 degrees from
    the actual orientation.

    Parameters
    ----------
    sigma_1 & sigma_2 : float
        The std. dev.'s of the semi-major and -minor axes. The larger of
        of the two is the semi-major.
    theta : float
        The orientation of sigma_1 in degrees.

    Returns
    -------
    orientation: float
        The orientation of the unit's receptive field in degrees.
    """
    if sigma_1 > sigma_2:
        return wrap_angle_180(theta)
    return wrap_angle_180(theta - 90)


#######################################.#######################################
#                                                                             #
#                                PARAM CLEANER                                #
#                                                                             #
###############################################################################
class ParamCleaner(GaussianFitParamFormat):
    """
    A class that cleans a single parameter vector that contains the results
    of an elliptical Gaussian fit. The parameters should be ordered according
    to the GaussianFitParamFormat class.
    """
    def __init__(self, sem_thres=1):
        """
        Constructs a ParamCleaner object.

        Parameters
        ----------
        sem_thres : int, optional
            An SEM value is considered too big if it is above this threshold.
        """
        super().__init__()
        self.sem_thres = sem_thres
    
    def _err_is_too_big(self, sems):
        """
        Checks if any parameter has a SEM value (error) greater than the
        threshold. Returns True if the error is too big.

        The SEM value of theta is ignored because most units have circular
        receptive fields such that the error for theta is often big.
        """
        for i, sem in enumerate(sems):
            if i != self.THETA_IDX and sem > self.sem_thres:
                return True
        return False
    
    def _mu_is_outside_rf(self, mu_x, mu_y, box):
        y_min, x_min, y_max, x_max = box
        return not(x_min <= mu_x <= x_max and y_min <= mu_y <= y_max)

    def _wrap_angle_180(self, angle):
        while angle >= 180:
            angle -= 180
        while angle < 0:
            angle += 180
        return angle

    def _theta_to_ori(self, sigma_1, sigma_2, theta, theta_sem):
        """
        Translates theta into orientation. Needs this function because theta
        tells us the orientation of sigma_1, which may or may not be the semi-
        major axis, whereas orientation is always about the semi-major axis.
        Therefore, when sigma_2 > sigma_1, our theta is off by 90 degrees from
        the actual orientation.

        Parameters
        ----------
        sigma_1 & sigma_2 : float
            The std. dev.'s of the semi-major and -minor axes. The larger of
            of the two is the semi-major.
        theta : float
            The orientation of sigma_1 in degrees.
        theta_sem : float
            The error value of the fit of theta.

        Returns
        -------
        orientation: float
            The orientation of the unit's receptive field in degrees.
        """
        if theta_sem > self.sem_thres:
            return np.NAN
        if sigma_1 > sigma_2:
            return self._wrap_angle_180(theta)
        return self._wrap_angle_180(theta - 90)
        
    def clean(self, params, sems, box):
        """
        Cleans the parameters of a single unit.

        Parameters
        ----------
        params : numpy array [7, ]
            The parameters of the elliptical Gaussian. Ordered according to
            GaussianFitParamFormat.
        sems : numpy array [7, ]
            The standard errors of the means (SEMs) of the parameters.
        box : (int, int, int, int)
            The box of the RF in (vx_min, hx_min, vx_max, hx_max) format.

        Returns
        -------
        cleaned_params : array-like
            The original params with following modifications:
            (1) Take the absolute value of sigma_1 and 2.
            (2) Theta is translated into orientation (the direction of either
                sigma_1 or sigma_2 depending on which is longer.)
            (3) Transform the origin from top-left to image center.
        Returns None if :
            (1) Have at least one parameter with a SEM greater than the
                threshold.
            (2) Have a center (mu_y, mu_x) that is outside of receptive field.
        """
        if self._err_is_too_big(sems):
            return None
        if self._mu_is_outside_rf(params[self.MU_X_IDX],
                                  params[self.MU_Y_IDX],
                                  box):
            return None

        cleaned_params = params.copy()
        cleaned_params[self.SIGMA_1_IDX] = np.absolute(cleaned_params[self.SIGMA_1_IDX])
        cleaned_params[self.SIGMA_2_IDX] = np.absolute(cleaned_params[self.SIGMA_2_IDX])
        cleaned_params[self.THETA_IDX] = self._theta_to_ori(params[self.SIGMA_1_IDX], 
                                                            params[self.SIGMA_2_IDX], 
                                                            params[self.THETA_IDX], 
                                                            sems[self.THETA_IDX])
        return cleaned_params


#######################################.#######################################
#                                                                             #
#                                 PARAM LOADER                                #
#                                                                             #
###############################################################################
class ParamLoader(ParamCleaner):
    """
    Loads the parameters of the inidividual units and sorts them into lists of
    individual parameters. Needs this class because the gaussia fit data was
    stored on a unit-basis, but we need to aggregate them to analyze
    population results. The parameters are loaded once the ParamLoader is
    constructed.
    """
    def __init__(self, data_dir, file_names, rf_size, sem_thres=1):
        super().__init__(sem_thres)
        self.data_dir = data_dir
        self.file_names = file_names
        self.rf_size = rf_size

        self.As = []
        self.mu_xs = []
        self.mu_ys = []
        self.sigma_1s = []
        self.sigma_2s = []
        self.orientations = []
        self.offsets = []
        self._load_params()

    def _load_params(self):
        for file_name in self.file_names:
            param_file_path = os.path.join(self.data_dir, file_name)
            unit_params_sems = np.load(param_file_path)
            cleaned_params = self.clean(unit_params_sems[0, :],
                                        unit_params_sems[1, :],
                                        self.rf_size)
            if cleaned_params is not None:  # ignore poorly fit units.
                self.As.append(cleaned_params[self.A_IDX])
                self.mu_xs.append(cleaned_params[self.MU_X_IDX])
                self.mu_ys.append(cleaned_params[self.MU_Y_IDX])
                self.sigma_1s.append(cleaned_params[self.SIGMA_1_IDX])
                self.sigma_2s.append(cleaned_params[self.SIGMA_2_IDX])
                self.orientations.append(cleaned_params[self.THETA_IDX])
                self.offsets.append(cleaned_params[self.OFFSET_IDX])

        # Converts all of them into numpy arrays.
        self.As = np.array(self.As)
        self.mu_xs = np.array(self.mu_xs)
        self.mu_ys = np.array(self.mu_ys)
        self.sigma_1s = np.array(self.sigma_1s)
        self.sigma_2s = np.array(self.sigma_2s)
        self.offsets = np.array(self.As)

        # Removes NAN entries from orientations.
        self.orientations = np.array(self.orientations)
        self.orientations = self.orientations[np.isfinite(self.orientations)]
            

if __name__ == '__main__':
    model_name = 'alexnet'
    sum_mode = 'sqr'
    max_or_min = 'max'
    layer_name = 'conv2'
    rf_size = (51, 51)
    num_units = 192

    backprop_sum_dir = c.RESULTS_DIR + f'/ground_truth/gaussian_fit/{model_name}/{sum_mode}'
    file_names = [f"{max_or_min}_{layer_name}.{i}.npy" for i in range(num_units)]
    param_loader = ParamLoader(backprop_sum_dir, file_names, rf_size)
