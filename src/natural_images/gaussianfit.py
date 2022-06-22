"""
Code to fit a 2D pixel array to a 2D Gaussian. 

Tony Fu, June 17, 2022

This code is the modified version of the code found on the following thread:
https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
"""

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def twoD_Gaussian(xycoord, amplitude, mu_x, mu_y, sigma_1, sigma_2, 
                  theta, offset):
    """
    The model function of a 2D Gaussian. Intended to be the first input 
    argument for scipy.optimize.curve_fit(f, xdata, ydata).

    Parameters
    ----------
    According to the official documentation of scipy.optimize.curve_fit(), the 
    first argument of f (this function) should be the indepedent variable(s), 
    and the rest of the arguments should be function parameters.

    xycoord: a two-tuple of 2D numpy arrays = (x, y)
        The x and y coordinates, in unit pixels. x and y should be 2D array, 
        i.e., the result of meshgrid(x, y).
    amplitude: int or float
        The amplitude of the Gaussian, i.e., the max of the Gaussian if the 
        offset is zero. Unitless.
    mu_x and mu_y: int or float
        The center coordiates of the Gaussian, in unit pixels.
    sigma_1, sigma_2: int or float
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
        an 1D array as per official documentation of scipy.optimize.curve_fit().
    """
    x, y = xycoord
    mu_x = float(mu_x)
    mu_y = float(mu_y)
    theta = theta * np.pi / 180

    a = (np.cos(theta)**2)/(2*sigma_1**2) + (np.sin(theta)**2)/(2*sigma_2**2)
    b = -(np.sin(2*theta))/(4*sigma_1**2) + (np.sin(2*theta))/(4*sigma_2**2)
    c = (np.sin(theta)**2)/(2*sigma_1**2) + (np.cos(theta)**2)/(2*sigma_2**2)
    z = offset + amplitude*np.exp( - (a*((x-mu_x)**2) + 2*b*(x-mu_x)*(y-mu_y) 
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

def ellipticalfit(image, initial_guess=None, plot=True, show=False):
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
        mu_x and mu_y: int or float
            The center coordiates of the Gaussian, in unit pixels.
        sigma_1, sigma_2: int or float
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

    # Initialize intial guess if necessary.
    if initial_guess is None:
        sigma_1_guess = 40
        sigma_2_guess = 30
        theta_guess = 90
        offset_guess = 0
        initial_guess = (image.max(), x_size//2, y_size//2, 
                         sigma_1_guess, sigma_2_guess, theta_guess, offset_guess)
    
    # Fit the 2D Gaussian.
    param_estimate, params_covar = opt.curve_fit(twoD_Gaussian, (x, y), 
                                                 image.flatten(), 
                                                 p0=initial_guess, 
                                                 method='trf')
    param_sem = np.sqrt(np.diag(params_covar))

    # (Optional): plot the original image with the fitted curves.
    if plot:
        image_fitted = twoD_Gaussian((x, y), *param_estimate)
        # fig, ax = plt.subplots(1, 1)
        plt.imshow(image, cmap=plt.cm.jet)
        # plt.colorbar()
        plt.contour(x, y, image_fitted.reshape(y_size, x_size), 7, colors='w')
    
    if show:
        plt.show()

    return param_estimate, param_sem


def test_ellipticalfit():
    """Test fit a 2D Gaussian."""
    # Create x and y indices.
    x_size, y_size = 200, 250
    x = np.arange(x_size)
    y = np.arange(y_size)
    x, y = np.meshgrid(x, y)

    # Create data.
    data = twoD_Gaussian((x, y), 3, 80, 180, 40, 70, 45, 10).reshape(y_size, x_size)
    data_noisy = data + 0.2 *np.random.normal(size=data.shape)
        
    # Test fit.
    param_estimate, param_sem = ellipticalfit(data_noisy, show=True)
    print(param_estimate)
    print(param_sem)
    assert np.all(param_sem < 1), 'SEM too high.'


if __name__ == '__main__':
    test_ellipticalfit()

"""
Create a multi-page pdf file, each page contains the Gaussian fit of the 
(average guided-backpropagation of the) best and the worst image patches for 
each unit. 

However, this code block only works for me. For other users, please change 
the file paths, and make sure you have the .ppm files that contains the 
average back-propagation of the image patches.
"""

if __name__ == '__main__':
    # Mount google drive in google colab
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)
    import sys
    sys.path.insert(0, '/content/gdrive/My Drive/Colab Notebooks/Border Ownership Research/conv2_ppm')

    # Other imports for plotting
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    from tqdm import tqdm

    with PdfPages("/content/gdrive/My Drive/Colab Notebooks/Border Ownership Research/test.pdf") as pdf:
        num_units = 256
        for i in tqdm(range(num_units)):
            # Load the average back-propagation of the image patches.
            best_file_name = f"b_conv2.{i}.mu.ppm"
            best_avg_backprop = Image.open(dir + best_file_name)
            best_avg_backprop_np = np.asarray(best_avg_backprop)

            worst_file_name = f"w_conv2.{i}.mu.ppm"
            worst_avg_backprop = Image.open(dir + worst_file_name)
            worst_avg_backprop_np = np.asarray(worst_avg_backprop)
            
            # Fit 2D Gaussian, and plot them.
            plt.figure(figsize=(20,10))
            plt.suptitle(f"2D Gaussian of Average Backpropagation: Conv2 No.{i}", fontsize=20)
            plt.subplot(1,2,1)
            params, sems = ellipticalfit(best_avg_backprop_np[:,:,0])
            subtitle1 = f"A={params[0]:.2f}(err={sems[0]:.2f}), mu_x={params[1]:.2f}(err={sems[1]:.2f}), mu_y={params[2]:.2f}(err={sems[2]:.2f}),"
            subtitle2 = f"sigma_1={params[3]:.2f}(err={sems[3]:.2f}), sigma_2={params[4]:.2f}(err={sems[4]:.2f}),"
            subtitle3 = f"theta={params[5]:.2f}(err={sems[5]:.2f}), offset={params[6]:.2f}(err={sems[6]:.2f})"
            # plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.title(f"Best Image Patches\n({subtitle1}\n{subtitle2}\n{subtitle3})", fontsize=14)

            plt.subplot(1,2,2)
            params, param_sem = ellipticalfit(worst_avg_backprop_np[:,:,0])
            plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            subtitle1 = f"A={params[0]:.2f}(err={sems[0]:.2f}), mu_x={params[1]:.2f}(err={sems[1]:.2f}), mu_y={params[2]:.2f}(err={sems[2]:.2f}),"
            subtitle2 = f"sigma_1={params[3]:.2f}(err={sems[3]:.2f}), sigma_2={params[4]:.2f}(err={sems[4]:.2f}),"
            subtitle3 = f"theta={params[5]:.2f}(err={sems[5]:.2f}), offset={params[6]:.2f}(err={sems[6]:.2f})"
            # plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.title(f"Worst Image Patches\n({subtitle1}\n{subtitle2}\n{subtitle3})", fontsize=14)

            # Save the figure to pdf and close.
            pdf.savefig()
            plt.close()