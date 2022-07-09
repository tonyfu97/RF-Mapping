"""
Code to fit a 2D pixel array to a 2D Gaussian.

Tony Fu, June 17, 2022
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

sys.path.append('..')
from files import check_extension
from gaussian_fit import gaussian_fit
import constants as c


weighted_map_path = os.path.join(result_dir, f"{layer_name}.weighted.cumulative_map.npy")
threshold_map_path = os.path.join(result_dir, f"{layer_name}.threshold.cumulative_map.npy")
center_only_map_path = os.path.join(result_dir, f"{layer_name}.center_only.cumulative_map.npy")
np.save(weighted_map_path, weighted_bar_sum)
np.save(threshold_map_path, threshold_bar_sum)
np.save(center_only_map_path, center_only_bar_sum)

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

    backprop_sum_dir = c.REPO_DIR + f'/results/ground_truth/backprop_sum/{model_name}/{sum_mode}'
    pdf_dir = c.REPO_DIR + f'/results/ground_truth/gaussian_fit/{model_name}/test'

    layer_name = "conv5"
    num_units = 5
    best_file_names = [f"max_{layer_name}.{unit_i}.npy" for unit_i in range(num_units)]
    worst_file_names = [f"min_{layer_name}.{unit_i}.npy" for unit_i in range(num_units)]
    both_file_names = [f"both_{layer_name}.{unit_i}.npy" for unit_i in range(num_units)]
    pdf_name = f"{layer_name}.gaussian.pdf"
    plot_title = f"{model_name} {layer_name} (sum mode = {sum_mode})"

    make_pdf(backprop_sum_dir, best_file_names, worst_file_names, both_file_names,
             pdf_dir, pdf_name, plot_title)
