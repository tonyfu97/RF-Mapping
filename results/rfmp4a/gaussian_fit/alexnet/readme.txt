# Temporary README file for txt files.

COORDINATES: zero-centered, y-axis points downward.

file_names: n01_non_overlap_com.txt
source: src/rf_mapping/rfmp4a/4a_com_non_overlap_script.py
fields:
    - layer_name (str)
    - unit_i (int)
    - top_x (float) x-coordinate of the center of mass of the top bars
    - top_y (float) y...
    - top_rad_10 (float) radius containing the 10% of area from COM
    - top_rad_50 (float) ...50%
    - top_rad_90 (float)
    - top_num_bars (int) the number of bars included in this top rf_mapping
    - bot_x (float)
    - bot_y (float)
    - bot_rad_10 (float)
    - bot_rad_50 (float)
    - bot_rad_90 (float)
    - bot_num_bars (int)

file_names: n01_weighted_top.txt and n01_weighted_bot.txt
source: src/rf_mapping/rfmp4a/4a_gaussian_weighted_script.py
fields:
    - layer_name (str)
    - unit_i (int)
    - mux (float)
    - muy (float)
    - sd1 (float)
    - sd2 (float)
    - ori (float)
    - amp (float)
    - offset (float)
    - fxvar (float)
    - num_bars (int)
