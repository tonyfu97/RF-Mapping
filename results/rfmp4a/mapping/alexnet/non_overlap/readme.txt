# Temporary README file for txt files.

COORDINATES: zero-centered, y-axis points downward.

file_name: alexnet_rfmp4a_tb1.txt
source: bar.py > summarize_TB1()
fields:
    - layer_name (str)
    - unit_i (int)
    - top_i (int)
    - top_x (float)
    - top_y (float)
    - bot_i (int)
    - bot_x (float)
    - bot_y (float)


file_names: alexnet_rfmp4a_tbn.txt
source: bar.py > summarize_TBn()
fields:
    - layer_name (str)
    - unit_i (int)
    - top_avg_x (float)
    - top_avg_y (float)
    - bot_avg_x (float)
    - bot_avg_y (float)