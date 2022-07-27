# Border Ownership

The imagenet data (50,000 .npy files of 3 x 227 x 227) are not uploaded to GitHub. They are available at: http://wartburg.biostr.washington.edu/loc/course/artiphys/data/i50k.html

Before running any code, go to src/rf_mapping/constants.py and change the directories.

To run the script in a terminal environment:
(1) Double-check the paths in src/rf_mapping/constants.py
(2) Because all .npy files (except results/ground_truth/top_n/alexnet and /vgg16) are ignored by github, please make sure the source directory of the script contains the necessary files. Then, check if the result directory contains the folder(s) to put the result files.
(3) Change the current directory to the repository
(4) To run, for example, 'backprop_sums_script.py', use the command:
    python3 -m src.rf_mapping.ground_truth.backprop_sum_script
The '-m' option let python know that the script is part of a module. Note that there is no need to specify the '.py' extension.
(5) To detach the running script from the terminal (so the script can keep running even when you close the terminal). First press CTRL + Z to suspend the running process. Remember the job number.
(6) Then enter 'bg' in the terminal to run the last stopped process in the background.
(7) Finally enter 'disown -h %1' to detach the first running job from the terminal.
(8) In a separate terminal window, do " ps -e | grep 'python' " to make sure that the process (check the job number) is running in the background.


OR 

(1) Comment out script guard.
(2) Use something like:
    nohup python3 -m src.rf_mapping.ground_truth.backprop_sum_script &
(3) Check progress and error messages using
    ps -e | grep 'python'
or
    cat git_repos/borderownership/nohup.out | more
(4) Exit terminal.