"""
Testing if numpy speeds up a function or not.

Tony Fu, July 1, 2022
"""
from numba import njit
import numpy as np
from time import time


#######################################.#######################################
#                                                                             #
#                                  IS_FASTER                                  #
#                                                                             #
###############################################################################
def is_faster(func, num_times=10):
    """
    A decorator that prints out the time it takes to run the function without
    numba jit, during jit compilation, and after jit is compiled.

    Can also use it like this: is_faster(func)(*args, **kwargs)
    """
    def inner(*args, **kwargs):
        time_before = 0
        for i in range(num_times):
            start = time()
            func(*args, **kwargs)
            end = time()
            time_before = (end-start)/num_times
        print(f"        Before jit = {time_before:.8f} sec")
        
        func_jit = njit(func)
        start = time()
        func_jit(*args, **kwargs)
        end = time()
        time_comp = end - start
        print(f"During compilation = {time_comp:.8f} sec")
        
        time_after = 0
        for i in range(num_times):
            start = time()
            func_jit(*args, **kwargs)
            end = time()
            time_after = (end-start)/num_times
        print(f"        After jit  = {time_after:.8f} sec")
        
        speed_factor = (time_before)/(time_after)
        
        if speed_factor >= 1:
            print(f"Sped up by {speed_factor:.2f}x")
        else:
            print(f"Slowed down by {1/speed_factor:.2f}x")
        
    return inner


if __name__ == "__main__":
    x = np.arange(10000).reshape(100, 100)

    def go_fast(a):
        """An example funciton."""
        trace = 0.0
        for i in range(a.shape[0]):
            trace += np.tanh(a[i, i])
        return a + trace

    is_faster(go_fast)(x)