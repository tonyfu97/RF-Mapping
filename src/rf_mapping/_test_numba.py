"""
Tests if numba just-in-time (jit) compiler speeds up a function or not.

Tony Fu, July 1, 2022
"""
from numba import njit
import numpy as np
from time import time


#######################################.#######################################
#                                                                             #
#                                  TEST_NJIT                                  #
#                                                                             #
###############################################################################
def test_njit(func, num_times=10, parallel=False):
    """
    A decorator that prints out the time it takes to run the function without
    numba njit, during njit compilation, and after njit is compiled.

    Use it like this: test_njit(function_name)(function_arguments)
    """
    def inner(*args, **kwargs):
        time_before = 0
        for _ in range(num_times):
            start = time()
            func(*args, **kwargs)
            end = time()
            time_before += end - start
        time_before /= num_times
        print(f"        Before jit = {time_before:.8f} sec")

        func_jit = njit(func, parallel=parallel)  # Apply njit
        start = time()
        func_jit(*args, **kwargs)
        end = time()
        time_comp = end - start
        print(f"During compilation = {time_comp:.8f} sec")

        time_after = 0
        for _ in range(num_times):
            start = time()
            func_jit(*args, **kwargs)
            end = time()
            time_after += end - start
        time_after /= num_times
        print(f"         After jit = {time_after:.8f} sec")
        
        speed_factor = (time_before)/(time_after)
        print(f"-----------------------------------")
        if speed_factor >= 1:
            print(f"         Sped up by {speed_factor:.2f}x")
        else:
            print(f"         Slowed down by {1/speed_factor:.2f}x")

    return inner


# Test run
if __name__ == "__main__":
    def go_fast(a):
        """An example funciton."""
        trace = 0.0
        for i in range(a.shape[0]):
            trace += np.tanh(a[i, i])
        return a + trace

    x = np.arange(10000).reshape(100, 100)
    test_njit(go_fast)(x)  # Sped up by about 30x
