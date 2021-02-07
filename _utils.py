# Python imports
import numpy as np

MIN = 0
MAX = 4294967295

def rand_int(low=MIN, high=MAX):
    rand_int = np.random.randint(MIN, MAX, 'uint32')
    return low + rand_int % (high - low)
