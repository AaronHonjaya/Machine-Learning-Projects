import os
import numpy as np
from scipy.io import loadmat

data = loadmat("Data/Bounding Boxes/bounding_boxes_lfpw_trainset.mat")

def flatten_matlab_struct(mat_struct):
    # Unwrap single-element lists and arrays
    while isinstance(mat_struct, (list, tuple, np.ndarray)) and len(mat_struct) == 1:
        mat_struct = mat_struct[0]
    return mat_struct


print(data.keys())

box_info = flatten_matlab_struct(data["bounding_boxes"][0][0])
print(box_info)

xmin, ymin, xmax, ymax = flatten_matlab_struct(box_info[2])



# files = os.listdir("./Data/lfpw/trainset")
# print(files[0])