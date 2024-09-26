import numpy as np
from typing import Union

# type hints
t_img = np.array
from utils import bilinear_grid_sample

def plot_disparity(src, disparities) -> t_img:
    """plots the translated dst image given a src image and disparity map

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        disparities: a numpy array of shape [H x W] containing disparity map
    Returns:
        An image with the same size src.
    """
 
    xv = np.arange(src.shape[1])
    yv = np.arange(src.shape[0])
    X, Y = np.meshgrid(xv, yv)
    print(X.shape)
    print(disparities.shape)
    return bilinear_grid_sample(src, X + disparities, Y)
