import numpy as np
import cv2
from scipy.spatial import distance
from scipy.ndimage.filters import convolve
from LIME.Utils.gradient import get_grad_toeplitz

def gen_spacial_affinity_kernel(spatial_sigma: float =3, size: int = 15):

    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-0.5 * (distance.euclidean((i, j), (size // 2, size // 2)) ** 2) / (spatial_sigma ** 2))

    return kernel

def gen_weight(map:np.ndarray, x:int, kernel:np.ndarray, strat:int = 3, epsilon:float = 1e-5):
    if strat == 0:
        print('strat 1')
        return np.ones_like(map)
    if strat == 1:
        print('strat 2')
        One = np.ones_like(map)
        grad_T = get_grad_toeplitz(map, int(x == 1))
        return np.divide(One, np.abs(grad_T) + epsilon)
    if strat == 3:
        # print('strat 3')
        grad_T = get_grad_toeplitz(map, int(x == 1))
        numerator = convolve(np.ones_like(map), kernel, mode='constant')
        denominator = np.abs(convolve(grad_T, kernel, mode='constant')) + epsilon
        return numerator / denominator
