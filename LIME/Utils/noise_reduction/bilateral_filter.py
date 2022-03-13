import cv2
import numpy as np
from numpy import clip


def bilateral_filter(image: np.ndarray, kernel_size: int = 5, color_sigma: float = 25, spacial_sigma: float = 5):
    m, n, c = image.shape
    img = image.astype(np.int32)

    def gen_spacial_affinity_kernel(spatial_sigma: float = spacial_sigma, size: int = kernel_size):

        kernel = np.zeros((size, size, 3))
        half = int((size - 1) / 2)
        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                kernel[i, j] = np.exp(-0.5 * (i ** 2 + j ** 2 / (spatial_sigma ** 2)))
        return kernel

    color_gaussian_dict = {i: np.exp(-0.5 * (i ** 2 / color_sigma ** 2)) for i in range(256)}

    def fetch_color_gaussian_val(color_diff):
        return color_gaussian_dict[clip(int(255 * color_diff), 0, 255)]

    color_gaussian_kernel = np.vectorize(fetch_color_gaussian_val, otypes=[np.float64])
    spacial_gaussian_kernel = gen_spacial_affinity_kernel(spacial_sigma, kernel_size)
    result = img.astype(np.float64)
    gap = int(kernel_size / 2)
    half = int((kernel_size - 1) / 2)

    for i in range(gap, m - gap):
        for j in range(gap, n - gap):
            window = image[i - half:i + half + 1, j - half:j + half + 1]
            color_mat = color_gaussian_kernel(np.abs(window - image[i][j]))
            spacial_mat = spacial_gaussian_kernel
            fr_gs = np.multiply(color_mat, spacial_mat)
            weight = fr_gs / np.sum(fr_gs, axis=(0, 1))
            result[i][j] = np.sum(np.multiply(weight, window), axis=(0, 1))
    return clip(result, 0, 1)


if __name__ == '__main__':
    print(bilateral_filter(np.ones((12, 12, 3))))
