import numpy
import numpy as np


def get_init_map(init_image: np.ndarray):
    if init_image.shape[2] == 3:
        init_map = np.max(init_image,axis=2)
    else:
        jpg = init_image[:,:,:3]
        init_map = np.max(jpg,axis=2)
    return init_map

if __name__ == '__main__':
    print('main')