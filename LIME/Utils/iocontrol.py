import numpy as np
from skimage import io,img_as_float


def load(impath:str):
    init_image = img_as_float(io.imread(impath))
    return init_image

def load_batch(impath: list):
    init_image = []
    for path in impath:
        init_image.append(img_as_float(io.imread(path)))
    return init_image

def save(savepath:str, img:np.ndarray):
    io.imsave(savepath, img)
    return 0

def save_batch(savepath: list, img: list):
    for path, image in zip(savepath, img):
        io.imsave(path, image)
    return 0