import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from Utils.iocontrol import load, save, load_batch, save_batch

from Utils.initial_estimation import get_init_map
from Utils.solvers import exact_solver, sped_up_solver
from Utils.gradient import gen_D_conv

from Utils.noise_reduction.bilateral_filter import bilateral_filter
from Utils.recompose import recomposer


# from Utils.solvers import exact_solver
def get_img_list(dir: str):
    img_lst = []
    for roots, dirs, files in os.walk(dir):
        for file in files:
            print(file)
            if file.split('.')[-1].lower() in ['jpg', 'png', 'bmp']:
                img_lst.append(os.path.join(roots, file))
    print('Loaded ' + str(len(img_lst)) + ' image(s)')
    return img_lst


def LIME(img: np.ndarray, quick: bool = False, denoise: bool = False, recompose: bool = False, default: bool = True,
         *args, **kwargs):
    init_map = get_init_map(img)
    # get map estimation
    if quick:
        if default:
            solver = sped_up_solver(init_map)
        else:
            solver = sped_up_solver(init_map, kwargs['gamma'], kwargs['alpha'])
        solver.get_refined_map()
        refined_map = np.abs(solver.map_refined)
        refined_map_gamma = np.abs(solver.map_refined_gamma)
    else:
        if default:
            solver = exact_solver(init_map)
            solver.iterate(100)
        else:
            solver = exact_solver(init_map, kwargs['iterations'], kwargs['mu0'], kwargs['rho'], kwargs['alpha'],
                                  kwargs['gamma'])
            solver.iterate(kwargs['iterations'])
        refined_map = np.abs(solver.T)
        solver.gamma_correction()
        refined_map_gamma = np.abs(solver.map_refined_gamma)
    reflect = np.zeros_like(img)
    for ch in range(img.shape[2]):
        reflect[:, :, ch] = img[:, :, ch] / refined_map

    R = reflect

    if denoise:
        if kwargs['filter'] == 'bilateral':
            reflect_denoised = bilateral_filter(reflect)
            if recompose:
                reflect_recomposed = recomposer(reflect, reflect_denoised, refined_map)
                R = reflect_recomposed
            else:
                R = reflect_denoised

    enhanced_image = np.zeros_like(img)
    for ch in range(img.shape[2]):
        enhanced_image[:, :, ch] = R[:, :, ch] * refined_map_gamma
    return init_map, refined_map, refined_map_gamma, R, enhanced_image


def LIME_batch(img_lst, *args, **kwargs):
    image_list = []
    for path in img_lst:
        image_list.append(path)
    images = load_batch(image_list)
    enhanced_images = []
    for image in images:
        enhanced_images.append(LIME(image, quick=False, denoise=False, recompose=False, default=True)[4])
    return enhanced_images


if __name__ == '__main__':
    # img_dir = 'test_images'
    # img_lst = get_img_list(img_dir)
    # print(img_lst)
    # enhanced_images = LIME_batch(img_lst)
    # for i, images in enumerate(enhanced_images):
    #     print(images.shape)
    #     cv2.imshow(str(i), images)
    #     cv2.waitKey()
    impath = 'test_images/1.bmp'
    image = load(impath)
    init_map, refined_map, refined_map_gamma, R, enhanced_image = LIME(image)
    cv2.imshow('init_image', image)
    cv2.imshow('init_map', init_map)
    cv2.imshow('refined', refined_map)
    cv2.imshow('gamma', refined_map_gamma)
    cv2.imshow('reflect', R)
    cv2.imshow('result', enhanced_image)
    cv2.waitKey()