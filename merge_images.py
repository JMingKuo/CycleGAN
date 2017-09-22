import scipy.misc
import os
import numpy as np
import cv2
import glob

#reference: https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/utils.py
def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c)) + 255
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def color_correction(img):
    return cv2.cvtColor(cv2.bitwise_not(img), cv2.COLOR_RGB2BGR)

def read_img_list(img_list, type):
    images = []
    for file_name in img_list:
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if type in ['AB', 'BA']:
            image = color_correction(image)
        images.append(image)
    return np.array(images, dtype='uint8')

img_dir = 'out_imgs'
out_dir = 'merged_results'

for niter in range(250, 50000+1, 250):
    '''one image per snapshot'''
    for type in ['AB', 'ABA', 'BA', 'BAB']:
        match_str = '*_' + str(niter) + '_' + type + '.png'
        print(match_str)
        img_list = glob.glob(os.path.join(img_dir, match_str))
        images = read_img_list(img_list, type)
        save_images(images, [3, 7], os.path.join(out_dir, type, match_str[1:]) + '.png')
