import numpy as np
from scipy.misc import imread, imresize, imsave
import torch
import cv2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, returnShape=False):
    #print("LOADING IMAGE")
    img = cv2.imread(filepath)
    shape = img.shape

    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    # numpy.ndarray to FloatTensor
    # img = torch.from_numpy(img)
    img = imresize(img, (64,64))
    img = preprocess_img(img)
    if returnShape: 
        return img,shape
    else:
        return img


def save_img(img, filename):
    img = deprocess_img(img)
    img = img.numpy()
    img *= 255.0
    img = img.clip(0, 255)
    img = np.transpose(img, (1, 2, 0))
    img = imresize(img, (250, 200, 3))
    img = img.astype(np.uint8)
    imsave(filename, img)
    print("Image saved as {}".format(filename))


def preprocess_img(img):
    # [0,255] image to [0,1]

    # img = imresize(img, (224, 224))

    min = img.min()
    max = img.max()
    diff = max - min
    if diff == 0:
       diff = 0.01
    img = ( img - min ) / diff

    # # check that input is in expected range
    assert img.max() <= 1, 'badly scaled inputs'
    assert img.min() >= 0, "badly scaled inputs"

    mean = (0.5, 0.5, 0.5)
    std  = (0.5, 0.5, 0.5)

    img = (img - mean) / std
    img = img * 2 - 1
    img = np.transpose(img, (2, 0, 1))

    img = torch.from_numpy(img)
    img = torch.FloatTensor(img.size()).copy_(img)


    # RGB to BGR
    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    # [0,1] to [-1,1]
    # img = img.mul_(2).add_(-1)

    # # check that input is in expected range
    # assert img.max() <= 1, 'badly scaled inputs'
    # assert img.min() >= -1, "badly scaled inputs"

    return img


def deprocess_img(img):
    # BGR to RGB
    idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)

    # [-1,1] to [0,1]
    img = img.add_(1).div_(2)

    return img
