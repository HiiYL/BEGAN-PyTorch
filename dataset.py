# python train.py --cuda --dataset aesthetics-full


from os import listdir
from os.path import join
import os

import torch.utils.data as data

from util import is_image_file, load_img
from pandas import HDFStore
import numpy as np
import torch
import random


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = image_dir
        self.image_filenames = [ x for x in listdir(image_dir) if is_image_file(x)]

    def __getitem__(self, index):
        # Load Image
        image = load_img(join(self.image_dir, self.image_filenames[index]))

        return image

    def __len__(self):
        return len(self.image_filenames)

