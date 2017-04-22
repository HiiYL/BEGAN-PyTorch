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
from PIL import Image
import pickle



class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        super(DatasetFromFolder, self).__init__()
        
        if "flowers" in image_dir:
            self.image_dir = 'dataset/flowers/'
            with open("dataset/flowers/train/char-CNN-RNN-embeddings.pickle", "rb") as f:
                self.embeddings = pickle.load(f, encoding='latin1')

            with open("dataset/flowers/train/filenames.pickle", "rb") as f:
                filenames = pickle.load(f, encoding='latin1')
                #filenames = [ (filename[:10] + '0' + filename[10:]) for filename in filenames]
                self.image_filenames = [ "{}.jpg".format(filename) for filename in filenames ]
        else:
            self.image_dir = image_dir
            self.image_filenames = [ x for x in listdir(image_dir) if is_image_file(x)]
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        # Load Image
        image = Image.open(join(self.image_dir, self.image_filenames[index])).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        embedding = random.choice(self.embeddings[index])
        wrong_embedding = random.choice(random.choice(self.embeddings))

        return image, embedding, wrong_embedding

    def __len__(self):
        return len(self.image_filenames)

