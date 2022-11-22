from __future__ import absolute_import
import os.path as osp

from PIL import Image
import numpy as np


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        if isinstance(self.dataset, tuple):
            return len(self.dataset[0])
        else:
            return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        if isinstance(self.dataset, tuple):
            X_data = self.dataset[0]
            y_data = self.dataset[1]
            image_path , class_i = X_data[index], y_data[index]
            # fpath = fname
            # if self.root is not None:
            #     fpath = osp.join(self.root, fname)
            img = np.asarray(Image.open(image_path).convert('RGB'))
            if self.transform is not None:
                img = self.transform(img)
            return img, class_i
        else: # for test data
            X_data = self.dataset
            image_path = X_data[index]
            img = np.asarray(Image.open(image_path).convert('RGB'))
            if self.transform is not None:
                img = self.transform(img)
            img_name = image_path.split('/')[-1].split('.')[0] # return list of one element
            return img, img_name