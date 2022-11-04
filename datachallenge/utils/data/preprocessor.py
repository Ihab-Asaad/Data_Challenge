from __future__ import absolute_import
import os.path as osp

from PIL import Image


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        X_data = self.dataset[0]
        y_data = self.dataset[1]
        image_path , class_i = X_data[index], y_data[index]
        # fpath = fname
        # if self.root is not None:
        #     fpath = osp.join(self.root, fname)
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, class_i