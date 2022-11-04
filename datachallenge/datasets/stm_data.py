from __future__ import print_function, absolute_import
import os
import os.path as osp
import requests
import zipfile
import gdown
from google_drive_downloader import GoogleDriveDownloader as gdd
print(__name__)
# from ..utils.data import Dataset
# from ..utils.data.dataset import download_extract
from datachallenge.utils.osutils import mkdir_if_missing
from datachallenge.utils.serialization import write_json, read_json

from sklearn.model_selection import train_test_split
from collections import Counter

# class STM_DATA(Dataset):
class STM_DATA():
    def __init__(self, extract_to=None, val_split= 0.15, test_split= 0.2, download_to =None, google_id = None, download=True):
        if google_id == None:
            google_id = "1H5sMjtAT_AEmjoOaElGHDN8G_v6PFcfU"
        self.id = google_id
        self.val_split = val_split
        self.test_split = test_split
        self.extract_to = extract_to
        if download_to == None:
            download_to = extract_to
        self.download_to = download_to
        mkdir_if_missing(self.download_to)
        if download:
            self.download()

        self.scan()
        self.split()
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

    def _check_integrity(self): # name 'images' as it is in your zip: 'train'
        return osp.isdir(osp.join(self.extract_to, 'train')) and \
               osp.isfile(osp.join(self.extract_to, 'meta.json')) and \
               osp.isfile(osp.join(self.extract_to, 'splits.json'))

    def download(self):
        if osp.isfile('./stm_data.zip'): # custom check_integrity from custom Dataset class, used to check if 'images' folder, 'meta.json', 'splits.json' exist.
            print("File already downloaded...")
            return
        file = gdown.download(id=self.id, output='./stm_data.zip', quiet=False)
        # print(self.download_to)
        # gdd.download_file_from_google_drive(file_id=self.id,
                                    # dest_path=osp.join('./stm_data.zip'),
                                    # unzip=False)

        with zipfile.ZipFile('./stm_data.zip', 'r') as zip_ref:
            zip_ref.extractall(self.extract_to)
    def scan(self):
        path_to_images = osp.join(self.extract_to, 'train')
        self.images_dir = path_to_images
        list_dirs = sorted(os.listdir(path_to_images)) # this should contains the classes sorted:
        self.num_classes = len(list_dirs)
        class_paths = [[] for _ in range(len(list_dirs))]
        size = 0
        X,y = [],[]
        for i, class_path in enumerate(list_dirs):
            class_i = i
            img_paths = osp.join(path_to_images, class_path)
            size +=len(os.listdir(img_paths))
            for img_path in os.listdir(img_paths):
                img_full_path = osp.join(img_paths, img_path)
                class_paths[i].append(img_full_path)
            X.extend(class_paths[i])
            y.extend([i]*len(class_paths[i]))
        self.X = X
        self.y = y
        self.size = size
        meta = {'name': 'STM_Dataset', 'num_classes': self.num_classes,
                'dataset_size' : self.size, 'images': class_paths, 'X': self.X,'y':self.y}
        write_json(meta, osp.join(self.extract_to, 'meta.json'))

    def split(self):
        X,y = self.X, self.y
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=self.test_split, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=self.val_split, random_state=42)
        # # Save meta information into a json file
        splits = {'X_train': X_train, 'y_train': y_train,'X_val': X_val,'y_val':y_val,
                'X_test': X_test,'y_test': y_test}
        write_json(splits, osp.join(self.extract_to, 'splits.json'))

        self.X_trainval= X_train_val
        self.y_trainval = y_train_val
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.weights_trainval = list(Counter(y_train_val).values())
        self.weights_train = list(Counter(y_train).values())
        # print(len(X_train),len(y_train),len(X_val), len(y_val),len(X_test), len(y_test))

if __name__ == "__main__":
    print("stm_data")
    # market_set = STM_DATA(root = "/content/test_folder")