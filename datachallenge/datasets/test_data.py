from __future__ import print_function, absolute_import
import os
import os.path as osp
import requests
import zipfile
import gdown
from google_drive_downloader import GoogleDriveDownloader as gdd
from datachallenge.utils.osutils import mkdir_if_missing
from datachallenge.utils.serialization import write_json, read_json

from sklearn.model_selection import train_test_split
from collections import Counter

import os

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# downloading from kaggle.com/c/sentiment-analysis-on-movie-reviews
# there are two files, train.tsv.zip and test.tsv.zip
# we write to the current directory with './'
# api.competition_download_file('msiam-sigma-dc-2223',
#                               'test_new.zip',
#                               path='./')

class TEST_SUBMIT():
    def __init__(self, extract_to=None, download_to =None, google_id = None, download=True):
        # if google_id == None:
        #     google_id = "1H5sMjtAT_AEmjoOaElGHDN8G_v6PFcfU"
        self.id = google_id
        self.extract_to = extract_to
        if download_to == None:
            download_to = extract_to
        self.download_to = download_to
        mkdir_if_missing(self.download_to)
        if download:
            self.download()

        self.summary()

    def _check_integrity(self): # name 'images' as it is in your zip: 'train'
        return osp.isdir(osp.join(self.extract_to, 'test_new')) and \
               osp.isfile(osp.join(self.extract_to, 'meta_test.json'))

    def download(self):
        if osp.isfile('./msiam-sigma-dc-2223.zip'): # custom check_integrity from custom Dataset class, used to check if 'images' folder, 'meta.json', 'splits.json' exist.
            print("File already downloaded...")
            return
        os.environ["KAGGLE_USERNAME"] = "ihabasaad"
        os.environ["KAGGLE_KEY"] = "743ea9ddf4935aa2a7a41ac72b038849"
        api = KaggleApi()
        # api.authenticate()
        api.competition_download_files('msiam-sigma-dc-2223',
                              path='./')
        with zipfile.ZipFile('./msiam-sigma-dc-2223.zip', 'r') as zip_ref:
            zip_ref.extractall(self.extract_to)

    def scan(self):
        path_to_images = osp.join(self.extract_to, 'test_new')
        self.images_dir = path_to_images
        list_dirs = sorted(os.listdir(path_to_images)) # this should contains the classes sorted:
        self.size = len(list_dirs)
        # X is a list containing the paths of all images in dataset, y contains their labels
        X= []
        for i, img_path in enumerate(list_dirs):
            img_path_abs = osp.join(path_to_images, img_path)
            X.append(img_path_abs)
        self.X = X
        meta = {'name': 'STM_Dataset_Test', 'dataset_size' : self.size, 'X': self.X}
        write_json(meta, osp.join(self.extract_to, 'meta_test.json'))

    def summary(self, verbose = True):
        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  Test_data   | # images")
            print("  ---------------------------")
            print("  test     | {:5d}".format(len(self.X)))

if __name__ == "__main__":
    print("stm_data_test")