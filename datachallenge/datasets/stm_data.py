from __future__ import print_function, absolute_import
from kaggle.api.kaggle_api_extended import KaggleApi
import kaggle
import os
import os.path as osp
import requests
import zipfile
import gdown
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd
from datachallenge.utils.osutils import mkdir_if_missing
from datachallenge.utils.serialization import write_json, read_json

from sklearn.model_selection import train_test_split
from collections import Counter

user_name = 'ihabasaad'
key = '7e284af09589e68770a3d479ef215d07'
if user_name == '':
    raise KeyError("enter you kaggle account first")

os.environ["KAGGLE_USERNAME"] = user_name
os.environ["KAGGLE_KEY"] = key


class STM_DATA():
    def __init__(self, extract_to=None, val_split=0.15, test_split=0.2, download_to=None, google_id=None, download=True):
        if google_id == None:
            google_id = "1H5sMjtAT_AEmjoOaElGHDN8G_v6PFcfU&confirm=t"
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

        self.summary()

    def _check_integrity(self):  # name 'images' as it is in your zip: 'train'
        return osp.isdir(osp.join(self.extract_to, 'train_new'))
            #  and \
            # osp.isfile(osp.join(self.extract_to, 'meta.json')) and \
            # osp.isfile(osp.join(self.extract_to, 'splits.json'))

    # def download(self):
    #     if osp.isfile('./stm_data.zip'): # custom check_integrity from custom Dataset class, used to check if 'images' folder, 'meta.json', 'splits.json' exist.
    #         print("File already downloaded...")
    #         return
    #     file = gdown.download(id=self.id, output='./stm_data.zip', quiet=False )
    #     # gdd.download_file_from_google_drive(file_id=self.id,
    #                                 # dest_path=osp.join('./stm_data.zip'),
    #                                 # unzip=False)

    #     with zipfile.ZipFile('./stm_data.zip', 'r') as zip_ref:
    #         zip_ref.extractall(self.extract_to)

    def download(self):
        # custom check_integrity from custom Dataset class, used to check if 'images' folder, 'meta.json', 'splits.json' exist.
        if osp.isfile('./msiam-sigma-dc-2223.zip'):
            print("File already downloaded...")
            return

        api = KaggleApi()
        api.authenticate()
        api.competition_download_files('msiam-sigma-dc-2223',
                                       path='./', quiet=False)
        with zipfile.ZipFile('./msiam-sigma-dc-2223.zip', 'r') as zip_ref:
            zip_ref.extractall(self.extract_to)

    # def scan(self):
    #     path_to_images = osp.join(self.extract_to, 'train')
    #     self.images_dir = path_to_images
    #     list_dirs = sorted(os.listdir(path_to_images)) # this should contains the classes sorted:
    #     # classes string is the required output according to the problem:
    #     self.classes_str = [1,20,21,22,32,41,44,45]
    #     self.num_classes = len(list_dirs)
    #     # class_path is a list of length (num_classes), each list is of length number of images in each class:
    #     class_paths = [[] for _ in range(len(list_dirs))]
    #     size = 0
    #     # X is a list containing the paths of all images in dataset, y contains their labels
    #     X,y = [],[]
    #     for i, class_path in enumerate(list_dirs):
    #         class_i = i
    #         img_paths = osp.join(path_to_images, class_path)
    #         size +=len(os.listdir(img_paths))
    #         for img_path in os.listdir(img_paths):
    #             img_full_path = osp.join(img_paths, img_path)
    #             class_paths[i].append(img_full_path)
    #         X.extend(class_paths[i])
    #         y.extend([i]*len(class_paths[i]))
    #     self.X = X
    #     self.y = y
    #     self.size = size
    #     meta = {'name': 'STM_Dataset', 'num_classes': self.num_classes,
    #             'dataset_size' : self.size, 'images': class_paths, 'X': self.X,'y':self.y}
    #     write_json(meta, osp.join(self.extract_to, 'meta.json'))
    def extract_csv(self, path):
        if not osp.exists(path):
            raise ValueError
        # print("file probs_train not exist")
        else:
            dict_img_class = dict()
            with open(path, 'r') as file:
                image_class = pd.read_csv(file, skiprows=0, delimiter='\n')
                for i in range(len(image_class)):
                    row = image_class.iloc[i, 0]
                    img_name, true_class = row.split(',')
                    img_name_jpg = img_name+'.jpg'
                    true_class = self.classes_str.index(int(true_class))
                    dict_img_class[img_name_jpg] = true_class
            return dict_img_class

    def scan(self):
        path_to_images = osp.join(self.extract_to, 'train_new')
        self.images_dir = path_to_images
        # this should contains the classes sorted:
        list_dirs = sorted(os.listdir(path_to_images))
        # classes string is the required output according to the problem:
        self.classes_str = [1, 20, 21, 22, 32, 41, 44, 45]
        self.num_classes = len(self.classes_str)
        # class_path is a list of length (num_classes), each list is of length number of images in each class:
        class_paths = [[] for _ in range(self.num_classes)]
        path_to_csv = osp.join(self.extract_to, 'new_train.csv')
        dict_imgname_class = self.extract_csv(path_to_csv)
        size = 0
        # X is a list containing the paths of all images in dataset, y contains their labels
        X, y = [], []
        size = len(os.listdir(path_to_images))
        # print(path_to_images)
        # for i, img_name in enumerate(path_to_images):
        for img_name in os.listdir(path_to_images):
            # class_i = i
            # img_paths = osp.join(path_to_images, class_path)

            img_full_path = osp.join(path_to_images, img_name)
            # print(img_name)
            class_i = dict_imgname_class[img_name]
            class_paths[class_i].append(img_full_path)
            X.append(img_full_path)
            y.append(class_i)
        self.X = X
        self.y = y
        self.size = size
        # meta = {'name': 'STM_Dataset', 'num_classes': self.num_classes,
        #         'dataset_size' : self.size, 'images': class_paths, 'X': self.X,'y':self.y}
        # write_json(meta, osp.join(self.extract_to, 'meta.json'))

    def split(self):

        # if cross_val:
        #     X,y = self.X, self.y
        #     X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=self.test_split, random_state=0)
        #     X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=self.val_split, random_state=42)
        #     kf = KFold(n_splits=3, random_state=42, shuffle=True)
        #     for i, (train_index, test_index) in enumerate(kf.split(X)):
        #         pass
        # else:
        X, y = self.X, self.y
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_split, random_state=0)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.val_split, random_state=42)
        # Save meta information into a json file
        # splits = {'X_train': X_train, 'y_train': y_train,'X_val': X_val,'y_val':y_val,
        #         'X_test': X_test,'y_test': y_test}
        # write_json(splits, osp.join(self.extract_to, 'splits.json'))

        self.X_trainval = X_train_val
        self.y_trainval = y_train_val
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.weights_trainval = list(Counter(y_train_val).values())
        self.weights_train = list(Counter(y_train).values())

    def summary(self, verbose=True):
        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  Subset   | # images")
            print("  ---------------------------")
            print("  dataset  | {:5d}".format(len(self.X)))
            print("  classes  | {:5d}".format(self.num_classes))
            print("  train    | {:5d}".format(len(self.X_train)))
            print("  val      | {:5d}".format(len(self.X_val)))
            print("  trainval | {:5d}".format(len(self.X_trainval)))
            print("  test     | {:5d}".format(len(self.X_test)))


if __name__ == "__main__":
    print("stm_data")
