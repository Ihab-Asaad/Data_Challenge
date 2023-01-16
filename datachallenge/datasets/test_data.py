from __future__ import print_function, absolute_import
import os
import os.path as osp
import zipfile
from datachallenge.utils.osutils import mkdir_if_missing
from datachallenge.utils.serialization import write_json

user_name = 'ihabasaad'
key = '7e284af09589e68770a3d479ef215d07'
if user_name == '':
    raise KeyError("enter you kaggle account first")

user_name = 'ihabasaad'
key = ''
os.environ["KAGGLE_USERNAME"] = user_name
os.environ["KAGGLE_KEY"] = key
from kaggle.api.kaggle_api_extended import KaggleApi


class TEST_SUBMIT():
    def __init__(self, extract_to=None, download_to=None, google_id=None, download=True, user_name = '', key = ''):
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

        self.scan()

        self.summary()

    def _check_integrity(self):  # name 'images' as it is in your zip: 'train'
        return osp.isdir(osp.join(self.extract_to, 'test_new')) and \
            osp.isfile(osp.join(self.extract_to, 'meta_test.json'))

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

    def scan(self):
        path_to_images = osp.join(self.extract_to, 'test_new')
        self.images_dir = path_to_images
        # this should contains the classes sorted:
        list_dirs = sorted(os.listdir(path_to_images))
        self.size = len(list_dirs)
        # X is a list containing the paths of all images in dataset, y contains their labels
        X = []
        for i, img_path in enumerate(list_dirs):
            img_path_abs = osp.join(path_to_images, img_path)
            X.append(img_path_abs)
        self.X = X
        meta = {'name': 'STM_Dataset_Test',
                'dataset_size': self.size, 'X': self.X}
        write_json(meta, osp.join(self.extract_to, 'meta_test.json'))

    def summary(self, verbose=True):
        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  Test_data   | # images")
            print("  ---------------------------")
            print("  test     | {:5d}".format(len(self.X)))


if __name__ == "__main__":
    print("stm_data_test")
