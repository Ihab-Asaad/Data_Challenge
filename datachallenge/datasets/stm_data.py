from __future__ import print_function, absolute_import
import os.path as osp
import requests
import zipfile
import gdown
from google_drive_downloader import GoogleDriveDownloader as gdd
print(__name__)
# from ..utils.data import Dataset
# from ..utils.data.dataset import download_extract
from datachallenge.utils.osutils import mkdir_if_missing
# from ..utils.serialization import write_json


# class STM_DATA(Dataset):
class STM_DATA():
    def __init__(self, extract_to=None, val_split= 0.15, test_split= 0.2, download_to =None, google_id = None, download=True):
        if google_id == None:
            google_id = "1H5sMjtAT_AEmjoOaElGHDN8G_v6PFcfU"
        self.id = google_id
        self.extract_to = extract_to
        if download_to == None:
            download_to = extract_to
        self.download_to = download_to
        mkdir_if_missing(self.download_to)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.scan()
        self.split()

    def _check_integrity(self): # name 'images' as it is in your zip: 'train'
        return osp.isdir(osp.join(self.extract_to, 'train')) and \
               osp.isfile(osp.join(self.extract_to, 'meta.json')) and \
               osp.isfile(osp.join(self.extract_to, 'splits.json'))

    def download(self):
        # if osp.isfile('./stm_data.zip'): # custom check_integrity from custom Dataset class, used to check if 'images' folder, 'meta.json', 'splits.json' exist.
        #     print("File already downloaded...")
        #     return
        # file = gdown.download(id=self.id, output=self.download_to, quiet=False)
        print(self.id)
        file = gdown.download(id="1H5sMjtAT_AEmjoOaElGHDN8G_v6PFcfU", output='./stm_data.zip', quiet=False)
        # print(self.download_to)
        # gdd.download_file_from_google_drive(file_id=self.id,
                                    # dest_path=osp.join('./stm_data.zip'),
                                    # unzip=False)

        with zipfile.ZipFile('./stm_data.zip', 'r') as zip_ref:
            zip_ref.extractall(self.extract_to)
    def scan(self):
        path_to_images = osp.isdir(osp.join(self.extract_to, 'train'))
        list_dirs = os.list_dirs(path_to_images)
        print(list_dirs)
    def split(self):
        return
        # # Save meta information into a json file
        # meta = {'name': 'Market1501', 'shot': 'multiple', 'num_cameras': 6,
        #         'identities': identities}
        # write_json(meta, osp.join(self.root, 'meta.json'))

        # # Save the only training / test split
        # splits = [{
        #     'trainval': sorted(list(trainval_pids)),
        #     'query': sorted(list(query_pids)),
        #     'gallery': sorted(list(gallery_pids))}]
        # write_json(splits, osp.join(self.root, 'splits.json'))

if __name__ == "__main__":
    print("stm_data")
    # market_set = STM_DATA(root = "/content/test_folder")