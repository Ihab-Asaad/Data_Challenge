from __future__ import absolute_import
import warnings

from .stm_data import STM_DATA
from .test_data import TEST_SUBMIT

__factory = {
    'stm_data': STM_DATA,
    'test_data': TEST_SUBMIT,
}


def names():
    return sorted(__factory.keys())


def create(name, extract_to, *args, **kwargs):
    """
    Create a dataset instance.
    Parameters
    ----------
    name : str
        The dataset name. Can be the given dataset 'stm_data'
        or one of external datasets 'exi_data'
    extract_to : str
        The path to the dataset directory. join(data_dir from .yaml , name of dataset)
    val_split : float, optional
        Split the trainval set into train and val set with val_split ratio. Default: 0.15
    test_split : float, optional
        Split the data into trainval and test set with test_split ratio. Default: 0.2
    download_to: str, optional
        Download the data into another folder (just to split the compressed files from the extracted ones). Default: extract_to
    google_id: str, optional
        If the data in Google drive, use google_id to download it.
    download : bool, optional
        If True, will download the dataset. Default: True. 
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    print(args, kwargs)
    return __factory[name](extract_to, *args, **kwargs)
