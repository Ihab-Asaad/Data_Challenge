from __future__ import absolute_import
import warnings

from .stm_data import STM_DATA

__factory = {
    'stm_data': STM_DATA,
    # 'ex1_data': EX1_DATA, # add external data to train supervised/unsupervised models
    # 'ex2_data': EX2_DATA,
    # 'ex3_data': EX3_DATA,
    # 'ex4_data': EX4_DATA,
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
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: True
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](extract_to, *args, **kwargs)


def get_dataset(name, extract_to, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, extract_to, *args, **kwargs)