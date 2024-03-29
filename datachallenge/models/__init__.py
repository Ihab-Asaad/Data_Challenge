from __future__ import absolute_import

from datachallenge.models.inception import *
from datachallenge.models.resnet import *
from datachallenge.models.resnext import *
from datachallenge.models.efficient import *
from datachallenge.models.cusnet import CusNet
from datachallenge.models.vit import *


__factory = {
    'inception': inception,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50': resnext50,
    'efficientnet': efficientnet,
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_b4': efficientnet_b4,
    'efficientnet_b5': efficientnet_b5,
    'efficientnet_b6': efficientnet_b6,
    'efficientnet_b7': efficientnet_b7,
    'cusnet':CusNet,
    'vit':VisionTransformer
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.
    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)

if __name__=="__main__":
    model=resnet50(num_features=120,
                          dropout=0.5, num_classes=100)
    print(model)

def get_configs(args, num_classes):
    model_configs = dict()
    model_configs["name"] = args["net"]["arch"]
    model_configs["num_features"] =args["training"]["features"]
    model_configs["dropout"] = args["training"]["dropout"]
    model_configs["num_classes"] = num_classes
    
    return model_configs