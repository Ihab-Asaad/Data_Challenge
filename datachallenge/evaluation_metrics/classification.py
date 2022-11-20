from __future__ import absolute_import

from ..utils import to_torch
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.functional import precision_recall
# from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy


def accuracy(output, target):
    Acc = MulticlassAccuracy(num_classes = 8)
    return Acc(output.detach().cpu(), target.detach().cpu())

def prec_rec(output, target):
    prec_, rec_ = precision_recall(output, target, average='macro', num_classes=8) # https://torchmetrics.readthedocs.io/en/stable/classification/precision_recall.html
    return prec_, rec_

def f1(output, target):
    metric = MulticlassF1Score(num_classes=8, average = 'weighted') # https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html
    f1_ = metric(output, target)
    return f1_