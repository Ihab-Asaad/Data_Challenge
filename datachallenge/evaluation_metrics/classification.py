from __future__ import absolute_import

from ..utils import to_torch
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.functional import precision_recall
# from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy


def accuracy(output, target, topk=(1,)):
    Acc = MulticlassAccuracy(num_classes = 8)
    return Acc(output.detach().cpu(), target.detach().cpu())
    # output, target = to_torch(output), to_torch(target)
    # maxk = max(topk)
    # batch_size = target.size(0)

    # _, pred = output.topk(maxk, 1, True, True)
    # pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))

    # ret = []
    # for k in topk:
    #     correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
    #     ret.append(correct_k.mul_(1. / batch_size))
    # return ret

def prec_rec(output, target):
    # prec_, rec_ = precision_recall(preds, target, average='macro', num_classes=8)
    prec_, rec_ = precision_recall(preds, target, average='macro', num_classes=8) # https://torchmetrics.readthedocs.io/en/stable/classification/precision_recall.html
    return prec_, rec_

def f1(output, target):
    metric = MulticlassF1Score(num_classes=8) # https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html
    f1_ = metric(output, target)
    return f1_