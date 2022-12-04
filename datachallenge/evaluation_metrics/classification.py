from __future__ import absolute_import

from ..utils import to_torch
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import precision_recall
from torchmetrics import Precision, Recall
# from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
# from torchmetrics.classification import MulticlassRecall, MulticlassPrecision
from torchmetrics.classification import MulticlassConfusionMatrix
# from torchmetrics import ConfusionMatrix
import torch 


def accuracy(output, target):
    Acc = MulticlassAccuracy(num_classes = 8, average='micro')
    return Acc(output.detach().cpu(), target.detach().cpu())

def accuracy_micro(output, target):
    Acc = MulticlassAccuracy(num_classes = 8, average='micro')
    return Acc(output.detach().cpu(), target.detach().cpu())

def prec_rec(output, target):
    # precision = Precision(task="multiclass", average='macro', num_classes=8)
    precision = Precision(task="multiclass", average='micro', num_classes=8)
    prec_ = precision(output, target)
    # recall = Recall(task="multiclass", average='macro', num_classes=8)
    recall = Recall(task="multiclass", average='micro', num_classes=8)
    rec_ = recall(output,target)
    # prec_, rec_ = precision_recall(output, target, average='macro', num_classes=8) # https://torchmetrics.readthedocs.io/en/stable/classification/precision_recall.html
    return prec_, rec_

def f1(output, target):
    metric = MulticlassF1Score(num_classes=8, average = 'micro') # https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html
    f1_ = metric(output, target)
    return f1_

def conf_matrix(output, target):
    # confmat = ConfusionMatrix(task='multiclass',num_classes=8)
    confmat = MulticlassConfusionMatrix(task='multiclass',num_classes=8)
    return confmat(output, target)

def top2acc(output, target, top = 2):
    class_i = -1*torch.zeros([target.shape[0],top], dtype=torch.long)
    idx = torch.tensor(range(0, target.shape[0]), dtype = torch.long)
    for j in range(top):
        print(class_i.shape, target.shape[0])
        class_i[:,j] = torch.argmax(output, axis = 1)
        # print(class_i[:,j])
        output[idx,class_i[:,j]] = -100000000

    # class_1 = torch.argmax(output, axis = 1)
    # output[class_1]=-10000
    # class_2 = torch.argmax(output, axis = 1)
    cnt = 0
    for i in range(target.shape[0]):
        for j in range(top):
            if target[i]==class_i[i,j]:
                cnt = cnt +1 
                break
    return cnt/target.shape[0]
