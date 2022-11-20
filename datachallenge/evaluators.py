from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch

from .evaluation_metrics import accuracy, prec_rec, f1
from .utils.meters import AverageMeter
from datachallenge.utils import to_torch

def get_logits_batch(model, inputs , device = torch.device('cpu'), modules=None):
    model.eval()
    inputs = inputs.to(device)
    # inputs = to_torch(inputs).to(device)
    if modules is None:
        with torch.no_grad(): ## ??
            outputs = model(inputs).cpu()
            return outputs

def get_logits_all(model, data_loader, print_freq=1, device = torch.device('cpu')):
    model.eval()
    batch_time = AverageMeter()
    end = time.time()
    targets , logits = [], []
    for i, (imgs, classes) in enumerate(data_loader):
        batch_time.update(time.time() - end) # the time of getting new batch
        outputs = get_logits_batch(model, imgs, device)
        targets.append(classes)
        logits.append(outputs)
        if (i + 1) % print_freq == 0:
            print('Get outputs: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader), batch_time.val, batch_time.avg))
        end = time.time()
        targets_ = torch.cat([x for x in targets], dim=0)
        logits_ = torch.cat([x for x in logits], dim=0)
    return logits_, targets_

def get_logits_all_test(model, data_loader, print_freq=1, device = torch.device('cpu')):
    model.eval()
    batch_time = AverageMeter()
    end = time.time()
    logits, imgs_names = [], []
    for i, (img, img_name) in enumerate(data_loader): # batch size here is one
        batch_time.update(time.time() - end) # the time of getting new batch
        outputs = get_logits_batch(model, img, device)
        logits.append(outputs)
        if (i + 1) % print_freq == 0:
            print('Get outputs: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader), batch_time.val, batch_time.avg))
        end = time.time()
        logits_ = np.argmax(torch.cat([x for x in logits], dim=0))
        imgs_names.append(img_name[0])
        print(img_name)
    return imgs_names, logits_

def evaluate_all(logits, targets):
    acc_ = accuracy(logits, targets)
    prec_, rec_ = prec_rec(logits, targets)
    f1_ = f1(logits, targets)
    return acc_, prec_, rec_, f1_
    # Compute mean AP
    # mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    # print('Mean AP: {:4.1%}'.format(mAP))

    # # Compute all kinds of CMC scores
    # cmc_configs = {
    #     'allshots': dict(separate_camera_set=False,
    #                      single_gallery_shot=False, # if true, each gallery identity has only one instance.
    #                      first_match_break=False),
    #     'cuhk03': dict(separate_camera_set=True,
    #                    single_gallery_shot=True,
    #                    first_match_break=False),
    #     'market1501': dict(separate_camera_set=False,
    #                        single_gallery_shot=False,
    #                        first_match_break=True)}
    # cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
    #                         query_cams, gallery_cams, **params)
    #               for name, params in cmc_configs.items()}

    # print('CMC Scores{:>12}{:>12}{:>12}'
    #       .format('allshots', 'cuhk03', 'market1501'))
    # for k in cmc_topk:
    #     print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
    #           .format(k, cmc_scores['allshots'][k - 1],
    #                   cmc_scores['cuhk03'][k - 1],
    #                   cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    # return cmc_scores['allshots'][0]


class Evaluator(object):
    def __init__(self, model, device):
        super(Evaluator, self).__init__()
        self.model = model
        self.device = device

    def evaluate(self, data_loader):
        logits, targets = get_logits_all(self.model, data_loader, device = self.device)
        acc_ , prec_, rec_, f1_ = evaluate_all(logits, targets)
        print("Accuracy: ", acc_, "  Precision: ", prec_, "  Recall: ", rec_, " F1: ", f1_)
        return acc_ , prec_, rec_, f1_

    def predict(self, data_loader):
        imgs_names, logits = get_logits_all_test(self.model, data_loader, device = self.device)
        print(imgs_names)
        df = pd.DataFrame({'id': imgs_names, 'label': logits.tolist()})
        df.to_csv('submission.csv', index=False)