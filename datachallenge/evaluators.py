from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch import nn

from .evaluation_metrics import accuracy, prec_rec, f1, top2acc, conf_matrix
from .utils.meters import AverageMeter
from datachallenge.utils import to_torch
from datachallenge.utils.serialization import load_checkpoint
import os.path as osp
from datachallenge import models
import gdown

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
    targets , logits, imgs_names= [], [], []
    for i, (imgs, classes, imgs_names_batch) in enumerate(data_loader):
        batch_time.update(time.time() - end) # the time of getting new batch
        outputs = get_logits_batch(model, imgs, device)
        targets.append(classes)
        logits.append(outputs)
        imgs_names.extend(imgs_names_batch)
        if (i + 1) % print_freq == 0:
            print('Get outputs: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader), batch_time.val, batch_time.avg))
        end = time.time()
    targets_ = torch.cat([x for x in targets], dim = 0)
    logits_ = torch.cat([x for x in logits], dim = 0)
    # imgs_names_ = torch.cat([x for x in imgs_names], dim = 0)
    return logits_, targets_, imgs_names

def get_logits_all_test(model, data_loader, print_freq=1, device = torch.device('cpu')):
    model.eval()
    batch_time = AverageMeter()
    end = time.time()
    logits, imgs_names = [], []
    for i, (img, img_name) in enumerate(data_loader): # batch size here is one
        batch_time.update(time.time() - end) # the time of getting new batch
        outputs = get_logits_batch(model, img, device)
        # logits.append(np.argmax(outputs))
        logits.append(outputs)
        imgs_names.append(img_name[0])
        if (i + 1) % print_freq == 0:
            print('Get outputs: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader), batch_time.val, batch_time.avg))
        end = time.time()
        # logits_ = torch.cat([x for x in logits], dim=0)
    logits_ = torch.cat([x for x in logits], dim=0)
    # logits_ = torch.FloatTensor(logits) 
    return imgs_names, logits_

def get_logits_all_test_ensemble(model1, model2, model3, data_loader, print_freq=1, device = torch.device('cpu')):
    model1.eval()
    model2.eval()
    model3.eval()
    batch_time = AverageMeter()
    end = time.time()
    logits, imgs_names = [], []
    for i, (img, img_name) in enumerate(data_loader): # batch size here is one
        batch_time.update(time.time() - end) # the time of getting new batch
        outputs1 = get_logits_batch(model1, img, device)
        outputs2 = get_logits_batch(model2, img, device)
        output3 = get_logits_batch(model3, img, device)
        logits.append(np.argmax((outputs1+outputs2+output3)/3.))
        imgs_names.append(img_name[0])
        if (i + 1) % print_freq == 0:
            print('Get outputs: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader), batch_time.val, batch_time.avg))
        end = time.time()
        # logits_ = torch.cat([x for x in logits], dim=0)
    print(logits)
    logits_ = torch.IntTensor(logits)        
    return imgs_names, logits_

def evaluate_all(logits, targets):
    acc_ = accuracy(logits, targets)
    prec_, rec_ = prec_rec(logits, targets)
    f1_ = f1(logits, targets)
    acc2 = top2acc(logits, targets)
    return acc_, prec_, rec_, f1_, acc2
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

    def evaluate(self, data_loader,ensemble = False, paths_ids = []):
        if ensemble:
            got_first = False
            for idx, path in enumerate(paths_ids):
                if osp.exists(path):
                    checkpoint = load_checkpoint(paths_ids[idx])
                else:
                    # download from google drive:
                    # 
                    self.download(paths_ids[idx], save_to = '/content/Data_Challenge/datachallenge/downloaded_model.tar')
                    checkpoint = load_checkpoint('/content/Data_Challenge/datachallenge/downloaded_model.tar')
                # model = models.create("resnet50", num_features=256,
                #           dropout=0.2, num_classes=8).to(self.device)
                model_configs = checkpoint['configs']
                print(model_configs)
                # model_configs["name"] = "efficientnet_b5"
                # raise Exception
                # return
                model = models.create(**model_configs).to(self.device)
                # model = self.model
                model.load_state_dict(checkpoint['state_dict'])
                # model = checkpoint['model'].to(self.device)
                logits, targets, imgs_names = get_logits_all(model, data_loader, device = self.device)
                logits_soft = nn.Softmax(dim=1)(logits)
                if not got_first:
                    logits_final = logits_soft
                    got_first = True
                else:
                    logits_final = logits_final+ logits_soft
            logits_final = logits_final/len(paths_ids)
        else:
            logits_final, targets, imgs_names = get_logits_all(self.model, data_loader, device = self.device)
        # logits = torch.argmax(logits_final, axis = 1)
        logits = logits_final # don't take argmax if you need top k
        pred_idx = torch.argmax(logits_final, axis = 1)
        probs = logits_final[pred_idx]
        df_probs = pd.DataFrame({'id': imgs_names, 'prob': [logits_final[i][pred_idx[i]].item() for i in range(len(pred_idx.tolist()))], 'pred_class': pred_idx, 'true_class': targets})
        df_probs.to_csv('probs_train.csv', index=False)
        # logits, targets = get_logits_all(self.model, data_loader, device = self.device)
        confusion_matrix = conf_matrix(logits, targets)
        acc_ , prec_, rec_, f1_, top2acc_ = evaluate_all(logits, targets)
        print("Accuracy: ", acc_, "  Precision: ", prec_, "  Recall: ", rec_, " F1: ", f1_, " Top2: ", top2acc_)
        print("Confusion_matrix: \n", confusion_matrix)
        return acc_ , prec_, rec_, f1_, top2acc_, confusion_matrix

    def predict(self, data_loader, classes_str, ensemble = False, paths_ids = [], num_pred_per_model = 3):
        if ensemble:
            got_first = False
            for idx, path in enumerate(paths_ids):
                if osp.exists(path):
                    checkpoint = load_checkpoint(paths_ids[idx])
                else:
                    # download from google drive:
                    self.download(paths_ids[idx], save_to = '/content/Data_Challenge/datachallenge/downloaded_model.tar')
                    checkpoint = load_checkpoint('/content/Data_Challenge/datachallenge/downloaded_model.tar')
                model_configs = checkpoint['configs']
                # model_configs["name"] = "efficientnet_b5"
                model = models.create(**model_configs).to(self.device)
                # model = self.model
                model.load_state_dict(checkpoint['state_dict'])

                # model = checkpoint['model'].to(self.device)
                for i in range(num_pred_per_model):
                    imgs_names, logits = get_logits_all_test(model, data_loader, device = self.device)
                    logits_soft = nn.Softmax(dim=1)(logits)
                    if not got_first:
                        logits_final = logits_soft
                        got_first = True
                    else:
                        logits_final = logits_final+ logits_soft
            logits_final = logits_final/(len(paths_ids)*num_pred_per_model)
            logits = torch.argmax(logits_final, axis = 1)
            probs = logits_final[logits]
            df = pd.DataFrame({'id': imgs_names, 'Category': [classes_str[i] for i in logits.tolist()]})
            df.to_csv('submission_ensemble.csv', index=False)
            df_probs = pd.DataFrame({'id': imgs_names, 'prob': [logits_final[i][logits[i]].item() for i in range(len(logits.tolist()))]})
            df_probs.to_csv('probs.csv', index=False)
        else:
            imgs_names, logits = get_logits_all_test(self.model, data_loader, device = self.device)
            logits_soft = nn.Softmax(dim=1)(logits)
            logits = torch.argmax(logits_soft, axis = 1)
            df = pd.DataFrame({'id': imgs_names, 'Category': [classes_str[i] for i in logits.tolist()]})
            df.to_csv('submission.csv', index=False)


    def download(self, id, save_to = './downloaded_model.zip'):
        file = gdown.download(id=id, output=save_to, quiet=False )