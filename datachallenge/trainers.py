from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from datachallenge.evaluation_metrics import accuracy
from datachallenge.utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, device, custom_loss):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.device = device
        self.customloss = custom_loss

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets, _ = self._parse_data(inputs)
            # x = inputs.to(device)
            # y = targets.to(device)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.data.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            ##
            # torch.cuda.empty_cache()
            ## 
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Acc {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, classes= inputs
        # inputs = [Variable(imgs)] # depricated 
        # targets = Variable(pids.cuda())
        inputs = imgs.to(self.device) # Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
        targets = classes.to(self.device)
        # inputs = imgs # if using Dataparallel, you can use this two lines without .to(device), as input can be on any device, including CPU 
        # targets = pids
        return inputs, targets

    def _forward(self, inputs, targets):
        # outputs = self.model(*inputs)

        outputs = self.model(inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss) and not self.customloss:
            loss = self.criterion(outputs, targets) # check the loss device ??
            prec= accuracy(outputs, targets)
            # prec= accuracy_micro(outputs, targets)
            
            # prec = prec[0]
        else:
            if self.customloss:
                loss = self.criterion(outputs, targets, inputs) # check the loss device ??
                prec= accuracy(outputs, targets)
            else:
                raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec