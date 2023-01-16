from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from datachallenge.evaluation_metrics import accuracy
from datachallenge.utils.meters import AverageMeter
from torch.utils.tensorboard import SummaryWriter


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
        # writer = SummaryWriter(log_dir='./')
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            # x = inputs.to(device)
            # y = targets.to(device)
            # writer.add_image('Input Images', inputs[1,:,:,:], i)
            # writer.close()
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
        # writer.close()
    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, classes, _ = inputs  # ignore img_name
        # inputs = [Variable(imgs)] # depricated
        # targets = Variable(pids.cuda())
        # Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
        inputs = imgs.to(self.device)
        targets = classes.to(self.device)
        # inputs = imgs # if using Dataparallel, you can use this two lines without .to(device), as input can be on any device, including CPU
        # targets = pids
        return inputs, targets

    def _forward(self, inputs, targets):
        # outputs = self.model(*inputs)

        outputs = self.model(inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss) and not self.customloss:
            if self.device == torch.device('cpu'):
                loss = self.criterion(outputs, targets.type(torch.LongTensor)) # for cpu
            else:
                loss = self.criterion(outputs, targets) # for GPU, will be fixed later
            prec = accuracy(outputs, targets)
            # prec= accuracy_micro(outputs, targets)

            # prec = prec[0]
        else:
            if self.customloss:
                # check the loss device ??
                loss = self.criterion(outputs, targets, inputs)
                prec = accuracy(outputs, targets)
            else:
                raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec
