from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp

import numpy as np
import random
import yaml
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import gc

from datachallenge import datasets
from datachallenge import models
# from datachallenge.dist_metric import DistanceMetric
from datachallenge.trainers import Trainer
from datachallenge.evaluators import Evaluator
from datachallenge.utils.data import transformers as T
from datachallenge.utils.data.preprocessor import Preprocessor
from datachallenge.utils.logging import Logger  
from datachallenge.utils.serialization import load_checkpoint, save_checkpoint


def get_data(name, val_split, test_split, data_dir, height, width, batch_size, workers, combine_trainval):
    extract_to = osp.join(data_dir, name)
    
    dataset = datasets.create(name, extract_to, val_split= val_split, test_split= test_split, download = True)
    dataset_test = datasets.create('test_data', osp.join(data_dir, 'test_data_submit'), download = True)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = (dataset.X_trainval, dataset.y_trainval) if combine_trainval else (dataset.X_train, dataset.y_train)
    val_set = (dataset.X_val, dataset.y_val)
    test_set = (dataset.X_test, dataset.y_test)
    test_set_submit = (dataset_test.X)
    num_classes = dataset.num_classes

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    test_submit_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        # Preprocessor is the main class, pass dataset with path to images and transformer, override len , getitem
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, 
        pin_memory=True, # avoid one implicit CPU-to-CPU copy, from paged CPU memory to non-paged CPU memory, which is required before copy tensor to cuda using x.cuda().
        drop_last=True) 

    val_loader = DataLoader(
        Preprocessor(val_set, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    print(len(val_set[0]))

    test_loader = DataLoader(
        Preprocessor(test_set, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_submit_loader = DataLoader(
        Preprocessor(test_set_submit, root = dataset_test.images_dir, transform=test_submit_transformer),
        batch_size = 1, num_workers=workers, shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader, test_submit_loader

def seed_all(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  cudnn.deterministic = True
  cudnn.benchmark = True # to speed up the training, when computation graph does not change (fixed input size)
  torch.cuda.manual_seed_all(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)


def main(args):
    seed_all(args["training_configs"]["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache() 
    gc.collect()
    
    # Redirect print to both console and log file
    if not args["training_configs"]["evaluate"]:
        sys.stdout = Logger(osp.join(args["logging"]["logs_dir"], 'log.txt'))
    
    # Create data loaders
    height = args["net"]["height"]
    width = args["net"]["width"]
    if height is None or width is None:
        height, width = (224, 224) 

    dataset, num_classes, train_loader, val_loader, test_loader, test_submit_loader = \
        get_data(args["dataset"]["name"], args["training_configs"]["val_split"], \
        args["training_configs"]["test_split"], args["logging"]["data_dir"], height, width, \
        args["training"]["batch_size"], args["training"]["workers"], args["training_configs"]["combine_trainval"])

    
    # Create model
    model = models.create(args["net"]["arch"], num_features=args["training"]["features"],
                          dropout=args["training"]["dropout"], num_classes=num_classes).to(device) # no need to use .to(device) as below we are using DataParallel
    # print(model)
    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args["training_configs"]["resume"]:
        print("Load from saved model...")
        checkpoint = load_checkpoint(args["training_configs"]["resume"])
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    # model = nn.DataParallel(model).cuda() # this add attribute 'module' to model
    
    # Distance metric
    # metric = DistanceMetric(algorithm=args["metric_learning"]["dist_metric"], device = device)
    
    # Evaluator
    evaluator = Evaluator(model, device)

    if args["training_configs"]["predict"]:
        print("Prediction:")
        evaluator.predict(test_submit_loader, dataset.classes_str)
        return

    if args["training_configs"]["evaluate"]:
        # metric.train(model, train_loader)
        print("Validation:")
        evaluator.evaluate(val_loader)
        print("Test:")
        evaluator.evaluate(test_loader)
        # make a folder for misclassified images:
        return
    # Criterion
    repeat = dataset.weights_trainval if args["training_configs"]["combine_trainval"] else dataset.weights_train
    print(repeat)
    torch_repeat = torch.Tensor(repeat)
    class_weights = sum(torch_repeat)/torch_repeat
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda() 

    # Optimizer
    # if hasattr(model.module, 'base'):
    #     base_param_ids = set(map(id, model.module.base.parameters()))
    #     new_params = [p for p in model.parameters() if
    #                   id(p) not in base_param_ids]
    #     param_groups = [
    #         {'params': model.module.base.parameters(), 'lr_mult': 0.1},
    #         {'params': new_params, 'lr_mult': 1.0}]
    # else:
    #     param_groups = model.parameters()
    # optimizer = torch.optim.SGD(param_groups, lr=args["training"]["lr"],
    #                             momentum=args["training"]["momentum"],
    #                             weight_decay=args["training"]["weight_decay"],
    #                             nesterov=True)
    if hasattr(model, 'base'):
        base_param_ids = set(map(id, model.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=args["training"]["lr"],
                                momentum=args["training"]["momentum"],
                                weight_decay=args["training"]["weight_decay"],
                                nesterov=True)
    # optimizer = torch.optim.Adam(param_groups, lr=args["training"]["lr"],
    #                             weight_decay=args["training"]["weight_decay"])

    # Trainer
    trainer = Trainer(model, criterion, device)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 60 if args["net"]["arch"] == 'inception' else 40
        lr = args["training"]["lr"] * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args["training"]["epochs"]):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)
        if epoch < args["training_configs"]["start_save"]:
            continue
        metrics_ = evaluator.evaluate(val_loader)
        top1 = metrics_[3] # accuracy
        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            # 'state_dict': model.module.state_dict(),
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args["logging"]["logs_dir"], 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args["logging"]["logs_dir"], 'model_best.pth.tar'))
    # model.module.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    # metric.train(model, train_loader)
    evaluator.evaluate(test_loader)

    # Predict on external test set:


if __name__ == '__main__':
    with open(r'/content/Data_Challenge/datachallenge/config.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
        args = yaml.safe_load(file)
    main(args)

    # check: 
    # https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader
    # torch.no_grad()(then no need .detach() , or for example var.detach().cpu().numpy()) with model.eval() : https://stackoverflow.com/questions/55322434/how-to-clear-cuda-memory-in-pytorch