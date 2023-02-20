from __future__ import print_function, absolute_import
import os
import os.path as osp

import numpy as np
import random
import yaml
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
import gc
import math

from datachallenge import datasets
from datachallenge import models
from datachallenge.trainers import Trainer
from datachallenge.evaluators import Evaluator
from datachallenge.utils.data import transformers as T
from datachallenge.utils.data.preprocessor import Preprocessor
from datachallenge.utils.logging import Logger
from datachallenge.utils.serialization import load_checkpoint, save_checkpoint
from datachallenge.loss.loss_fn import CustomCrossEntropyLoss
from sklearn.model_selection import StratifiedKFold


def get_data(name, val_split, test_split, data_dir, user_name = '', key = ''):
    extract_to = osp.join(data_dir, name)
    # pass the random state number to split the data for two models in different ways , or apply CV
    # create dataset:
    dataset = datasets.create(
        name, extract_to, val_split=val_split, test_split=test_split, download=True, user_name = user_name, key = key)

    # create test dataset, this is the unlabeled data to be submitted:
    dataset_test = datasets.create('test_data', extract_to, download=True, user_name = user_name, key = key)
    return dataset, dataset_test, dataset.num_classes


def dataset_dataloader(dataset, dataset_test, height, width, batch_size, workers, combine_trainval):
    print("combine_trainval: ", combine_trainval)
    train_set = (dataset.X_trainval, dataset.y_trainval) if combine_trainval else (
        dataset.X_train, dataset.y_train)
    val_set = (dataset.X_val, dataset.y_val)
    test_set = (dataset.X_test, dataset.y_test)
    test_set_submit = (dataset_test.X)
    all_dataset = (dataset.X, dataset.y)

    # define some transformers before passing the image to our model:
    train_transformer = T.Compose([
        T.train_tranforms(height, width)])

    test_transformer = T.Compose([
        T.test_tranforms(height, width)])

    test_submit_transformer = T.Compose([
        T.test_tranforms(height, width)])

    train_loader = DataLoader(
        # Preprocessor is the main class, pass dataset with path to images and transformer, override len , getitem
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

    val_loader = DataLoader(
        Preprocessor(val_set, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    alldata_loader = DataLoader(  # has to have its own transforms : just resize and convert to tensor.
        Preprocessor(all_dataset, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(test_set, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_submit_loader = DataLoader(
        Preprocessor(test_set_submit, root=dataset_test.images_dir,
                     transform=test_submit_transformer),
        batch_size=1, num_workers=workers, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, test_submit_loader, alldata_loader


def test_test_submit_dataloader(X_test, y_test, X_test_submit, images_dir, images_dir_test, height, width, batch_size, workers):
    test_set = (X_test, y_test)
    test_set_submit = (X_test_submit)

    test_transformer = T.Compose([
        T.test_tranforms(height, width),
    ])

    test_submit_transformer = T.Compose([
        T.test_tranforms(height, width),
    ])

    test_loader = DataLoader(
        Preprocessor(test_set, root=images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_submit_loader = DataLoader(
        Preprocessor(test_set_submit, root=images_dir_test,
                     transform=test_submit_transformer),
        batch_size=1, num_workers=workers, shuffle=False, pin_memory=True)

    return test_loader, test_submit_loader


def train_val_dataloader(X_train, y_train, X_val, y_val, images_dir, height, width, batch_size, workers):
    train_set = (X_train, y_train)
    val_set = (X_val, y_val)

    train_transformer = T.Compose([
        T.train_tranforms(height, width),
    ])

    test_transformer = T.Compose([
        T.test_tranforms(height, width)])

    train_loader = DataLoader(
        # Preprocessor is the main class, pass dataset with path to images and transformer, override len , getitem
        Preprocessor(train_set, root=images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

    val_loader = DataLoader(
        Preprocessor(val_set, root=images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return train_loader, val_loader


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    # to speed up the training, when computation graph does not change (fixed input size)
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


start_epoch = 0
best_top1 = 0


def create_model(args, log_path=''):
    global start_epoch
    global best_top1
    start_epoch = 0
    best_top1 = 0
    if args["training_configs"]["resume"] != '':
        print("Load from saved model...")
        # if osp.exists(args["training_configs"]["resume"]):
        if log_path!='' and osp.exists(log_path):
            checkpoint = load_checkpoint(
                osp.join(log_path, 'model_best.pth.tar'), device = args["device"])
        else:
            if osp.exists(args["training_configs"]["resume"]):
                checkpoint = load_checkpoint(args["training_configs"]["resume"], device = args["device"])
            else:
                print("Model or path ot exist...Creating new model")
                model = models.create(args["net"]["arch"], num_features=args["training"]["features"],
                              dropout=args["training"]["dropout"], num_classes=args["num_classes"]).to(args["device"])  # no need to use .to(device) as below we are using DataParallel
                return model
        model_configs = checkpoint['configs']
        model = models.create(**model_configs).to(args["device"])
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    else:
        # Create new model
        model = models.create(args["net"]["arch"], num_features=args["training"]["features"],
                              dropout=args["training"]["dropout"], num_classes=args["num_classes"]).to(args["device"])  # no need to use .to(device) as below we are using DataParallel
    return model


def main(args):
    global best_top1
    global start_epoch

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

    dataset, dataset_test, num_classes = \
        get_data(args["dataset"]["name"], args["training_configs"]["val_split"],
                 args["training_configs"]["test_split"], args["logging"]["data_dir"], args["dataset"]["user_name"], args["dataset"]["key"])
    args["num_classes"] = dataset.num_classes
    args["device"] = device
    # add ensemble to .yaml file
    evaluator = Evaluator(create_model(args), device)
    train_loader, val_loader, test_loader, test_submit_loader, alldata_loader = dataset_dataloader(
        dataset, dataset_test, height, width, args["training"]["batch_size"], args["training"]["workers"], args["training_configs"]["combine_trainval"])

    if args["training_configs"]["predict"]:
        print("Prediction:")
        evaluator.predict(test_submit_loader, dataset.classes_str, ensemble=True,
                          paths_ids=args['training_configs']['path_to_models'])
        return
    if args["training_configs"]["evaluate"]:
        paths_ids = args['training_configs']['path_to_models']

        # uncomment the following to check the accuarcy on val/test/training sets
        # print("Validation:")
        # evaluator.evaluate(val_loader, ensemble = True, paths_ids = paths_ids)
        print("Test:")
        evaluator.evaluate(test_loader, ensemble=True, paths_ids=paths_ids)
        # print("Train:")
        # evaluator.evaluate(train_loader, ensemble = True, paths_ids = paths_ids)
        return

    if args["training_configs"]["cv"]:
        kf = StratifiedKFold(
            n_splits=args["training_configs"]["folds"], random_state=42, shuffle=True)
    else:
        kf = StratifiedKFold(n_splits=3)

    test_loader, test_submit_loader = test_test_submit_dataloader(
        dataset.X_test, dataset.y_test, dataset_test.X, dataset.images_dir, dataset_test.images_dir, height, width, args["training"]["batch_size"], args["training"]["workers"])

    training_dataset_X, training_dataset_y = np.array(
        dataset.X), np.array(dataset.y)
    path_to_models = []
    for i, (train_index, test_index) in enumerate(kf.split(training_dataset_X, training_dataset_y)):
        print(" Fold: ", i)
        X_train_fold, y_train_fold = list(training_dataset_X[train_index]), list(
            training_dataset_y[train_index])
        X_val_fold, y_val_fold = list(training_dataset_X[test_index]), list(
            training_dataset_y[test_index])

        train_loader, val_loader = train_val_dataloader(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                                        dataset.images_dir, height, width, args["training"]["batch_size"], args["training"]["workers"])

        log_path = args["logging"]["logs_dir"] + "_" + str(i)
        path_to_models.append(log_path)

        model = create_model(args, log_path)

        # create summary:
        total_parameters = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel()
                                   for p in model.parameters() if p.requires_grad)
        print("Total parameters: ", total_parameters,
              "Trainable parameters: ", trainable_parameters)

        # Evaluator
        evaluator = Evaluator(model, device)

        # Criterion: pass weights to loss function:
        # repeat = dataset.weights_trainval if args["training_configs"][
        #     "combine_trainval"] else dataset.weights_train

        # torch_repeat = torch.Tensor(repeat)
        # class_weights = sum(torch_repeat)/torch_repeat
        custom_loss = False  # make it in .yaml
        if custom_loss:
            criterion = CustomCrossEntropyLoss(device=device).cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()

        # in case you are using a pretrained model, train its weights with lower lr to not destroy the prelearned features.
        if hasattr(model, 'base'):
            base_param_ids = set(map(id, model.base.parameters()))
            new_params = [p for p in model.parameters() if
                          id(p) not in base_param_ids]
            param_groups = [
                {'params': model.base.parameters(), 'lr_mult': 0.1},
                # {'params': model.base.parameters(), 'lr_mult': 1},
                {'params': new_params, 'lr_mult': 1}]
        else:
            param_groups = model.parameters()
        optimizer = torch.optim.SGD(param_groups, lr=args["training"]["lr"],
                                    momentum=args["training"]["momentum"],
                                    weight_decay=args["training"]["weight_decay"],
                                    nesterov=True)

        # Trainer
        trainer = Trainer(model, criterion, device, custom_loss)

        # def adjust_lr(optimizer, epoch, total_epochs, initial_lr):
        #     lr = initial_lr * \
        #         (1 + math.cos(math.pi * 3*epoch / total_epochs)) / 2
        #     for g in optimizer.param_groups:
        #         g['lr'] = lr * g.get('lr_mult', 1)

        def adjust_lr(optimizer, epoch, total_epochs, initial_lr):
            # step_size = 60 if args["net"]["arch"] == 'inception' else 40
            step_size = total_epochs
            lr = args["training"]["lr"] * (0.1 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

        # Start training
        for epoch in range(start_epoch, args["training"]["epochs"]):
            adjust_lr(optimizer, epoch,
                      args["training"]["epochs"], args["training"]["lr"])
            trainer.train(epoch, train_loader, optimizer)
            if epoch < args["training_configs"]["start_save"]:
                continue
            metrics_ = evaluator.evaluate(val_loader)
            top1 = metrics_[1]  # acc macro
            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)

            configs = models.get_configs(args, num_classes)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
                'configs': configs,
            }, is_best, fpath=osp.join(log_path, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, top1, best_top1, ' *' if is_best else ''))

    paths_ids = [osp.join(path, 'model_best.pth.tar')
                 for path in path_to_models]

    print("All data:")
    evaluator.evaluate(alldata_loader, ensemble=True, paths_ids=paths_ids)

    print("Predict:")
    evaluator.predict(test_submit_loader, dataset.classes_str,
                      ensemble=True, paths_ids=paths_ids)

def predict(images_path):
    seed_all(args["training_configs"]["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args["device"] = device
    torch.cuda.empty_cache()
    gc.collect()
    if not os.path.isdir(images_path) and not os.path.isfile(images_path):
        print(images_path, "is neither a folder nor a file.")
        return
    if os.path.isdir(images_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif'] # '.jpg' tested
        images = []
        for file in os.listdir(images_path):
            # this images_path should contain the images:
            _, file_extension = os.path.splitext(file)
            if file_extension in image_extensions:
                images.append(os.path.join(images_path, file))
        
    elif os.path.isfile(images_path):
        print(images_path, "is a file.")
        images = [images_path]
    test_set_submit = (images)
    test_submit_transformer = T.Compose([T.test_tranforms(args["net"]["height"], args["net"]["width"])])
    test_submit_loader = DataLoader(
            Preprocessor(test_set_submit, root='',
                    transform=test_submit_transformer),
    batch_size=1, num_workers=1, shuffle=False, pin_memory=True)
    classes_str = [1, 20, 21, 22, 32, 41, 44, 45]
    args["num_classes"] = len(classes_str)
    evaluator = Evaluator(create_model(args), device)
    
    evaluator.predict(test_submit_loader, classes_str, ensemble=True,
                        paths_ids=args['training_configs']['path_to_models'])

if __name__ == '__main__':
    with open(r'./config.yaml') as file:
        args = yaml.safe_load(file)
    if args["task"] == "I&O":
        predict(args["image_path"])
    else:
        main(args)
