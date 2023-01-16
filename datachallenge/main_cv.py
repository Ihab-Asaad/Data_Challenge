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

from datachallenge import datasets
from datachallenge import models
from datachallenge.trainers import Trainer
from datachallenge.evaluators import Evaluator
from datachallenge.utils.data import transformers as T
from datachallenge.utils.data.preprocessor import Preprocessor
from datachallenge.utils.logging import Logger
from datachallenge.utils.serialization import load_checkpoint, save_checkpoint
from pytorch_metric_learning import samplers
from datachallenge.loss.loss_fn import CustomCrossEntropyLoss
import gdown
from torchviz import make_dot
from sklearn.model_selection import KFold, StratifiedKFold
from visualize import visualize


def get_data(name, cross_val, num_folds, val_split, test_split, data_dir, combine_trainval):

    extract_to = osp.join(data_dir, name)
    # pass the random state number to split the data for two models in different ways , or apply CV
    # create dataset:
    dataset = datasets.create(
        name, extract_to, val_split=val_split, test_split=test_split, download=True)

    # create test dataset, this is the unlabeled data to be submitted:
    # dataset_test = datasets.create('test_data', osp.join(data_dir, 'test_data_submit'), download = True)
    dataset_test = datasets.create('test_data', extract_to, download=True)

    return dataset, dataset_test, dataset.num_classes


def dataset_dataloader(dataset, dataset_test, height, width, batch_size, workers, combine_trainval):
    # All pretrained torchvision models have the same preprocessing, which is to normalize as following (input is RGB format):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # get the data portions, to pass them later to datalaoders:
    print("combine_trainval: ", combine_trainval)
    train_set = (dataset.X_trainval, dataset.y_trainval) if combine_trainval else (
        dataset.X_train, dataset.y_train)
    val_set = (dataset.X_val, dataset.y_val)
    test_set = (dataset.X_test, dataset.y_test)
    test_set_submit = (dataset_test.X)
    all_dataset = (dataset.X, dataset.y)
    # num_classes = dataset.num_classes

    # define some transformers before passing the image to our model:
    train_transformer = T.Compose([
        T.train_tranforms(height, width),
        # T.RectScale(height, width),
        # T.ToTensor(),
        # normalizer,
        # T.RandomSizedRectCrop(height, width),
        # T.RandomHorizontalFlip(),
        # convert PIL(RGB) or numpy(type: unit8) in range [0,255] to torch tensor a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        # T.ToTensor(),
    ])

    test_transformer = T.Compose([
        T.test_tranforms(height, width),
        # T.RectScale(height, width),
        # T.ToTensor(),
        # normalizer,
    ])

    test_submit_transformer = T.Compose([
        T.test_tranforms(height, width),
        # T.RectScale(height, width),
        # T.ToTensor(),
        # normalizer,
    ])

    # https://pytorch.org/docs/master/data.html#torch.utils.data.sampler.WeightedRandomSampler
    # https://stackoverflow.com/questions/67535660/how-to-construct-batch-that-return-equal-number-of-images-for-per-classes

    # https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/3
    # train_loader = DataLoader(
    #     # Preprocessor is the main class, pass dataset with path to images and transformer, override len , getitem
    #     Preprocessor(train_set, root=dataset.images_dir,
    #                  transform=train_transformer),
    #     batch_size=batch_size, num_workers=workers,
    #     shuffle=True,
    #     pin_memory=True, # avoid one implicit CPU-to-CPU copy, from paged CPU memory to non-paged CPU memory, which is required before copy tensor to cuda using x.cuda().
    #     drop_last=True)
    loader = DataLoader(Preprocessor(
        train_set, root=dataset.images_dir, transform=train_transformer))
    labels_list = []
    for _, label, _ in loader:  # return img, target, image_path
        labels_list.append(label)
    labels = torch.LongTensor(labels_list)
    # balanced_sampler = samplers.MPerClassSampler(labels, 2, length_before_new_iter = 2*len(labels)) # does this requires deleting weights?, count the number of images in an epoch, check if the same number of the dataset
    # sample from distribution of weights: sampler = WeightedRandomSampler(weights  = sample_weights, num_samples = len(labels), replacement = True)
    train_loader = DataLoader(
        # Preprocessor is the main class, pass dataset with path to images and transformer, override len , getitem
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        # shuffle=True,
        shuffle=True,
        # sampler=balanced_sampler,
        # avoid one implicit CPU-to-CPU copy, from paged CPU memory to non-paged CPU memory, which is required before copy tensor to cuda using x.cuda().
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
    # All pretrained torchvision models have the same preprocessing, which is to normalize as following (input is RGB format):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_set = (X_test, y_test)
    test_set_submit = (X_test_submit)

    test_transformer = T.Compose([
        T.test_tranforms(height, width),
        # T.RectScale(height, width),
        # T.ToTensor(),
        # normalizer,
    ])

    test_submit_transformer = T.Compose([
        T.test_tranforms(height, width),
        # T.RectScale(height, width),
        # T.ToTensor(),
        # normalizer,
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
    # All pretrained torchvision models have the same preprocessing, which is to normalize as following (input is RGB format):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    # get the data portions, to pass them later to datalaoders:
    train_set = (X_train, y_train)
    val_set = (X_val, y_val)

    # define some transformers before passing the image to our model:
    train_transformer = T.Compose([
        T.train_tranforms(height, width),
        # T.RectScale(height, width),
        # T.ToTensor(),
        # normalizer,
        # T.RandomSizedRectCrop(height, width),
        # T.RandomHorizontalFlip(),
        # convert PIL(RGB) or numpy(type: unit8) in range [0,255] to torch tensor a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        # T.ToTensor(),
    ])

    test_transformer = T.Compose([
        T.test_tranforms(height, width),
        # T.RectScale(height, width),
        # T.ToTensor(),
        # normalizer,
    ])
    print(train_set[0][0])
    loader_weights = DataLoader(
        Preprocessor(train_set, root=images_dir, transform=train_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False)

    # imgs_weights_dict = visualize()
    # labels_list, imgs_weights = [],[]
    # for ii, (imgs, classes, imgs_names_batch) in enumerate(loader_weights):
    #     labels_list.extend(classes)
    #     imgs_weights.extend([imgs_weights_dict[path_img] for path_img in imgs_names_batch])
    # labels = torch.LongTensor(labels_list)
    # print(len(imgs_weights))
    # balanced_sampler = samplers.MPerClassSampler(labels, 2, length_before_new_iter = 2*len(labels)) # does this requires deleting weights?, count the number of images in an epoch, check if the same number of the dataset
    # sample from distribution of weights: sampler = WeightedRandomSampler(weights  = sample_weights, num_samples = len(labels), replacement = True)
    # sampler = WeightedRandomSampler(weights  = imgs_weights, num_samples = 3*len(labels), replacement = True)
    train_loader = DataLoader(
        # Preprocessor is the main class, pass dataset with path to images and transformer, override len , getitem
        Preprocessor(train_set, root=images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True,
        # shuffle = True,
        # sampler=balanced_sampler,
        # sampler = sampler,
        # avoid one implicit CPU-to-CPU copy, from paged CPU memory to non-paged CPU memory, which is required before copy tensor to cuda using x.cuda().
        pin_memory=True,
        drop_last=True)

    val_loader = DataLoader(
        Preprocessor(val_set, root=images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    print(len(val_set[0]))

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
    # Load from checkpoint:
    # print learning rate while training to check if resuming take the currect lr
    global start_epoch
    global best_top1
    start_epoch = 0
    best_top1 = 0
    if args["training_configs"]["resume"] != '':
        print("Load from saved model...")
        # if osp.exists(args["training_configs"]["resume"]):
        if osp.exists(log_path):
            checkpoint = load_checkpoint(
                osp.join(log_path, 'model_best.pth.tar'))
        else:
            if osp.exists(args["training_configs"]["resume"]):
                checkpoint = load_checkpoint(
                    osp.join(args["training_configs"]["resume"], 'model_best.pth.tar'))
        # else:
        # # download from google drive:
        # #
        #     file = gdown.download(id=args["training_configs"]["resume"], output='/content/Data_Challenge/datachallenge/downloaded_model.tar', quiet=False )
        #     checkpoint = load_checkpoint('/content/Data_Challenge/datachallenge/downloaded_model.tar')
        # model = models.create("resnet50", num_features=256,
        #           dropout=0.2, num_classes=8).to(self.device)
        model_configs = checkpoint['configs']
        model = models.create(**model_configs).to(args["device"])
        # model = self.model
        model.load_state_dict(checkpoint['state_dict'])
        # checkpoint = load_checkpoint(args["training_configs"]["resume"])
        # model_configs = checkpoint['configs']
        # model = models.create(**model_configs).to(device)
        # model.load_state_dict(checkpoint['state_dict'])
        # print("Device: ",device)
        # model = checkpoint['model'].to(device) # is to(device necessary)
        # model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    # model = nn.DataParallel(model).cuda() # this add attribute 'module' to model
    else:
        # Create model
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
        get_data(args["dataset"]["name"], args["training_configs"]["cv"], args["training_configs"]["folds"], args["training_configs"]["val_split"],
                 args["training_configs"]["test_split"], args["logging"]["data_dir"], args["training_configs"]["combine_trainval"])
    args["num_classes"] = dataset.num_classes
    args["device"] = device
    # add ensemble to .yaml file
    evaluator = Evaluator(create_model(args), device)
    train_loader, val_loader, test_loader, test_submit_loader, alldata_loader = dataset_dataloader(
        dataset, dataset_test, height, width, args["training"]["batch_size"], args["training"]["workers"], args["training_configs"]["combine_trainval"])
    ensemble = True  # for sure
    paths = []
    if args["training_configs"]["predict"]:
        print("Prediction:")
        # you have to test the accuracy of ensemble on training dataloader before submission.
        models_names = ['resnet50']
        # for model_name in models_names:
        # pass the paths of the trained models first, otherwise download from google:
        evaluator.predict(test_submit_loader, dataset.classes_str, ensemble=True,
                          paths_ids=args['training_configs']['path_to_models'])
        return
    if args["training_configs"]["evaluate"]:
        # metric.train(model, train_loader)
        paths_ids = args['training_configs']['path_to_models']
        print(paths_ids)
        # print("Validation:")
        # evaluator.evaluate(val_loader, ensemble = True, paths_ids = paths_ids)
        print("Test:")
        evaluator.evaluate(test_loader, ensemble=True, paths_ids=paths_ids)
        # print("Train:") #
        # evaluator.evaluate(train_loader, ensemble = True, paths_ids = paths_ids)

        # # make a folder for misclassified images:
        return

    if args["training_configs"]["cv"]:
        # kf = KFold(n_splits=args["training_configs"]["folds"], random_state=42, shuffle=True)
        kf = StratifiedKFold(
            n_splits=args["training_configs"]["folds"], random_state=42, shuffle=True)
    else:
        # kf = KFold(n_splits=10, random_state=42, shuffle=True) # and break after the first fold
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

        # train_loader, val_loader, test_loader, test_submit_loader = dataset_dataloader(dataset, dataset_test, height, width, args["training"]["batch_size"], args["training"]["workers"], args["training_configs"]["combine_trainval"])
        log_path = args["logging"]["logs_dir"] + "_" + str(i)
        path_to_models.append(log_path)

        model = create_model(args, log_path)

        # create summary:
        total_parameters = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel()
                                   for p in model.parameters() if p.requires_grad)
        print("Total parameters: ", total_parameters,
              "Trainable parameters: ", trainable_parameters)
        # print(model)

        # Distance metric, will be added later
        # metric = DistanceMetric(algorithm=args["metric_learning"]["dist_metric"], device = device)

        # Evaluator
        evaluator = Evaluator(model, device)

        # Criterion: pass weights to loss function:
        repeat = dataset.weights_trainval if args["training_configs"][
            "combine_trainval"] else dataset.weights_train

        torch_repeat = torch.Tensor(repeat)
        class_weights = sum(torch_repeat)/torch_repeat
        # criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
        custom_loss = False  # make it in .yaml
        if custom_loss:
            criterion = CustomCrossEntropyLoss(device=device).cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()

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

        # in case you are using a pretrained model, train its weights with lower lr to not destroy the prelearned features.
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
        trainer = Trainer(model, criterion, device, custom_loss)

        # print lr with metrics:
        # Schedule learning rate, see automation functions in torch, see also:
        # https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        def adjust_lr(epoch):
            step_size = 60 if args["net"]["arch"] == 'inception' else 40
            lr = args["training"]["lr"] * (0.1 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

        # Start training

        for epoch in range(start_epoch, args["training"]["epochs"]):
            adjust_lr(epoch)
            trainer.train(epoch, train_loader, optimizer)
            # trainer.train(epoch, alldata_loader, optimizer)
            if epoch < args["training_configs"]["start_save"]:
                continue
            metrics_ = evaluator.evaluate(val_loader)
            top1 = metrics_[1]  # acc macro
            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)

            configs = models.get_configs(args, num_classes)
            save_checkpoint({
                # 'state_dict': model.module.state_dict(),
                'state_dict': model.state_dict(),
                # 'model': model,
                'epoch': epoch + 1,
                'best_top1': best_top1,
                'configs': configs,
            }, is_best, fpath=osp.join(log_path, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, top1, best_top1, ' *' if is_best else ''))

    # path_to_models = ['1Qajyh0DLb4PlEQ2W5H29LiSoyRPeR8CL&confirm=t', \
    #             '1zbRMjmk7I0V4Wupla7g3d-eVxbXRg4LA&confirm=t', \
    #             '1Au0w6ZxY-0BRN631vZFtmWya6UzDiaUy&confirm=t', \
    #             '1iT-2TdDQYDbka9lf6-S40oNxVSm-Co-O&confirm=t', \
    #             '1YgOZ_fQsGFqLHqBZwYq_6lzimqvynZsP&confirm=t', \
    #             "1y3_QidklP12vYe1C3Sdl1RrS0DNDRN0Q&confirm=t", \
    #             "1wuBa5R5DiPCo-96-euXQlckuKgFE9gln&confirm=t", \
    #             "1aUMvJKEfya-u1ihM0FjIVIMqPGilHNJq&confirm=t", \
    #             "1QdtciWd4VHyKYb9g30O-HayTvYDmbffO&confirm=t", \
    #             "1zF8f-0G1O98YVP5CaHx-SsphNKEJJDyg&confirm=t"]
    # paths_ids = ["1y3_QidklP12vYe1C3Sdl1RrS0DNDRN0Q&confirm=t", \
    #             "1wuBa5R5DiPCo-96-euXQlckuKgFE9gln&confirm=t", \
    #             "1aUMvJKEfya-u1ihM0FjIVIMqPGilHNJq&confirm=t", \
    #             "1QdtciWd4VHyKYb9g30O-HayTvYDmbffO&confirm=t", \
    #             "1zF8f-0G1O98YVP5CaHx-SsphNKEJJDyg&confirm=t"]
    # paths_ids = ["/content/Data_Challenge/datachallenge/logs/test_loss_0/model_best.pth.tar",
    #              "/content/Data_Challenge/datachallenge/logs/test_loss_1/model_best.pth.tar",
    #              "/content/Data_Challenge/datachallenge/logs/test_loss_2/model_best.pth.tar",
    #              "/content/Data_Challenge/datachallenge/logs/test_loss_3/model_best.pth.tar",
    #              "/content/Data_Challenge/datachallenge/logs/test_loss_4/model_best.pth.tar"]
    paths_ids = [osp.join(path, 'model_best.pth.tar')
                 for path in path_to_models]
    # # paths_ids = [osp.join(path,'checkpoint.pth.tar') for path in path_to_models]
    # # print("Validation:")
    # # evaluator.evaluate(val_loader, ensemble = True, paths_ids = paths_ids)
    # # print("Test:")
    # # evaluator.evaluate(test_loader, ensemble = True, paths_ids = paths_ids)
    # # print("Train:") #
    # # evaluator.evaluate(train_loader, ensemble = True, paths_ids = paths_ids)

    print("All data:")
    evaluator.evaluate(alldata_loader, ensemble=True, paths_ids=paths_ids)

    print("Predict:")
    evaluator.predict(test_submit_loader, dataset.classes_str,
                      ensemble=True, paths_ids=paths_ids)

    # # Final test
    # print('Test with best model:')
    # checkpoint = load_checkpoint(osp.join(args["logging"]["logs_dir"], 'model_best.pth.tar'))
    # # model.module.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(checkpoint['state_dict'])
    # # model = torch.load(checkpoint['model'])
    # # metric.train(model, train_loader)
    # evaluator.evaluate(test_loader)

    # # Predict on external test set:


if __name__ == '__main__':
    with open(r'/content/Data_Challenge/datachallenge/config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        args = yaml.safe_load(file)
    # print(type(args['training_configs']['path_to_models'][0]))
    main(args)

    # check:
    # https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader
    # torch.no_grad()(then no need .detach() , or for example var.detach().cpu().numpy()) with model.eval() : https://stackoverflow.com/questions/55322434/how-to-clear-cuda-memory-in-pytorch
