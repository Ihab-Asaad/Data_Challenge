
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

import torch
from torch.utils.data.sampler import Sampler
from pytorch_metric_learning import samplers
from pytorch_metric_learning.utils import common_functions as c_f


# # modified from
# # https://raw.githubusercontent.com/bnulihaixia/Deep_metric/master/utils/sampler.py
# class MPerClassSampler(Sampler):
#     """
#     At every iteration, this will return m samples per class. For example,
#     if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
#     each will be returned
#     """

#     def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000):
#         if isinstance(labels, torch.Tensor):
#             labels = labels.numpy()
#         self.m_per_class = int(m)
#         self.batch_size = int(batch_size) if batch_size is not None else batch_size
#         self.labels_to_indices = c_f.get_labels_to_indices(labels)
#         self.labels = list(self.labels_to_indices.keys())
#         self.length_of_single_pass = self.m_per_class * len(self.labels)
#         self.list_size = length_before_new_iter
#         if self.batch_size is None:
#             if self.length_of_single_pass < self.list_size:
#                 self.list_size -= (self.list_size) % (self.length_of_single_pass)
#         else:
#             assert self.list_size >= self.batch_size
#             assert (
#                 self.length_of_single_pass >= self.batch_size
#             ), "m * (number of unique labels) must be >= batch_size"
#             assert (
#                 self.batch_size % self.m_per_class
#             ) == 0, "m_per_class must divide batch_size without any remainder"
#             self.list_size -= self.list_size % self.batch_size

#     def __len__(self):
#         return self.list_size

#     def __iter__(self):
#         idx_list = [0] * self.list_size
#         i = 0
#         num_iters = self.calculate_num_iters()
#         for _ in range(num_iters):
#             c_f.NUMPY_RANDOM.shuffle(self.labels)
#             if self.batch_size is None:
#                 curr_label_set = self.labels
#             else:
#                 curr_label_set = self.labels[: self.batch_size // self.m_per_class]
#             for label in curr_label_set:
#                 t = self.labels_to_indices[label]
#                 idx_list[i : i + self.m_per_class] = c_f.safe_random_choice(
#                     t, size=self.m_per_class
#                 )
#                 i += self.m_per_class
#         return iter(idx_list)

#     def calculate_num_iters(self):
#         divisor = (
#             self.length_of_single_pass if self.batch_size is None else self.batch_size
#         )
#         return self.list_size // divisor if divisor < self.list_size else 1

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
        
if __name__=="__main__":
    n_classes = 10
    n_samples = 2

    mnist_train =  torchvision.datasets.MNIST(root="mnist/mnist_train", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))

    # balanced_batch_sampler = BalancedBatchSampler(mnist_train, n_classes, n_samples)
    loader = DataLoader(mnist_train)
    labels_list = []
    for _, label in loader:
        labels_list.append(label)
    labels = torch.LongTensor(labels_list)
    print(labels.shape)
    print("here1")
    balanced_batch_sampler = samplers.MPerClassSampler(labels,2 , length_before_new_iter = len(labels))
    print("here2")
    dataloader = torch.utils.data.DataLoader(mnist_train, sampler=balanced_batch_sampler, batch_size = 30)
    my_testiter = iter(dataloader)
    # print(type(my_testiter))
    print(type(dataloader))
    # images, target = dataloader.next()
    # images,target = next(my_testiter)
    # print(target)
    # images,target = next(my_testiter)
    # print(target)
    j = 0
    labels_list=[]
    for _, label in dataloader:
        j = j+1
        labels_list.extend(label.tolist())
        print("Iter: , ", j)
        for i in range(10):
            print("i:, ", labels_list.count(i))

    # def imshow(img):
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # imshow(torchvision.utils.make_grid(images))