from __future__ import absolute_import

from torchvision.transforms import * # Normalize, ToTensor, RandomHorizontalFlip,...
from PIL import Image
import random
import math
import albumentations as albu # see: https://albumentations.ai/docs/getting_started/image_augmentation/
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np

# All transformations accept PIL Image, Tensor Image or batch of Tensor Images as input

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)

class SomeTrans():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        # IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.transform = albu.Compose([
            # read about transforms and add more from here: https://albumentations.ai/docs/api_reference/augmentations/transforms/
                albu.RandomResizedCrop(height=self.width, width=self.height, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, p=1.0),
                albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=0, value=0, p=0.25),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.OneOf([
                    albu.MotionBlur(p=.2),
                    albu.MedianBlur(blur_limit=3, p=0.1),
                    albu.Blur(blur_limit=3, p=0.1),
                ], p=0.25),
                # albu.OneOf([
                #     albu.CLAHE(clip_limit=2),
                #     albu.IAASharpen(),
                #     albu.IAAEmboss(),
                #     albu.RandomBrightnessContrast(),            
                # ], p=0.25),
                albu.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.25),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ])
    def __call__(self, img):
        image = np.array(img) # albu takes np array as input
        image = self.transform(image=image)['image']
        return image
