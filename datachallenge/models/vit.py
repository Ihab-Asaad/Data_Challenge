import torch
from torch import nn # delete it 
# from vit_pytorch import ViT
from pytorch_pretrained_vit import ViT
import timm


# reference : https://github.com/lucidrains/vit-pytorch
class ViTransformer(nn.Module):
    # __factory = {
    #     18: torchvision.models.resnet18,
    #     34: torchvision.models.resnet34,
    #     50: torchvision.models.resnet50,
    #     101: torchvision.models.resnet101,
    #     152: torchvision.models.resnet152,
    # }

    def __init__(self, depth = 6, pretrained= True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, train_pretrained = True):
        super(ViTransformer, self).__init__()

        # self.v = ViT(
        #     image_size = 224,
        #     patch_size = 32,
        #     num_classes = num_classes,
        #     dim = 1024,
        #     depth = 6,
        #     heads = 16,
        #     mlp_dim = 2048,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )
        # https://pytorch.org/tutorials/beginner/vt_tutorial.html
        # self.v = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True, num_classes = num_classes)
        self.v = ViT('B_16_imagenet1k', pretrained=True, num_classes = num_classes, image_size = 224)
        # https://github.com/lukemelas/PyTorch-Pretrained-ViT/blob/master/pytorch_pretrained_vit/model.py
    def forward(self, x):
        return self.v(x)

def VisionTransformer(**kwargs):
    return ViTransformer(**kwargs)