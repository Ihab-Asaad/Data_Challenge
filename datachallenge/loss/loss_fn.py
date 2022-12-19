import torch
from torch import nn
import gdown
from datachallenge.utils.serialization import load_checkpoint
from datachallenge import models
from torch.nn import functional as F


class CustomCrossEntropyLoss(nn.CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, device):
        super(CustomCrossEntropyLoss, self).__init__()
        self.device = device
        google_id = "1HrBMuIIdXwBPGkYmYF2iPE75QLPVDrl3&confirm=t"
        self.download(google_id, save_to = '/content/Data_Challenge/datachallenge/model_loss.tar')
        checkpoint = load_checkpoint('/content/Data_Challenge/datachallenge/model_loss.tar')
        model_configs = checkpoint['configs']
        model = models.create(**model_configs).to(self.device)
        model.load_state_dict(checkpoint['state_dict'])

        self.model2weights = model
        self.model2weights.eval()

    def download(self, id, save_to = './downloaded_model.zip'):
        file = gdown.download(id=id, output=save_to, quiet=False )

    def forward(self, input, target, imgs):
        # input here is the logits
        # weights = # get the images and take the model(imgs) and then softamx as weights
        logits = self.model2weights(imgs)
        logits_soft = nn.Softmax(dim=1)(logits)
        input_weights = logits_soft[target]
        print(input_weights)
        print("Targets: ")
        print(target)
        return F.cross_entropy(input, target, weight=input_weights,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

