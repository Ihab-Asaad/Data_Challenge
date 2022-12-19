import torch
from torch import nn
import gdown
from datachallenge.utils.serialization import load_checkpoint


class CustomCrossEntropyLoss(nn.CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.download(paths_ids[idx], save_to = '/content/Data_Challenge/datachallenge/downloaded_model.tar')
        checkpoint = load_checkpoint('/content/Data_Challenge/datachallenge/downloaded_model.tar')
        model_configs = checkpoint['configs']
        model = models.create(**model_configs).to(self.device)
        model.load_state_dict(checkpoint['state_dict'])

        self.model2weights = model
        self.model2weights.eval()

    def download(self, id, save_to = './downloaded_model.zip'):
        file = gdown.download(id=id, output=save_to, quiet=False )

    def forward(self, input: Tensor, target: Tensor, imgs: Tensor) -> Tensor:
        # input here is the logits
        # weights = # get the images and take the model(imgs) and then softamx as weights
        logits = self.model2weights(imgs)
        logits_soft = nn.Softmax(dim=1)(logits)
        input_weights = logits_soft[target]
        return F.cross_entropy(input, target, weight=input_weights,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

                               