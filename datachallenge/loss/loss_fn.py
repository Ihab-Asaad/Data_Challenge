import torch
from torch import nn

class CustomCrossEntropyLoss(nn.CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, model, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.model2weights = model
        self.model2weights.eval()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # weights = # get the images and take the model(imgs) and then softamx as weights
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)