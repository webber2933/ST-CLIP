import torch

from .sigmoid_focal_loss import SigmoidFocalLoss
from .softmax_focal_loss import SoftmaxFocalLoss

__all__ = [
           "SigmoidFocalLoss", "SoftmaxFocalLoss",
          ]

