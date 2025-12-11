import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from timm.models.registry import register_model
from timm.models.resnet import ResNet 
__all__ = ["mobilenet_v2", ]
@register_model
def mobilenet_v2(**kwargs):
    model = tv.mobilenet_v2(num_classes = 200,pretrained=False)
    return model    