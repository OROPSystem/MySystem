import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models
import timm
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .util import set_parameter_requires_grad
from .registry import ModelRegistry

__all__ = ['resnet18', 'resnet50']


class ResNets(nn.Module):
    def __init__(self, model_name='resnet18', num_layers=12, num_classes=5, feature_extract=False, use_pretrained=True):
        super(ResNets, self).__init__()
        self.resnet = timm.create_model(model_name, pretrained=use_pretrained)
        # self.resnet = eval(model_name)(pretrained=use_pretrained)
        set_parameter_requires_grad(self.resnet, feature_extract)
        modules = list(self.resnet.children())
        self.feature_extract = nn.Sequential(*modules[0:-1], nn.Flatten())
        num_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_in_features, num_classes)

    def forward(self, x):
        y = self.feature_extract(x)
        x = self.resnet(x)
        return y, x


@ModelRegistry.register
def resnet18():
    model = ResNets(model_name='resnet18')
    return model


@ModelRegistry.register
def resnet50():
    model = ResNets(model_name='resnet50')
    return model