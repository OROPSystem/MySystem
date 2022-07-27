import torch
import timm


def set_parameter_requires_grad(net, feature_extract):
    if feature_extract:
        for param in net.parameters():
            param.requires_grad = False
