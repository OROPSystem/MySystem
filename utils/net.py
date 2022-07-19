import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models
import timm
from torchvision.models.vgg import vgg19, vgg11, vgg13, vgg16, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def set_parameter_requires_grad(net, feature_extract):
    if feature_extract:
        for param in net.parameters():
            param.requires_grad = False


class ResNets(nn.Module):
    def __init__(self, model_name='resnet18', num_layers=12, num_classes=5, feature_extract=False, use_pretrained=True):
        super(ResNets, self).__init__()
        self.resnet = eval(model_name)(pretrained=use_pretrained)
        set_parameter_requires_grad(self.resnet, feature_extract)
        modules = list(self.resnet.children())
        self.feature_extract = nn.Sequential(*modules[0:-1], nn.Flatten())
        num_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_in_features, num_classes)

    def forward(self, x):
        y = self.feature_extract(x)
        x = self.resnet(x)
        return y, x


class VITs(nn.Module):
    # 'vit_base_patch16_224', 'vit_base_patch16_224_in21k', 'vit_base_patch16_384',
    # 'vit_base_patch32_224_in21k', 'vit_base_patch32_384', 'vit_base_resnet50_224_in21k',
    # 'vit_base_resnet50_384', 'vit_deit_base_distilled_patch16_224', 'vit_deit_base_distilled_patch16_384',
    # 'vit_deit_base_patch16_224', 'vit_deit_base_patch16_384', 'vit_deit_small_distilled_patch16_224',
    # 'vit_deit_small_patch16_224', 'vit_deit_tiny_distilled_patch16_224', 'vit_deit_tiny_patch16_224',
    # 'vit_large_patch16_224', 'vit_large_patch16_224_in21k', 'vit_large_patch16_384', 'vit_large_patch32_224_in21k',
    # 'vit_large_patch32_384', 'vit_small_patch16_224',
    def __init__(self, model_name='vit_base_patch16_224', num_layers=12, num_classes=5, feature_extract=False, use_pretrained=True):
        super(VITs, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=use_pretrained)
        set_parameter_requires_grad(self.vit, feature_extract)
        self.vit.blocks = self.vit.blocks[:num_layers]

        num_in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(num_in_features, num_classes)

    def forward_feature(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit.pos_drop(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        return x[:, 0]

    def forward(self, x):
        y = self.forward_feature(x)
        x = self.vit(x)
        # x = self.vit(x)
        return y, x



class MLPMixers(nn.Module):
    # 'mixer_b16_224', 'mixer_b16_224_in21k', 'mixer_l16_224', 'mixer_l16_224_in21k'
    def __init__(self, model_name='mixer_b16_224', num_layers=12, num_classes=6, feature_extract=False, use_pretrained=True):
        super(MLPMixers, self).__init__()
        self.mlp_mixer = timm.create_model(model_name, pretrained=use_pretrained)
        set_parameter_requires_grad(self.mlp_mixer, feature_extract)
        num_in_features = self.mlp_mixer.head.in_features
        self.mlp_mixer.head = nn.Linear(num_in_features, num_classes)

    def forward(self, x):
        x = self.mlp_mixer(x)
        return x
