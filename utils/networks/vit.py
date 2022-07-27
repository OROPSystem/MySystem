import torch.nn as nn
import timm
from .util import set_parameter_requires_grad
from .registry import ModelRegistry


class VITs(nn.Module):
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


@ModelRegistry.register
def vit_base_patch16_224():
    model = VITs(model_name='vit_base_patch16_224')
    return model
