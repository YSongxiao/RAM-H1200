import torch

from nnunetv2.nets.SwinUMamba import SwinUMamba, GatedSwinUMamba, DualPathMergerSwinUMamba, load_pretrained_ckpt
import torch.nn as nn
import torch
from models.UnetPlusPlus import UnetPlusPlus


class RefinedSwinUMamba(nn.Module):
    def __init__(self, base_model, in_chans, out_chans):
        super().__init__()
        self.base_model = base_model
        self.refined_model = UnetPlusPlus(spatial_dims=2, in_channels=in_chans+out_chans, out_channels=out_chans)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()

    def forward(self, x):
        with torch.no_grad():
            corase_out = self.base_model(x)
        x_in = torch.cat([x, corase_out], dim=1)
        residual = self.refined_model(x_in)
        out = corase_out + residual
        return out



def get_SwinUMamba(in_channels, num_classes, feat_size=[48, 96, 192, 384, 768], hidden_size=768, deep_supervision=False):
    model = SwinUMamba(
        in_chans=in_channels,
        out_chans=num_classes,
        feat_size=feat_size,
        deep_supervision=deep_supervision,
        hidden_size=hidden_size,
    )
    return model


def get_DPMSwinUMamba(in_channels, num_classes, num_overlap_classes, feat_size=[48, 96, 192, 384, 768], hidden_size=768, deep_supervision=False):
    model = DualPathMergerSwinUMamba(
        in_chans=in_channels,
        out_chans=num_classes,
        overlap_out_chans=num_overlap_classes,
        feat_size=feat_size,
        deep_supervision=deep_supervision,
        hidden_size=hidden_size,
    )
    return model


def get_PriorGatedSwinUMamba(in_channels, prior_in_channels, num_classes, feat_size=[48, 96, 192, 384, 768], hidden_size=768, deep_supervision=False):
    model = GatedSwinUMamba(
        in_chans=in_channels,
        prior_chans=prior_in_channels,
        out_chans=num_classes,
        feat_size=feat_size,
        deep_supervision=deep_supervision,
        hidden_size=hidden_size,
    )
    return model


def get_RefinedSwinUMamba(in_channels, num_classes, base_ckpt):
    base_model = get_SwinUMamba(in_channels, num_classes)
    base_model.load_state_dict(torch.load(base_ckpt)["model"])
    model = RefinedSwinUMamba(
        in_chans=in_channels,
        out_chans=num_classes,
        base_model=base_model
    )
    return model
