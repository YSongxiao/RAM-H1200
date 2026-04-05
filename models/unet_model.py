""" Full assembly of the parts to form the complete network """
import torch

from models.unet_parts import *
# from nnunetv2.nets.SwinUMamba import ScalarGateBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock, UnetResBlock, UnetBasicBlock


class ScalarGateBlock(nn.Module):
    def __init__(self, enc_chans, prior_chans, out_chans):
        super().__init__()
        self.gate = nn.Conv2d(in_channels=enc_chans+prior_chans, out_channels=enc_chans+prior_chans, kernel_size=3, stride=1, padding=1)
        self.proj = nn.Conv2d(in_channels=enc_chans+prior_chans, out_channels=out_chans, kernel_size=1, stride=1)

    def forward(self, enc, prior):
        """
        enc:   (B, enc_chans, H, W)
        prior: (B, prior_chans, H, W)
        """

        # 1. concat image + prior
        x = torch.cat([enc, prior], dim=1)  # (B, enc+prior, H, W)

        # 2. compute gate logits
        g = self.gate(x)  # (B, enc+prior, H, W)

        # 3. collapse to scalar gate
        #    keep scalar semantics: one gate per spatial location
        g = g.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        g = torch.sigmoid(g)

        # 4. project prior to enc channel space
        p_proj = self.proj(x)  # (B, out_chans, H, W)

        # 5. gated residual fusion
        out = enc + g * p_proj

        return out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, emb_supervision=False, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.emb_supervision = emb_supervision

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.emb_supervision:
            return logits, x5
        else:
            return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class DualPathMergerUNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_overlap_classes, emb_supervision=False, bilinear=False):
        super(DualPathMergerUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_overlap_classes = n_overlap_classes
        self.bilinear = bilinear
        self.emb_supervision = emb_supervision

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

        self.up1_1 = (Up(512, 256 // factor, bilinear))
        self.up2_1 = (Up(256, 128 // factor, bilinear))
        self.up3_1 = (Up(128, 64 // factor, bilinear))
        self.up4_1 = (Up(64, 32, bilinear))
        self.outc_1 = (OutConv(32, n_classes))

        self.merger = UNet(n_channels=n_channels+n_classes+n_overlap_classes, n_classes=n_overlap_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        u_x4_1 = self.up1_1(x5, x4)
        u_x3_1 = self.up2_1(u_x4_1, x3)
        u_x2_1 = self.up3_1(u_x3_1, x2)
        u_x1_1 = self.up4_1(u_x2_1, x1)
        u_x4 = self.up1(x5, x4)
        u_x3 = self.up2(u_x4, x3)
        u_x2 = self.up3(u_x3, x2)
        u_x1 = self.up4(u_x2, x1)
        overlap = self.outc_1(u_x1_1)
        coarse_overall = self.outc(u_x1)
        out_1 = torch.concat([coarse_overall, overlap], dim=1)
        delta_overall = self.merger(torch.concat([x, coarse_overall, overlap], dim=1))
        refined_overall = coarse_overall + delta_overall
        if self.emb_supervision:
            return out_1, refined_overall, x5
        else:
            return out_1, refined_overall


class GatedUNet(nn.Module):
    def __init__(self, n_channels=1, prior_chans=2, n_classes=13, feat_size=[32, 64, 128, 256, 512], spatial_dims=2, norm_name="instance", emb_supervision=False, bilinear=False):
        super(GatedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_prior_channels = prior_chans
        self.bilinear = bilinear
        self.emb_supervision = emb_supervision
        self.feat_size = feat_size

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))

        self.prior_encoder1 = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=self.n_prior_channels,
            out_channels=self.feat_size[0] // 2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name
        )

        self.gate1 = ScalarGateBlock(
            enc_chans=self.feat_size[0],
            prior_chans=self.feat_size[0] // 2,
            out_chans=self.feat_size[0]
        )

        self.prior_encoder2 = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0] // 2,
            out_channels=self.feat_size[1] // 2,
            kernel_size=3,
            stride=2,
            norm_name=norm_name
        )

        self.gate2 = ScalarGateBlock(
            enc_chans=self.feat_size[1],
            prior_chans=self.feat_size[1] // 2,
            out_chans=self.feat_size[1]
        )

        self.prior_encoder3 = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1] // 2,
            out_channels=self.feat_size[2] // 2,
            kernel_size=3,
            stride=2,
            norm_name=norm_name
        )

        self.gate3 = ScalarGateBlock(
            enc_chans=self.feat_size[2],
            prior_chans=self.feat_size[2] // 2,
            out_chans=self.feat_size[2]
        )

        self.prior_encoder4 = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2] // 2,
            out_channels=self.feat_size[3] // 2,
            kernel_size=3,
            stride=2,
            norm_name=norm_name
        )

        self.gate4 = ScalarGateBlock(
            enc_chans=self.feat_size[3],
            prior_chans=self.feat_size[3] // 2,
            out_chans=self.feat_size[3]
        )

        self.prior_encoder5 = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3] // 2,
            out_channels=self.feat_size[4] // 2,
            kernel_size=3,
            stride=2,
            norm_name=norm_name
        )

        self.gate5 = ScalarGateBlock(
            enc_chans=self.feat_size[4],
            prior_chans=self.feat_size[4] // 2,
            out_chans=self.feat_size[4]
        )

        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x_in = x[:, 0:1, ...]
        p_in = x[:, 1:, ...]
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        p_enc1 = self.prior_encoder1(p_in)
        p_enc2 = self.prior_encoder2(p_enc1)
        p_enc3 = self.prior_encoder3(p_enc2)
        p_enc4 = self.prior_encoder4(p_enc3)
        p_enc5 = self.prior_encoder5(p_enc4)

        enc1 = self.gate1(x1, p_enc1)
        enc2 = self.gate2(x2, p_enc2)
        enc3 = self.gate3(x3, p_enc3)
        enc4 = self.gate4(x4, p_enc4)
        enc5 = self.gate5(x5, p_enc5)

        dec4 = self.up1(enc5, enc4)
        dec3 = self.up2(dec4, enc3)
        dec2 = self.up3(dec3, enc2)
        dec1 = self.up4(dec2, enc1)
        out = self.outc(dec1)
        return out


class UNetMid(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetMid, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNetSmall(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetSmall, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


# if __name__ == '__main__':
#     net = GatedUNet(n_channels=1, prior_chans=3, n_classes=1)
#     dummy_inp = torch.zeros([2, 4, 256, 256])
#     print(net(dummy_inp).shape)