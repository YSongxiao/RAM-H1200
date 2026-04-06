from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import LayerNorm2d
except ImportError:  # pragma: no cover
    from timm.models.layers import LayerNorm2d


_OFFICIAL_SEGMENTATION_BACKBONE_KWARGS = {
    # Official configs from NVlabs/MambaVision semantic_segmentation/configs/mamba_vision/*.py
    "mamba_vision_T": dict(
        depths=(1, 3, 8, 4),
        num_heads=(2, 4, 8, 16),
        window_size=(8, 8, 64, 32),
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.3,
    ),
    "mamba_vision_S": dict(
        depths=(3, 3, 7, 5),
        num_heads=(2, 4, 8, 16),
        window_size=(8, 8, 160, 56),
        dim=96,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.7,
    ),
}


def _strip_common_prefixes(key):
    prefixes = ("module.", "model.", "state_dict.", "feature_extractor.", "backbone.", "encoder.")
    stripped = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                changed = True
    return stripped


def _build_norm(norm, channels):
    norm = str(norm).lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "gn":
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm type: {norm}")


def _resize(x, size, align_corners=False):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


class _ConvModule(nn.Sequential):
    def __init__(self, in_chans, out_chans, kernel_size, padding=0, norm="bn", act=True):
        layers = [
            nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding, bias=False),
            _build_norm(norm, out_chans),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class _PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_chans, channels, norm="bn", align_corners=False):
        self.pool_scales = tuple(pool_scales)
        self.align_corners = bool(align_corners)
        super().__init__(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    _ConvModule(in_chans, channels, kernel_size=1, padding=0, norm=norm),
                )
                for scale in self.pool_scales
            ]
        )

    def forward(self, x):
        ppm_outs = []
        target_size = x.shape[-2:]
        for ppm in self:
            ppm_out = ppm(x)
            ppm_out = _resize(ppm_out, size=target_size, align_corners=self.align_corners)
            ppm_outs.append(ppm_out)
        return ppm_outs


class _UPerHead(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        num_classes,
        pool_scales=(1, 2, 3, 6),
        dropout=0.1,
        norm="bn",
        align_corners=False,
    ):
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.channels = int(channels)
        self.align_corners = bool(align_corners)
        self.psp_modules = _PPM(
            pool_scales=pool_scales,
            in_chans=self.in_channels[-1],
            channels=self.channels,
            norm=norm,
            align_corners=self.align_corners,
        )
        self.bottleneck = _ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm=norm,
        )
        self.lateral_convs = nn.ModuleList(
            [
                _ConvModule(in_ch, self.channels, kernel_size=1, padding=0, norm=norm)
                for in_ch in self.in_channels[:-1]
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                _ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm=norm)
                for _ in self.in_channels[:-1]
            ]
        )
        self.fpn_bottleneck = _ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm=norm,
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)

    def _forward_feature(self, inputs):
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))

        used_backbone_levels = len(laterals)
        for level_idx in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[level_idx - 1].shape[2:]
            laterals[level_idx - 1] = laterals[level_idx - 1] + _resize(
                laterals[level_idx],
                size=prev_shape,
                align_corners=self.align_corners,
            )

        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        fpn_outs.append(laterals[-1])

        for level_idx in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[level_idx] = _resize(
                fpn_outs[level_idx],
                size=fpn_outs[0].shape[2:],
                align_corners=self.align_corners,
            )

        return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.dropout(output)
        return self.cls_seg(output)


class _FCNHead(nn.Module):
    def __init__(self, in_chans, channels, num_classes, dropout=0.1):
        super().__init__()
        self.block = _ConvModule(in_chans, channels, kernel_size=3, padding=1, norm="bn")
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.block(x)
        x = self.dropout(x)
        return self.classifier(x)


class _MambaVisionFeatureExtractor(nn.Module):
    def __init__(self, variant, in_chans, image_size=224, pretrained=False, out_indices=(0, 1, 2, 3), norm_layer="ln2d"):
        super().__init__()

        try:
            from mambavision import create_model as create_mambavision_model
        except ImportError as exc:
            raise ImportError(
                "MambaVision segmentation is selected but the 'mambavision' package is not available "
                "in the current environment."
            ) from exc

        backbone_kwargs = dict(_OFFICIAL_SEGMENTATION_BACKBONE_KWARGS.get(variant, {}))
        self.backbone = create_mambavision_model(
            variant,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            resolution=image_size,
            **backbone_kwargs,
        )

        stem_channels = self.backbone.patch_embed.conv_down[3].out_channels
        self.out_indices = tuple(out_indices)
        self.dims = [stem_channels * (2 ** idx) for idx in range(len(self.backbone.levels))]
        self.out_channels = [self.dims[idx] for idx in self.out_indices]
        self.channel_first = True

        norm_layers = {
            "ln2d": LayerNorm2d,
            "bn": nn.BatchNorm2d,
        }
        norm_cls = norm_layers.get(str(norm_layer).lower())
        if norm_cls is None:
            raise ValueError(f"Unsupported backbone norm layer: {norm_layer}")

        for idx in self.out_indices:
            setattr(self, f"outnorm{idx}", norm_cls(self.dims[idx]))

        if hasattr(self.backbone, "norm"):
            del self.backbone.norm
        if hasattr(self.backbone, "head"):
            del self.backbone.head

    @staticmethod
    def _forward_level(level, x):
        _, _, height, width = x.shape
        padded_height = height
        padded_width = width
        pad_right = 0
        pad_bottom = 0

        if level.transformer_block:
            pad_right = (level.window_size - width % level.window_size) % level.window_size
            pad_bottom = (level.window_size - height % level.window_size) % level.window_size
            if pad_right > 0 or pad_bottom > 0:
                x = F.pad(x, (0, pad_right, 0, pad_bottom))
                _, _, padded_height, padded_width = x.shape
            from mambavision.models.mamba_vision import window_partition, window_reverse

            x = window_partition(x, level.window_size)

        for block in level.blocks:
            x = block(x)

        if level.transformer_block:
            from mambavision.models.mamba_vision import window_reverse

            x = window_reverse(x, level.window_size, padded_height, padded_width)
            if pad_right > 0 or pad_bottom > 0:
                x = x[:, :, :height, :width].contiguous()

        stage_output = x
        next_x = stage_output if level.downsample is None else level.downsample(stage_output)
        return next_x, stage_output

    def forward(self, x):
        x = self.backbone.patch_embed(x)
        outputs = []
        for idx, level in enumerate(self.backbone.levels):
            x, stage_output = self._forward_level(level, x)
            if idx in self.out_indices:
                norm = getattr(self, f"outnorm{idx}")
                outputs.append(norm(stage_output).contiguous())
        return outputs

    def load_pretrained_weights(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("model", checkpoint)

        backbone_state = self.backbone.state_dict()
        filtered_state = {}
        skipped_keys = []

        for key, value in state_dict.items():
            candidate_keys = [key]
            stripped_key = _strip_common_prefixes(key)
            if stripped_key != key:
                candidate_keys.append(stripped_key)

            matched = False
            for candidate in candidate_keys:
                if candidate in backbone_state and backbone_state[candidate].shape == value.shape:
                    filtered_state[candidate] = value
                    matched = True
                    break
            if not matched:
                skipped_keys.append(key)

        load_result = self.backbone.load_state_dict(filtered_state, strict=False)
        print(
            f"MambaVision pretrained load: loaded {len(filtered_state)} tensors from {checkpoint_path}. "
            f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}, "
            f"skipped_shape_or_name={len(skipped_keys)}"
        )
        return load_result


class MambaVisionSeg(nn.Module):
    def __init__(self, variant, in_chans, num_classes, image_size=224, pretrained=False, use_auxiliary=False):
        super().__init__()
        self.feature_extractor = _MambaVisionFeatureExtractor(
            variant=variant,
            in_chans=in_chans,
            image_size=image_size,
            pretrained=pretrained,
        )
        self.backbone = self.feature_extractor.backbone
        self.out_channels = list(self.feature_extractor.out_channels)
        self.use_auxiliary = bool(use_auxiliary)
        self.decode_head = _UPerHead(
            in_channels=self.out_channels,
            channels=512,
            num_classes=num_classes,
            pool_scales=(1, 2, 3, 6),
            dropout=0.1,
        )
        self.auxiliary_head = _FCNHead(
            in_chans=self.out_channels[2],
            channels=256,
            num_classes=num_classes,
            dropout=0.1,
        )
        if not self.use_auxiliary:
            self._freeze_auxiliary_head()

    def _freeze_auxiliary_head(self):
        for param in self.auxiliary_head.parameters():
            param.requires_grad = False

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.feature_extractor(x)

        logits = self.decode_head(features)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        if not self.training or not self.use_auxiliary:
            return logits

        aux_logits = self.auxiliary_head(features[2])
        aux_logits = F.interpolate(aux_logits, size=input_size, mode="bilinear", align_corners=False)
        return logits, aux_logits

    def load_pretrained_weights(self, checkpoint_path):
        return self.feature_extractor.load_pretrained_weights(checkpoint_path)


def get_MambaVisionSeg(variant, in_chans, num_classes, image_size=224, pretrained=False, use_auxiliary=False):
    return MambaVisionSeg(
        variant=variant,
        in_chans=in_chans,
        num_classes=num_classes,
        image_size=image_size,
        pretrained=pretrained,
        use_auxiliary=use_auxiliary,
    )
