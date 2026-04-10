import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralOrdinalModel(nn.Module):
    """
    CORAL-style ordinal regression wrapper.

    The wrapped backbone must output one scalar latent severity score per sample.
    This module converts that scalar into K-1 ordered threshold logits:

        logit_k = latent_score - cutpoint_k

    with cutpoints constrained to be monotonically increasing.
    """

    def __init__(self, backbone, num_thresholds):
        super().__init__()
        self.backbone = backbone
        self.num_thresholds = int(num_thresholds)
        if self.num_thresholds < 1:
            raise ValueError(f"num_thresholds must be >= 1, got {num_thresholds}")

        self.first_cutpoint = nn.Parameter(torch.tensor(0.0))
        if self.num_thresholds > 1:
            self.cutpoint_deltas = nn.Parameter(torch.zeros(self.num_thresholds - 1))
        else:
            self.register_parameter("cutpoint_deltas", None)

    def ordered_cutpoints(self):
        if self.num_thresholds == 1:
            return self.first_cutpoint.view(1)
        positive_deltas = F.softplus(self.cutpoint_deltas)
        return torch.cat(
            [
                self.first_cutpoint.view(1),
                self.first_cutpoint + torch.cumsum(positive_deltas, dim=0),
            ],
            dim=0,
        )

    def forward(self, x):
        latent_score = self.backbone(x)
        if not isinstance(latent_score, torch.Tensor):
            latent_score = latent_score[0]
        if latent_score.ndim == 1:
            latent_score = latent_score.unsqueeze(1)
        if latent_score.ndim != 2 or latent_score.shape[1] != 1:
            raise RuntimeError(
                "CORAL backbone must output shape (B, 1), "
                f"got {tuple(latent_score.shape)}"
            )
        cutpoints = self.ordered_cutpoints().to(device=latent_score.device, dtype=latent_score.dtype)
        return latent_score - cutpoints.view(1, -1)
