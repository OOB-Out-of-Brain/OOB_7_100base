import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class StrokeSegmentor(nn.Module):
    """
    설정된 encoder 기반 U-Net 뇌 출혈 병변 분할 모델.
    기본 encoder: efficientnet-b4 (config.yaml 기준).
    입력: (B, 3, H, W)  출력: (B, 1, H, W) 로짓
    """

    def __init__(self, encoder_name: str = "efficientnet-b4",
                 encoder_weights: str = "imagenet"):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """확률 맵 반환 (B, 1, H, W) — 0~1"""
        return torch.sigmoid(self.forward(x))

    def freeze_encoder(self):
        for p in self.unet.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.unet.encoder.parameters():
            p.requires_grad = True

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """이진 마스크 반환 (B, 1, H, W) — 0 또는 1"""
        logits = self.forward(x)
        return (torch.sigmoid(logits) > threshold).float()
