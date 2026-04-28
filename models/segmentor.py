import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class StrokeSegmentor(nn.Module):
    """
    ResNet-34 인코더 기반 U-Net 허혈성 병변 분할 모델.
    입력: (B, 3, H, W)  출력: (B, 1, H, W) 로짓
    """

    def __init__(self, encoder_name: str = "resnet34",
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

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """이진 마스크 반환 (B, 1, H, W) — 0 또는 1"""
        logits = self.forward(x)
        return (torch.sigmoid(logits) > threshold).float()
