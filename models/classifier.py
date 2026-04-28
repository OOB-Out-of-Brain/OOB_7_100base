import torch
import torch.nn as nn
import timm


class StrokeClassifier(nn.Module):
    """EfficientNet 기반 이진 분류 모델. 출력: normal(0) / hemorrhagic(1)"""

    def __init__(self, model_name: str = "efficientnet_b4",
                 num_classes: int = 2, pretrained: bool = True,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def predict(self, x: torch.Tensor,
                hemorrhagic_idx: int = 1,
                threshold: float = 0.5):
        """softmax 확률과 class index 반환. threshold < 0.5 이면 출혈에 더 민감."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        if threshold != 0.5 and probs.shape[1] > hemorrhagic_idx:
            pred = (probs[:, hemorrhagic_idx] >= threshold).long()
        else:
            pred = probs.argmax(dim=1)
        return pred, probs

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
