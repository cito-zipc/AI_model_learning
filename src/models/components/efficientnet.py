import timm  # noqa: reportMissingImports
import torch
import torch.nn as nn


class EfficientNet(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet_b2",
        num_classes: int = 100,
        image_size: int = 32,
        patch_size: int = 2,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.efficientnet = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=image_size,
            patch_size=patch_size,
        )

        if freeze_backbone:
            for name, param in self.efficientnet.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)
