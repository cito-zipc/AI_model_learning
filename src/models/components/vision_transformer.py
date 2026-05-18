import timm  # noqa: reportMissingImports
import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_tiny_patch4_32",
        num_classes: int = 100,
        image_size: int = 32,
        patch_size: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=image_size,
            patch_size=patch_size,
        )

        if freeze_backbone:
            for name, param in self.vit.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
