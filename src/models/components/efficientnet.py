import timm  # noqa: reportMissingImports
import torch
import torch.nn as nn


class EfficientNet(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet_b2",
        num_classes: int = 100,
        input_size: tuple = (32, 32),
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.input_size = input_size

        self.efficientnet = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.efficientnet.conv_stem.stride = (1, 1)

        if freeze_backbone:
            for name, param in self.efficientnet.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)

    def set_input_size(self, size: tuple) -> None:
        self.input_size = size
