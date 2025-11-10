import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class _PreprocessMixin:
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        if x.max() > 1.0:
            x = x / 255.0
        x = self.resize(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, feature_dim: int = 512, in_channels: int = 3, device: str = "cuda", freeze: bool = False):
        super().__init__()
        self.device = torch.device(device)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, feature_dim, bias=True)
        self.feature_dim = feature_dim
        self.to(self.device)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        if x.max() > 1.0:
            x = x / 255.0
        x = x.to(self.device)
        feats = self.cnn(x)
        feats = self.pool(feats).flatten(1)
        feats = self.proj(feats)
        feats = F.normalize(feats, dim=1)
        return feats


class ResnetEncoder(nn.Module, _PreprocessMixin):
    def __init__(self, pretrained: bool = True, feature_dim: int = 512, device: str = "cuda", freeze: bool = True):
        super().__init__()
        self.device = torch.device(device)
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone.to(self.device)
        self.proj = nn.Identity() if feature_dim == 512 else nn.Linear(512, feature_dim, bias=True).to(self.device)
        self.feature_dim = feature_dim
        self.resize = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.proj.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x).to(self.device)
        feats = self.backbone(x)
        feats = self.proj(feats)
        feats = F.normalize(feats, dim=1)
        return feats


class Encoder(nn.Module):
    def __init__(self, encoder_type: str = "resnet", feature_dim: int = 512, device: str = "cuda", pretrained: bool = True, freeze: bool = True):
        super().__init__()
        encoder_type = encoder_type.lower()
        if encoder_type not in ("resnet", "cnn"):
            raise ValueError(f"encoder_type must be 'resnet' or 'cnn', got {encoder_type}")
        if encoder_type == "resnet":
            self.encoder = ResnetEncoder(pretrained=pretrained, feature_dim=feature_dim, device=device, freeze=freeze)
        else:
            self.encoder = CNNEncoder(feature_dim=feature_dim, device=device, freeze=freeze)
        self.feature_dim = feature_dim
        self.encoder_type = encoder_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)