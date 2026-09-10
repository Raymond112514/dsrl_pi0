import torch
import torch.nn as nn
import torch.nn.functional as F


class SVMClassifier(nn.Module):
    def __init__(self, obs_dim=520, action_dim=32, feature_dim=512, hidden_dim=256, num_layers=2, lr=1e-5, clip_range=20):
        super().__init__()
        self.clip_range = clip_range
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
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
        self.proj = nn.Linear(256, feature_dim)
        layers = [nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.svm_classifier = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def encode(self, x):
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        if x.max() > 1.0:
            x = x / 255.0
        feats = self.pool(self.cnn(x)).flatten(1)
        return F.normalize(self.proj(feats), dim=1)

    def forward(self, state, pixel, action):
        x = torch.cat([state, self.encode(pixel), action], dim=1)
        return self.svm_classifier(x)

    def update(self, state, pixel, action, label):
        self.train()
        loss = F.binary_cross_entropy_with_logits(self.forward(state, pixel, action).squeeze(-1), label.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def loss(self, state, pixel, action, label):
        self.eval()
        return F.binary_cross_entropy_with_logits(self.forward(state, pixel, action).squeeze(-1), label.float())

    def rewards(self, state, pixel, action):
        self.eval()
        return torch.clamp(self.forward(state, pixel, action).squeeze(-1), -self.clip_range, self.clip_range)
