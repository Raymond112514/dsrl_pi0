import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder

class Classifier(nn.Module):
    def __init__(self,
                 encoder_type: str = "resnet",
                 obs_dim=520,
                 action_dim=32,
                 hidden_dim=256,
                 num_layers=2,
                 lr=1e-5,
                 classifier_type="default"):
        super(Classifier, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.classifier_type = classifier_type
        self.encoder = Encoder(encoder_type=encoder_type)
        assert classifier_type in ["default", "half", "prob", "log_prob", "exp_logit", "quotient"]

        self.net = nn.ModuleList()
        self.net.append(nn.Linear(obs_dim + action_dim, hidden_dim))
        self.net.append(nn.ReLU())
        for _ in range(num_layers):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_dim, 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, pixel, action):
        embedding = self.encoder(pixel)
        x = torch.cat([state, embedding, action], dim=1)
        for layer in self.net:
            x = layer(x)
        return x

    def update(self, state, pixel, action, label):
        self.train()
        logits = self.forward(state, pixel, action).squeeze(-1)
        labels = label.float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def loss(self, state, pixel, action, label):
        self.eval()
        logits = self.forward(state, pixel, action).squeeze(-1)
        labels = label.float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.train()
        return loss

    def rewards(self, state, pixel, action):
        self.eval()
        logits = self.forward(state, pixel, action).squeeze(-1)
        logits = torch.clamp(logits, -20, 20)
        if self.classifier_type == "log_prob":
            logits = -F.softplus(-logits)
            return logits
        elif self.classifier_type == "prob":
            logits = torch.sigmoid(logits)
            return torch.clamp(logits, -20, 20)
        elif self.classifier_type == "quotient":
            logits = torch.sigmoid(logits)
            return torch.clamp(logits / (1 - logits + 1e-6), -20, 20)
        elif self.classifier_type == "half":
            p = torch.sigmoid(logits)
            return torch.clamp(-torch.log(1 - p + 1e-6), -20, 20)
        else:
            return logits