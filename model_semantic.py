import torch
import torch.nn as nn

class SemanticHead(nn.Module):
    def __init__(self, input_dim=384, hidden=128, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)

class LayoutClassifier(nn.Module):
    def __init__(self, input_dim, hidden=64, num_classes=5):
        super().__init__()
        self.initial_fc = nn.Linear(input_dim, hidden)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, num_classes))

    def forward(self, x):
        x = self.initial_fc(x)
        return self.fc(x)

