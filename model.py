import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.feature_extractor = nn.Sequential(
            self._convGroup(1, 128),
            self._convGroup(128, 256),
            self._convGroup(256, 512)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _convGroup(self, in_features, out_features):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, 3),
            nn.BatchNorm2d(out_features),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(-1, 512)
        out = self.classifier(out)
        return out