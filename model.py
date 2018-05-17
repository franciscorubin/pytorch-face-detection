import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.feature_extractor = nn.Sequential(
            self._convGroup(1, 16),
            self._convGroup(16, 32),
            self._convGroup(32, 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
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
        mean = x.mean()
        std = x.std()
        x_norm = (x - mean) / std
        
        out = self.feature_extractor(x_norm)
        out = out.view(-1, 64)
        out = self.classifier(out)
        return out

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

        self.feature_extractor = nn.Sequential(
            self._convGroup(1, 8),
            self._convGroup(8, 16),
            self._convGroup(16, 32)
        )
        self.classifier_in = 32
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_in, 1),
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
        mean = x.mean()
        std = x.std()
        x_norm = (x - mean) / std
        
        out = self.feature_extractor(x_norm)
        out = out.view(-1, self.classifier_in)
        out = self.classifier(out)
        return out

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()

        self.feature_extractor = nn.Sequential(
            self._convGroup(1, 16),
            self._convGroup(16, 32),
            self._convGroup(32, 64)
        )
        self.classifier_in = 64
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_in, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 1),
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
        mean = x.mean()
        std = x.std()
        x_norm = (x - mean) / std
        
        out = self.feature_extractor(x_norm)
        out = out.view(-1, self.classifier_in)
        out = self.classifier(out)
        return out


class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()

        self.feature_extractor = nn.Sequential(
            self._convGroup(1, 64)
        )
        self.classifier_in = 64*11*11
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_in, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 1),
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
        mean = x.mean()
        std = x.std()
        x_norm = (x - mean) / std
        
        out = self.feature_extractor(x_norm)
        out = out.view(-1, self.classifier_in)
        out = self.classifier(out)
        return out


class Model5(nn.Module):
    def __init__(self):
        super(Model5, self).__init__()

        self.feature_extractor = nn.Sequential(
            self._convGroup(1, 64, pool_factor=4)
        )
        self.classifier_in = 64*5*5
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_in, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def _convGroup(self, in_features, out_features, pool_factor=2):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, 3),
            nn.BatchNorm2d(out_features),
            nn.ReLU(True),
            nn.MaxPool2d(pool_factor),
        )

    def forward(self, x):
        mean = x.mean()
        std = x.std()
        x_norm = (x - mean) / std
        
        out = self.feature_extractor(x_norm)
        out = out.view(-1, self.classifier_in)
        out = self.classifier(out)
        return out


class Model6(nn.Module):
    def __init__(self):
        super(Model6, self).__init__()

        self.feature_extractor = nn.Sequential(
            self._convGroup(1, 32),
            self._convGroup(32, 64),
            self._convGroup(64, 128)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
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
        mean = x.mean()
        std = x.std()
        x_norm = (x - mean) / std
        
        out = self.feature_extractor(x_norm)
        out = out.view(-1, 128)
        out = self.classifier(out)
        return out