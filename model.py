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



# ## CelebA_faster
# import torch
# from torch import nn

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()

#         self.feature_extractor = nn.Sequential(
#             self._convGroup(1, 64),
#             self._convGroup(64, 128),
#             self._convGroup(128, 256)
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )

#     def _convGroup(self, in_features, out_features):
#         return nn.Sequential(
#             nn.Conv2d(in_features, out_features, 3),
#             nn.BatchNorm2d(out_features),
#             nn.ReLU(True),
#             nn.MaxPool2d(2),
#         )

#     def forward(self, x):
#         mean = x.mean()
#         std = x.std()
#         x_norm = (x - mean) / std
        
#         out = self.feature_extractor(x_norm)
#         out = out.view(-1, 256)
#         out = self.classifier(out)
#         return out