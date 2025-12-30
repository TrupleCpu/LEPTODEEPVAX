import torch.nn as nn

class LeptoNetV2(nn.Module):
    def __init__(self, input_size):
        super(LeptoNetV2, self).__init__()
        # Feature Encoding Layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Priority Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)