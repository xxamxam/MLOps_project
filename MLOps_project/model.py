import torch.nn as nn


# MODEL FOR TRAIN
class CNN_new(nn.Module):
    def __init__(self, k=1):
        super(CNN_new, self).__init__()
        # your code here: define layers
        self.layers = nn.Sequential(
            nn.Conv2d(
                1, 6 * k, kernel_size=5, stride=1, padding=0
            ),  # padding = 1 чтобы не поменялся размер на выходе
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 28 -> 14
            nn.Conv2d(
                6 * k, 16 * k, kernel_size=5, stride=1, padding=0
            ),  # padding = 1 чтобы не поменялся размер на выходе
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(1, -1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * k, 10),
        )

    def forward(self, x):
        # your code here: transform x using layers
        x = self.layers(x)
        x = self.fc(x)
        return x
