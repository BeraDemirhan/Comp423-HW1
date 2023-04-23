import torch.nn as nn


class AffordancePredictor(nn.Module):
    """Affordance prediction network that takes images as input"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, img):
        x = nn.functional.relu(self.conv1(img))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = x.view(-1, 256 * 3 * 3)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # Apply sigmoid to traffic light state affordance
        x[:, 3] = nn.functional.sigmoid(x[:, 3])
        return x
