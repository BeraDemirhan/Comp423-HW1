import torch
import torch.nn as nn
import torchvision.models as models

class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self, num_actions):
        super(CILRS, self).__init__()
        self.num_actions = num_actions

        # Load the pre-trained ResNet18 model
        resnet18 = models.resnet18(pretrained=True)

        # Remove the fully connected layer at the end
        modules = list(resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)

        # Define the fully connected layers for predicting actions and speed
        self.fc1 = nn.Linear(resnet18.fc.in_features + 1, 512)
        self.fc2 = nn.Linear(512, num_actions)
        self.fc3 = nn.Linear(resnet18.fc.in_features + 1, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, img, speed, command):
        # Pass the image through the ResNet18 backbone
        features = self.resnet18(img)
        features = features.view(features.size(0), -1)

        # Concatenate speed and features
        speed = speed.unsqueeze(1)
        x = torch.cat((features, speed), dim=1)

        # Predict actions and speed
        x1 = nn.functional.relu(self.fc1(x))
        actions = self.fc2(x1)

        x2 = nn.functional.relu(self.fc3(x))
        speed_pred = self.fc4(x2)

        return actions, speed_pred.squeeze()