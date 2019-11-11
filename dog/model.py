import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(32*23*23, 256)
        self.fc2 = nn.Linear(256, 120)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        
        # print(x.shape)
        x = x.view(-1, 32*23*23)
        
        # print(len(x))
        x = self.fc1(x)
        x = F.relu(x)

        out = self.fc2(x)

        return out
