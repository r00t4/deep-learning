import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        out = self.fc3(x)
        return torch.sigmoid(out)
