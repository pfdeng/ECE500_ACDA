import torch.nn as nn
import torch.nn.functional as F
from ACDA.Conv_re import *

class LeNet_ACDA(nn.Module):
    def __init__(self, out_size):
        super(LeNet_ACDA, self).__init__()
        self.conv1 = ACDA(in_channels=3, out_channels=3, kernel_size=5)
        self.conv2 = ACDA(3, 3, 5)
        self.fc1 = nn.Linear(192, 100)
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, out_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Backward-compatible alias (the class was previously named ACDA_modified).
ACDA_modified = LeNet_ACDA

