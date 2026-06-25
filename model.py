import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(3,3)

        self.conv2 = nn.Conv2d(16,32,3,1,1)
        self.bn2 = nn.BatchNorm2d(32)
        #fc layer
        self.fc1 = nn.Linear(6272,84)
        self.fc2 = nn.Linear(84,64)
        self.out = nn.Linear(64,1)
         
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x,start_dim=1)
        #fc layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

      
