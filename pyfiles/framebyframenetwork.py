import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
from torch.optim import lr_scheduler
import torchvision.models as models

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.resnet = models.resnet18(pretrained=True)
        k = 1
        for param in self.resnet.parameters():
            k += 1
            if k < 56:
                param.requires_grad = False

        self.dense1 = nn.Linear(in_features=1000, out_features=512)
        self.relu = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(in_features=512, out_features=256)
        self.dense3 = nn.Linear(in_features=256, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.out(x)
        return F.softmax(x)