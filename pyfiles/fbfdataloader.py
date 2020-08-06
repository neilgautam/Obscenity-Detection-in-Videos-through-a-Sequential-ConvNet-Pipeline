import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

class pnp:
    def __init__(self, data_dir):
        self.transforms = {
            "train": transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "test": transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        self.data_dir = data_dir
        self.batch_size = 32
        self.images = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.transforms[x]) for x in
                       ["train", "val", "test"]}
        self.dataloader = {x: torch.utils.data.DataLoader(self.images[x], batch_size=self.batch_size, shuffle=True) for
                           x in ["val", "train", "test"]}

    def return_len(self, x):
        return len(self.images[x])

    def return_dataloader(self, x):
        return self.dataloader

    def return_batches(self, x):
        for i in range(0, len(self.images[x]), self.batch_size):
            inputs, classes = next(iter(self.dataloader[x]))
            yield (inputs, classes)

    def return_classes(self, x):
        return self.images[x].classes
