import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
from torch.optim import lr_scheduler
import torchvision.models as models
from torchsummary import summary
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from framebyframenetwork import Network
from fbfdataloader import pnp
import argparse

parser = argparse.ArgumentParser(description="Frame By Frame Video Classifier Network")
parser.add_argument("-dir", "--data_dir", help="Path of Folder containing in the data")
parser.add_argument("-ep", "--epoch", type=int, help="Number of Epochs for the network")
parser.add_argument("-lr", "--learning_rate", type=float, help="Learning Rate for the network")
parser.add_argument("-trgtd", "--target_directory", help = "Address of path where model dictionary is to be saved")


def train(data_dir, epoch, learning_rate, target_path):
    data_class = pnp(data_dir)
    net = Network().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(epoch):
        epoch_loss = 0.0
        epoch_total = 0.0
        epoch_correct = 0.0
        for data in data_class.return_batches("train"):
            x, y = data[0].to(device), data[1].to(device)
            net.zero_grad()
            y_pred =net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            y_pred = torch.argmax(y_pred, dim=1)
            epoch_total += y.shape[0]
            epoch_correct += torch.sum(y_pred == y)
        print("epoch {}: train_loss : {:.2f} || train_acc : {:.2f}".format(i + 1, epoch_loss,
                                                                           np.float(epoch_correct) / np.float(
                                                                               epoch_total)))

        with torch.no_grad():
            total = 0.0
            correct = 0.0
            val_loss = 0.0
            for data in data_class.return_batches("val"):
                x_val, y_val = data[0].to(device), data[1].to(device)
                y_val_p = net(x_val)
                loss = criterion(y_val_p, y_val)
                val_loss += loss
                y_val_p = torch.argmax(y_val_p, dim=1)
                for k in range(y_val.shape[0]):
                    if y_val_p[k] == y_val[k]:
                        correct += 1
                    total += 1
            print("         validation_loss : {:.2f} || validation Accuracy : {:.2f}".format(val_loss, np.float(
                correct) / np.float(total)))
    dictionary = {"model": net, "model_dict": net.state_dict(), "optim_dict": optimizer.state_dict(), "epoch": epoch}
    torch.save(dictionary, os.path.join(target_path,"model_dictionary.pt"))


if __name__ == "__main__":
    args = parser.parse_args()
    train(args.data_dir, args.epoch, args.learning_rate, args.target_directory)