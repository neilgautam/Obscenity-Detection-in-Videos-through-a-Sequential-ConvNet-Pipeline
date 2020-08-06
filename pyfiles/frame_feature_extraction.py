import torch
import torch.nn as nn
from torch import optim as optim
from torchvision import models,datasets,transforms
import numpy as np
import pandas as pd
import cv2
import time
import torch.nn.functional as F
import torchsummary
import os
import glob
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
from framebyframenetwork import Network
import argparse

parser = argparse.ArgumentParser(description= "Images features extraction using pre trained ResNet-18 fine tuned on the data with learning rate of 0.0005")

parser.add_argument("-sm", "--saved_model", "ResNet-18(trained) saved model ")
parser.add_argument("-p", "--data_path", "Directory of the folder that contains extracted frames")
parser.add_argument("-trg", "--target_path", "Address of folder to save the features")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(saved_model_add, device= device):
    net = Network().to(device)
    weights = torch.load(saved_model_add, map_location=device)
    net.load_state_dict(weights["model_dict"])
    net = nn.Sequential(*list(net.resnet.children())[:-2]).to(device)
    torchsummary.summary(net,input_size= (3,224,224))
    return net

def AddressResolve(path, label_dict):
    porn_p = os.path.join(path, porn)
    nonporn_p = os.path.join(path, non_porn)
    p_folders = os.listdir(porn_p)
    np_folders = os.listdir(nonporn_p)
    folder_address_dict = {}
    for _item in p_folders:
        folder_address_dict[_item] = {}
        folder_address_dict[_item]["label"] = label_dict["porn"]
        folder_address_dict[_item]["address"] = os.path.join(porn_p, _item)
        frames = os.listdir(folder_address_dict[_item]["address"])
        frames = sorted(frames, key=lambda x: int(x.split("_")[1]))
        folder_address_dict[_item]["frames"] = [os.path.join(folder_address_dict[_item]["address"], x) for x in frames]

    for _item in np_folders:
        folder_address_dict[_item] = {}
        folder_address_dict[_item]["label"] = label_dict["non_porn"]
        folder_address_dict[_item]["address"] = os.path.join(nonporn_p, _item)
        frames = os.listdir(folder_address_dict[_item]["address"])
        frames = sorted(frames, key=lambda x: int(x.split("_")[1]))
        folder_address_dict[_item]["frames"] = [os.path.join(folder_address_dict[_item]["address"], x) for x in frames]

    return folder_address_dict


def extract_feature(add_list, name, target_path, model):

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    images_list = []
    for i in range(len(add_list)):
        image = Image.open(add_list[i])
        image = data_transform(image)
        image = image.view(1, 3, 224, 224).to(device)
        with torch.no_grad():
            image_feature = feature_extractor(image)
            image_feature = image_feature.to(torch.device("cpu"))
            image_feature = image_feature.numpy()
            images_list.append(image_feature)
    images_list = np.array(images_list)
    images_list = images_list.reshape(-1, 512, 7, 7)
    images_list = torch.FloatTensor(images_list)
    save_dict = {name: images_list}
    torch.save(save_dict, os.path.join(target_path, name) + ".pt")
    print(images_list.shape)
    torch.cuda.empty_cache()



if __name__ == "__main__":
    args = parser.parse_args()

    label_dict = {
        "porn": 1,
        "non_porn": 0
    }
    porn = "porn"
    non_porn = "non_porn"

    train_dict = AddressResolve(train_add, label_dict)
    test_dict = AddressResolve(test_add, label_dict)
    val_dict = AddressResolve(val_add, label_dict)

    train_add = os.path.join(args.data_path, "train")
    val_add = os.path.join(args.data_path, "val")
    test_add = os.path.join(args.data_path, "test")

    torch.save(train_dict, "train_dict.pt")
    torch.save(test_dict, "test_dict.pt")
    torch.save(val_dict, "val_dict.pt")

    net = load_model(args.saved_model)

    for key in train_dict.keys():
        os.mkdir(os.path.join(args.target_path, "train"))
        os.mkdir(os.path.join(args.target_path, "train\\porn"))
        os.mkdir(os.path.join(args.target_path, "train\\non_porn"))
        if "vPorn" in key:
            extract_feature(train_dict[key]["frames"], key,
                            os.path.join(args.target_path, "train\\porn"), net)
        elif "vNonPorn" in key:
            extract_feature(train_dict[key]["frames"], key,
                            os.path.join(args.target_path, "train\\non_porn"), net)
    for key in test_dict.keys():
        os.mkdir(os.path.join(args.target_path, "test"))
        os.mkdir(os.path.join(args.target_path, "test\\porn"))
        os.mkdir(os.path.join(args.target_path, "test\\non_porn"))
        if "vPorn" in key:
            extract_feature(test_dict[key]["frames"], key,
                            os.path.join(args.target_path, "test\\porn"), net)
        elif "vNonPorn" in key:
            extract_feature(test_dict[key]["frames"], key,
                            os.path.join(args.target_path, "test\\non_porn"), net)
    for key in val_dict.keys():
        os.mkdir(os.path.join(args.target_path, "val"))
        os.mkdir(os.path.join(args.target_path, "val\\porn"))
        os.mkdir(os.path.join(args.target_path, "val\\non_porn"))
        if "vPorn" in key:
            extract_feature(val_dict[key]["frames"], key,
                            os.path.join(args.target_path, "val\\porn"), net)
        elif "vNonPorn" in key:
            extract_feature(val_dict[key]["frames"], key,
                            os.path.join(args.target_path, "val\\non_porn"), net)


