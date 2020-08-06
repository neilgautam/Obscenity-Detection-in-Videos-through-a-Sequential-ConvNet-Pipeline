import numpy as np
import tensorflow as tf
import cv2
import time
import os
import matplotlib.pyplot as plt
import PIL
from human_detection_model import DetectorAPI
import argparse

parser = argparse.ArgumentParser(description="Human Detection Model")
parser.add_argument("-mp", "--model_path", help="Trained tensorflow Human Detection Model (RCNN) ")
parser.add_argument("-th", "--threshold", type=int, help="Threshold score to consider human is present")
parser.add_argument("-dp", "--data_directory_path", help="Path of data(Folder that contains frames)")
parser.add_argument("-tp", "--target_path", help="Target Directory where you want your data stored")


def return_final_lists(video_list, directory,threshold):
    for x in list(video_list.keys()):
        print("Processing ", x, end=" ")
        frames_list = video_list[x]
        human_shown = {}
        for frame in frames_list:
            img = cv2.imread(frame)
            img = cv2.resize(img, (600, 400))
            boxes, scores, classes, num = odapi.processFrame(img)

            thresh_count = 0
            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:
                    thresh_count += 1
                    box = boxes[i]
                    cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
            if thresh_count == 0:
                human_shown[frame.split("\\")[-1]] = 0
            else:
                human_shown[frame.split("\\")[-1]] = 1
        np.save(os.path.join(directory, x) + ".npy", human_shown)
        print(" --> Files Saved")

if __name__ == "__main__" :
    args = parser.parse_args()
    odapi = DetectorAPI(path_to_ckpt=args.model_path)

    train_dir = os.path.join(args.data_directory_path, "train", "porn")
    val_dir = os.path.join(args.data_directory_path, "val", "porn")
    test_dir = os.path.join(args.data_directory_path, "test", "porn")

    train_video_list = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]
    test_video_list = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]
    val_video_list = [os.path.join(val_dir, x) for x in os.listdir(val_dir)]

    train_dict = {}
    for path in train_video_list:
        temp_list = os.listdir(path)
        temp_list.sort(key=lambda x: int(x.split("_")[1]))
        temp_list = [os.path.join(path, x) for x in temp_list]
        train_dict[path.split("\\")[-1]] = temp_list

    return_final_lists(train_dict, args.target_path,args.threshold)

