import os 
import warnings
import cv2
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import sys
import argparse
warnings.filterwarnings("ignore")

## this code is for NPDI dataset 
parser = argparse.ArgumentParser(description = "NPDI Video Frames Extraction")

parser.add_argument("-srcadd","--address",help = "Source Address Containing Folders of Videos")
parser.add_argument("-trgadd","--taddress",help = "Target Address")

def address_preprocessing(dir_path,target_address):
    folder_names = [os.path.join(dir_path,x) for x in os.listdir(dir_path)]
    
    for folder in folder_names:
        content= [os.path.join(folder,x) for x in os.listdir(folder)]
        extract_frames(content,target_address,folder.split("\\")[-1]) # extraction of frame will take place at 4 frame per second
        
def extract_frames(address,target_address,folder_name):
    os.mkdir(os.path.join(target_address,folder_name))
    save_path = os.path.join(target_address,folder_name)
    for video_index ,add in enumerate(address):
        y = add.split("\\")[-1]
        name = y
        name = name[:-4]
        Video = cv2.VideoCapture(add)
        print("Accessing Video : ", add,"\n")
        print("Video dimension : ({},{},3)".format(Video.get(3),Video.get(4)))
        frame_rate = int(Video.get(5))
        print("Frame Rate : {}".format(frame_rate))
        if frame_rate<=4:
            continue
        frames = int(frame_rate/4)
        print("Extraction of FRAMES AT : {} ".format(frames))
        total = 1
        sec_frame_count = 1
        total_frame_count=0
        try:
            os.mkdir(save_path+"\\frames_"+name)
        except:
            None    
        while Video.isOpened():
            ret,frame = Video.read()
            if ret != True:
                break
            resized_frame = cv2.resize(frame,(224,224))
            sec_frame_count+=1
            if sec_frame_count>frame_rate:
                sec_frame_count = 1  
                total += 1
            if int(sec_frame_count%frames) == 0:
                total_frame_count +=1
                cv2.imwrite(save_path+"\\frames_"+name+"\\"+name+"_{}_{}_{}.jpg".format(total_frame_count,total,sec_frame_count//frames),resized_frame)
        print("Video {}-->{} Processed".format(video_index,add))
        Video.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    args = parser.parse_args()
    address_preprocessing(args.address,args.taddress)    

