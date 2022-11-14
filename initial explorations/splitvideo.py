# file to read in video and export frames
# @oscars47
# see https://stackoverflow.com/questions/57688690/convert-video-to-frames-python for reference
from PIL import Image
import os
import cv2
import numpy as numpy
import matplotlib.pyplot as plt

#print(cv.__version__)

def get_frame(sec, vidcap, count, vid_name, data_dir):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        print('writing frame ' + str(count))
        cv2.imwrite(os.path.join(data_dir, 'images', vid_name +'_'+ str(count) + '.jpg'), image)     # save frame as JPG file
    return hasFrames

def convert_video(path, vid_name, data_dir):
    vidcap = cv2.VideoCapture(path)

    sec = 0
    frameRate = 240 # it will capture 240 images per second
    count=1
    success = get_frame(sec, vidcap, count, vid_name, data_dir)

    while success:
        count = count + 1
        sec = sec + 1 / frameRate
        #sec = round(sec, 10)
        success = get_frame(sec, vidcap, count, vid_name, data_dir)


#convert all videos in 'balls' directory
ball_dir = '/Volumes/PHLUID/Data/Balls/'
for name in os.listdir(ball_dir):
    if name.endswith('.MOV'):
        print('extracting movie ' + name)
        # extract vid name from the name
        vid_name = name.split('.')[0]
        # concatenate path
        path = os.path.join(ball_dir, name)
        # create folder for images if not done so already
        store_path = os.path.join(ball_dir, 'images')
        if not (os.path.isdir(store_path)):
            os.mkdir(store_path)
        convert_video(path, vid_name, ball_dir)