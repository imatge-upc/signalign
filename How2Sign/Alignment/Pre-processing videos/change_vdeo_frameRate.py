#!/usr/bin/env python3
import cv2
import os

"""
This script duplicates the frame rate of a video. The idea is to convert the rgb_front videos (25 frames/s) to 50 frames/s. For future approaches, the new frame should be interpolated from its neighbours.
"""

# SPLITS = ['val', 'test', 'train']
SPLITS = ['train']
for partition in SPLITS:    
    path = "/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/" + partition + "/rgb_front/raw_videos/"
    save_path = "/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/" + partition + "/video_50frames/rgb_front/"
    
    for video in os.listdir(path):
        #completing the paths
        vpath = path + "/" + video
        opath = save_path + "/" + video

        print(f"Converting video {video}", flush = True)
        cap = cv2.VideoCapture(vpath)
        first_iter = 1 
        while True:
            ret, frame = cap.read()
            if ret == False:
                break

            #obtaining the frame size only one time 
            if first_iter:
                height, width, layers = frame.shape
                size = (width,height)
                out = cv2.VideoWriter(opath,cv2.VideoWriter_fourcc(*'DIVX'), 50, size)
                first_iter=0
            
            #duplicating the frame to get a 50 frames/s video from a 25 frames/s one
            for i in range(2):    
                out.write(frame)

        out.release()
        del out
        print(f"Finished video {video}", flush = True)