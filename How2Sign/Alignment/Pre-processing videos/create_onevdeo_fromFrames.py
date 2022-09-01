#!/usr/bin/env python3
import cv2
import os
import argparse
import re

"""
This script creates one specific video from a set of images.
"""

def main(args):

    #specify some details of the name of the videos to differentiate them from other files in the same folder
    string_start_array = ["_Color", "_Depth"]
    string_finish = ".png"

    save_id = re.split("/", args.img_path)
    if len(save_id[-1]) > 5:
        save_id = save_id[-1]
    else:
        save_id = save_id[-2]

    for string_start in string_start_array:
        img_array = []
        size = None
        for filename in sorted(os.listdir(args.img_path)):
            print(f'{filename}', flush = True)                          
            if filename[-4:] == string_finish:
                if filename[:len(string_start)] == string_start:
                    #getting the images
                    img = cv2.imread(args.img_path + filename)
                    print(f'{filename}', flush = True)
                    height, width, _ = img.shape
                    size = (width,height)
                    img_array.append(img)

        #creating the video
        out = cv2.VideoWriter(args.save_path + save_id + string_start + '_video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_path', type=str, default='/mnt/gpid08/datasets/How2Sign/How2Sign_RGB-D/test/Data_2/05-17-2019/20190517_151520/', help='path where the images are stored')
    parser.add_argument('--save_path', type=str, default='/mnt/gpid08/datasets/How2Sign/How2Sign_RGB-D/videos/test/Data_2/05-17-2019/20190517_151520/', help='path where the generated video is to be saved')  

    args = parser.parse_args()
    main(args)

 