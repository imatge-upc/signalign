#!/usr/bin/env python3
import cv2
import os
import argparse

"""
This script creates videos from a specific set of images.
"""

def main(args):
    # SPLITS = ['val', 'test', 'train']
    SPLITS = ['train']

    #specify some details of the name of the videos to differentiate them from other files in the same folder
    string_start_array = ["_Color", "_Depth"]
    string_finish = ".txt"

    #iteratively explore the folders
    for partition in SPLITS:    
        for data_folder in os.listdir(args.img_path + partition + "/"):
            #os.mkdir(save_path + partition + "/" + data_folder)
            for date_folder in os.listdir(args.img_path + partition + "/" + data_folder + "/"):
                #os.mkdir(save_path + partition + "/" + data_folder + "/" + date_folder)
                for recording in os.listdir(args.img_path + partition + "/" + data_folder + "/" + date_folder + "/"):
                    #os.mkdir(save_path + partition + "/" + data_folder + "/" + date_folder + "/" + recording)
                    print(f'Starting {data_folder} - {date_folder} - {recording}', flush = True)
                    
                    for string_start in string_start_array:
                        try:
                            img_array = []
                            size = None
                            for filename in sorted(os.listdir(args.img_path + partition + "/" + data_folder + "/" + date_folder + "/" + recording + "/")):
                                
                                if filename[-4:] != string_finish:
                                    #processing only rgb or only depth images                        
                                    if  string_start in filename:
                                        #getting the images
                                        img = cv2.imread(args.img_path + partition + "/" + data_folder + "/" + date_folder + "/" + recording + "/" + filename)
                                        height, width, _ = img.shape
                                        size = (width,height)
                                        img_array.append(img)
  
                            #creating the video
                            out = cv2.VideoWriter(args.save_path + partition + "/" + data_folder + "/" + date_folder + "/" + recording + "/" + recording + string_start + '_video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
                            
                            for i in range(len(img_array)):
                                out.write(img_array[i])
                            out.release()

                            del out
                            del img_array

                            
                        except:
                            print(f'Skipping {recording}', flush = True)
                    
                    print(f'Finishing {data_folder} - {date_folder} - {recording}', flush = True)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_path', type=str, default='/mnt/gpid08/datasets/How2Sign/How2Sign_RGB-D/', help='root path where all the recordings are stored')
    parser.add_argument('--save_path', type=str, default='/mnt/gpid08/datasets/How2Sign/How2Sign_RGB-D/videos/', help='root path where the generated videos are to be saved')  


    args = parser.parse_args()
    main(args)
