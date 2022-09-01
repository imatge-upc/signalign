#!/usr/bin/env python3
import cv2
import pandas as pd 
#import glob
#import os
import json
import numpy as np
import matplotlib.pyplot as plt
#import argparse
from mpl_toolkits.mplot3d import Axes3D
import math
import h5py  


def rigid_transform_3D(A, B):
    """
    Calculates the 3D rigid transform of 2 sets of points, returning the R and T matrices that allow it. The code was extracted from an existing repository: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    """
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def load_json(path):
    """
    Loads the data from a json file 
    """
    with open(path, "r", encoding="utf8") as f:
        data = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")
    return data



def compare3DKps(kps1, kps2, label_kps1 = 'Recording', label_kps2 = 'Video1', title = "Keypoint comparison", save_img = 0, saving_path = "keypoint_comparison.png"):
    """
    Represents two sets of 3D features together to highlight the differences between them.
    Parameters:
        -label_kps1 and label_kps2 : name of each set of features.
        -title: title of the graphic.
        -save_img: if true saves the graphic.
        -saving_path: the path where the graphic is going to be saved

    """
    pointsA =[]
    pointsB =[]

    if len(kps1[0]) != len(kps2[0]):
        raise Exception("Array lenght must be the same")
    else:
        
        #Getting the points
        for i in range(len(kps1[0])):
            pointsA.append([kps1[0][i],kps1[1][i],kps1[2][i]])
            pointsB.append([kps2[0][i],kps2[1][i],kps2[2][i]])

        #Adapting the points 
        puntos = np.array(pointsA)
        puntos2 = np.array(pointsB)
        [xi, yi , zi] = np.transpose(puntos)
        [xi2, yi2 , zi2] = np.transpose(puntos2)

        #Plotting results
        figura = plt.figure()
        grafica = figura.add_subplot(111,projection = '3d')
        grafica.scatter(xi,yi,zi,
                        c = 'blue',
                        marker='o',
                        label = label_kps1)
        grafica.scatter(xi2,yi2,zi2,
                        c = 'red',
                        marker='o',
                        label = label_kps2)

        grafica.set_title(title)
        grafica.set_xlabel('eje x')
        grafica.set_ylabel('eje y')
        grafica.set_zlabel('eje z')
        grafica.legend()        
        if save_img:
            grafica.figure.savefig(saving_path)
        plt.show()

def get_kps_from_json_dict(vname, isRecording, split = "test"):
    """
    Loads the stored features in json format.
    The kps were extracted from 25 frames/s videos but 50 frames/s videos are needed to run this script. To get 50 frames/s, features of each frame were duplicated as if there were two consecutive equal pixels. 
    Parameters:
        -vname: name of the video (rgb videos and rgb from the rgb-d videos).
        -isRecording: True if the video is from the rgb-d set. 
        -split: train - test - val depending on the partition of the videos 
    """
    #path where features of both videos and recordings are stored
    rec_path = f'/mnt/gpid08/datasets/How2Sign/How2Sign_RGB-D/videos/{split}/MediaPipe_{split}_worldlmarks.h5'
    vid_path = f'/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/{split}/rgb_front/features/mediapipe/world-landmarks-V2/MediaPipe_{split}_worldlmarks.h5'

    #get the features from each h5 file
    if isRecording:
        h5File = h5py.File(rec_path, "r")
        kpsh5 = h5File[vname]
        #Considering only body and hand kps (21 + 21 + 33)  
        kps_array = np.zeros((len(kpsh5), 3, 75))
        
    else:    
        h5File = h5py.File(vid_path, "r")
        kpsh5 = h5File[vname]
        #Considering only body and hand kps (21 + 21 + 33). Multiplied by two to get 50 frames/s (the size is the double)
        kps_array = np.zeros((len(kpsh5)*2, 3, 75))

    cnt_frames = 0
    for frame in kpsh5:    
        x_pose,y_pose,z_pose = [],[],[]
        coordinate = 0

        #skipping face features
        for kp in frame[(468*4):]:
            #x
            if coordinate == 0:
                x_pose.append(kp)
                coordinate+=1
            #y
            elif coordinate == 1:
                y_pose.append(kp)
                coordinate+=1
            #z
            elif coordinate == 2:
                z_pose.append(kp)
                coordinate+=1

            #visibility
            else:
                coordinate=0
        
        for i in range(2):
            kps_array[cnt_frames, 0,:] = x_pose
            kps_array[cnt_frames, 1,:] = y_pose
            kps_array[cnt_frames, 2,:] = z_pose
            cnt_frames +=1

            #If is not recording stores a duplicate frame 
            if isRecording:
                break

    return kps_array


def getAlignmentParams(distv_vdeo, distv_rec, start_vdeo, start_rec):
    """
    Obtains the alignment parameters of the videos.
    Parameters:
        - distv_vdeo and distv_rec: the arrays where the minimum distance is stored.
        - start_vdeo and start_rec: the arrays where the start frame is stored.
    """

    #extracting the parameters of the recording that have fewer error
    minRec = np.amin(np.array(distv_rec))
    minRecPos = list(distv_rec).index(minRec)
    minStartRec = list(start_rec)[minRecPos]    
    
    #extracting the parameters of the video that have fewer error
    minVid = np.amin(np.array(distv_vdeo))
    minVidPos = list(distv_vdeo).index(minVid)
    minStartVid = list(start_vdeo)[minVidPos]

    #comparing them and returning the set that has fewer error
    if minRec<minVid:
        return 1,minRecPos,minStartRec,minRec
    else:
        return 0,minVidPos,minStartVid,minVid

def getVideoSequence(recording, video, offset_rec, offset_video):
    """
    Obtains the matching sequence of frames of both videos (per second).
    Parameters:
        -recording and video: the videos that are to be aligned.
        -offset_rec, offset_video: the increment to the beginning of each video. 
    """
    #Considering only body and hand kps (21 + 21 + 33)
    video_frec= np.zeros((int(round(len(video)/50)), 3, 75))
    recording_frec= np.zeros((int(round(len(recording)/30)), 3, 75))

    #saveKp_path, loadKp_path = ""

    #extracting video kps
    for i in range(round(len(video)/50)):
        try:
            video_frec[i,:,:] = video[50*(i) + offset_video,:,:]
            #img = cv2.imread(saveKp_path + "/Video/" + str(50*(i) + offset_video) + ".png")
            #cv2.imwrite(loadKp_path + "/Video/" + str(50*(i) + offset_video) + ".png", img)
        except:
            continue

    #extracting recording kps
    for i in range(round(len(recording)/30)):
        try:
            recording_frec[i,:,:] = recording[30*(i) + offset_rec,:,:]
            #img = cv2.imread(saveKp_path + "/Recording/" + str(30*(i) + offset_rec) + ".png")
            #cv2.imwrite(loadKp_path + "/Recording/" + str(30*(i) + offset_rec) + ".png", img)
        except:
            continue
    
    return recording_frec, video_frec


def getDifferenceBtwnKps(recording_frec, video_frec, considerHandKps = 0):
    """
    Obtains the distance (error) between two sets of points.
    Parameters:
        -recording_frec, video_frec: the calculated coincident arrays of frames.
    """
    dist_vector = []
    for frame in range(len(recording_frec)):
        mean = 0
        mean_g = 0
        cnt = 1
        try:
            for frame2 in range(len(video_frec)):
                #Visualizing the difference before the transformation
                #compare3DKps(video_frec[frame2,:,:25], recording_frec[frame+frame2,:,:25], title="Before")
                #compare3DKps(video_frec[frame2,:,33:], recording_frec[frame+frame2,:,33:], title="Before")

                #pose transformation
                R, t = rigid_transform_3D(recording_frec[frame+frame2,:,:33], video_frec[frame2,:,:33])
                recording_frec[frame+frame2,:,:33] = (np.array(R)@recording_frec[frame+frame2,:,:33]) + np.array(t)
                
                #hand transformation
                if considerHandKps:
                    R_h, t_h = rigid_transform_3D(recording_frec[frame+frame2,:,33:], video_frec[frame2,:,33:])
                    recording_frec[frame+frame2,:,33:] = (np.array(R_h)@recording_frec[frame+frame2,:,33:]) + np.array(t_h)
                
                #Visualizing the difference after the transformation
                #compare3DKps(video_frec[frame2,:,:25], recording_frec[frame+frame2,:,:25], title="After")
                #compare3DKps(video_frec[frame2,:,33:], recording_frec[frame+frame2,:,33:], title="After")

                #if only body estimation until 25 to avoid points below the waist
                iter_range = 25 if considerHandKps else 75

                #calculating the distance
                for i in range(iter_range):
                    x1 = recording_frec[frame+frame2][0][i]
                    y1 = recording_frec[frame+frame2][1][i]
                    z1 = recording_frec[frame+frame2][2][i]
                    x2 = video_frec[frame2][0][i]
                    y2 = video_frec[frame2][1][i]
                    z2 = video_frec[frame2][2][i]
                    
                    #skip bad poses
                    if (x1 == -1 and y1 == -1 and z1 == -1) or (x2 == -1 and y2 == -1 and z2 == -1):
                        continue
                    else:
                        #obtaining the error of 1 kp
                        d = math.pow(math.pow(float(x1)-float(x2),2) + math.pow(float(y1)-float(y2),2) +math.pow(float(z1)-float(z2),2),0.5)
                        mean = d + mean
                        cnt +=1
                
                mean_g =  mean_g + mean/cnt

            #storing the error of the kps of the whole frame       
            dist_vector.append(mean_g)
        except:
            break
    
    #returning also the number of frames just for visualization purposes
    n_frames=[]
    for i in range(len(dist_vector)):
        n_frames.append(i)

    return dist_vector, n_frames 

def plotDifference(x,y, xlbl="Frames", ylbl="Error", title = "Error between video Kps"):
    """
    Represents the evolution of the error. The represented minimum value should be where the start of both videos is the same.
    """
    plt.figure(figsize=(10,7))
    plt.plot(x, y, label = "Error")
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.legend(loc=1)
    plt.title(title)
    plt.show()


def align_main(recording_path, video_path, partition):

    #1. Import the saved kps for each frame of the long video
    recording = get_kps_from_json_dict(recording_path, 1, partition)
    video = get_kps_from_json_dict(video_path, 0, partition)

    #Initializing alignment parameters arrays 
    offset_video, offset_recording, start_video, start_recording  = [],[],[],[]

    #2. Iterate to get the parameters
    #first iter - wether apply offset to video or recording
    for step in range(2):
        #second iter - amount of applied offset (from 0-49 and from 0-29 because of frame rate)
        # 49 can be set to 29 to get quicker but a little less precise results. 
        for incr in range(49):

            if step == 0: 
                #applying offset in the video           
                recording_frec, video_frec= getVideoSequence(recording, video, 0, incr)
            else:
                #applying offset in the recording
                recording_frec, video_frec= getVideoSequence(recording, video, incr, 0)
                if incr == 29:
                    break   
            
            #Calculate the difference between transformed and video kps
            dist_vector, n_frames = getDifferenceBtwnKps(recording_frec, video_frec)

            #Save the minimum difference, the moment when that happens and the video where the offset has been applied
            if step == 0:
                offset_video.append(np.amin(np.array(dist_vector)))
                start_video.append(dist_vector.index(np.amin(np.array(dist_vector))))
                #print(f'{np.amin(np.array(dist_vector))}, {step}, {incr}', flush = True)
                #plotDifference(n_frames,dist_vector)
                
            else:
                offset_recording.append(np.amin(np.array(dist_vector))) 
                start_recording.append(dist_vector.index(np.amin(np.array(dist_vector))))          
                #print(f'{np.amin(np.array(dist_vector))}, {step}, {incr}', flush=True)
                #plotDifference(n_frames,dist_vector)
        
    #get the alignment parameters
    type_offset, offset,start,_ = getAlignmentParams(offset_video,offset_recording, start_video, start_recording)
    video_to_apply_offset = "recording" if type_offset else "video" 

    return [video_to_apply_offset, offset, start]

if __name__ == '__main__': 
    #Setting the proper environment to iteratively calculate the alignment for all the pairs of video and recording that are in the excel file.

    #configure running parameters

    partition = 'train'
    #path to the excel that contains the video-recording relation 
    RecVideoRelationPath = "../"
    #path to store the alignments
    save_params_path = "../"
    #processing the rows (excel) between start and end
    start = 0
    end = 200 

    #init variables 
    number_id = 0    
    algnment_row = []


    excel_path = f"{RecVideoRelationPath}/Recording_and_Video_Relation-{partition}.csv"
    
    try:    
        df = pd.read_csv(excel_path, sep=";")
    except:
        df = pd.read_csv(excel_path, sep=";", encoding="latin-1")


    for name, count in zip(df["Name_List"], range(len(df["Name_List"]))):
        if count >= start and count < end:
            #len(name)>5 to skip the rows that were manually added in the val and test excels
            if len(name)>5:
                #searching the video name in the excel column
                search = df.loc[:, "Name_List"] == name
                df_row = df.loc[search]
                #extracting the video and recording names
                vname = df_row["ID"] + df_row["Unnamed: 2"]
                recname = df_row["Recording"]
                
                try:
                    print(f'Computing: {count}\n{vname[count][:-4]}', flush = True)
                    print(f'{recname[count]}', flush = True)

                    #filter the names
                    video_path=vname[count][:-4]
                    recording_path=recname[count] + "_Color_video"

                    #obtain the alignment for those videos
                    params = align_main(recording_path, video_path, partition)

                    #add more information
                    params.append(vname[count][:-4])
                    params.append(recname[count])
                    params.append(partition)
                    
                    #save the obtained parameters
                    algnment_row.append(params)

                except:
                    #skip if there is any problem with the calculation of the alignment (maybe one video has few frames)                    
                    print(f'***SKipped***', flush = True)            
            
            else:
                #videos underlined in yellow in the original excel
                algnment_row.append([-1,-1,-1,-1,-1, partition])

            #save all parameters
            df2 = pd.DataFrame(algnment_row)
            df2.columns = ['Video to apply the offset', 'offset', 'start', 'Video', 'Recording', 'Partition']
            df2.to_csv(f'{save_params_path}/alignment_params_{partition}_{number_id}.csv', sep=';')