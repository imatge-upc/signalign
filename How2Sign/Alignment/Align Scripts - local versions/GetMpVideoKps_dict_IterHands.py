import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
import math

#calculates the transformation between 2 matrices
def rigid_transform_3D(A, B):
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

#load and save json files
def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")
    return data
def save_json(path, video):
    json_object = json.dumps(video, ensure_ascii= False) 
    with open(path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)

#compare two different sets of features 
def compare3DKps(kps1, kps2, label_kps1 = 'Recording', label_kps2 = 'Video1', title = "Keypoint comparison", save_img = 0, saving_path = "keypoint_comparison.png"):
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

#loading the extracted features in json format (dict)
def load_kps_from_json_dict(path = "../Recording.json"):
    
    kps_json= load_json(path)
    save_array = np.zeros((len(kps_json[0]), 3, 75))

    for frame  in kps_json[0]:    
        x_pose,y_pose,z_pose = [],[],[]
        for set_kps in kps_json[0][str(frame)]: 
            
            x_pose.append(set_kps["X"])
            y_pose.append(set_kps["Y"])
            z_pose.append(set_kps["Z"])
    
        save_array[int(frame), 0,:] = x_pose
        save_array[int(frame), 1,:] = y_pose
        save_array[int(frame), 2,:] = z_pose
    
    #print(save_array)
    return save_array

#gets the alignment parameters between video and recording (the one with minimum error)
def getAlignmentParams(distv_vdeo, distv_rec, start_vdeo, start_rec):
    #print(distv_rec)
    #print(distv_rec)
    a = np.amin(np.array(distv_rec))
    a_pos = list(distv_rec).index(a)
    start_a = list(start_rec)[a_pos]    
    
    b = np.amin(np.array(distv_vdeo))
    b_pos = list(distv_vdeo).index(b)
    start_b = list(start_vdeo)[b_pos]
    #print(a,b)
    if a<b:
        return 1,a_pos,start_a,a
    else:
        return 0,b_pos,start_b,b

#creates a sequence of frames (each second)
def getVideoSequence(recording, video, param1, param2):
    video_frec= np.zeros((int(round(len(video)/50)), 3, 75))
    recording_frec= np.zeros((int(round(len(recording)/30)), 3, 75))

    for i in range(round(len(video)/50)):
        try:
            video_frec[i,:,:] = video[50*(i) + param1,:,:]
            #img = cv2.imread("D:/CSIC/Proyectos/Alignment/alignment/Images/Video2/"+str(50*(i)+15) + ".png")
            #cv2.imwrite("TransformationKps/metadata/Images/vdeo" + str(50*(i)+15) + ".png", img)
        except:
            continue
    for i in range(round(len(recording)/30)):
        try:
            recording_frec[i,:,:] = recording[30*(i) + param2,:,:]
            #img2 = cv2.imread("D:/CSIC/Proyectos/Alignment/alignment/Images/VideoLong/"+str(30*(i)) + ".png")
            #cv2.imwrite("TransformationKps/metadata/Images/rec" + str(30*(i)) + ".png", img2)
        except:
            continue
    
    return recording_frec, video_frec

#gets the distance (error) between the sets of keypoints of two sequences of frames
def getDifferenceBtwnKps(recording_frec, video_frec):
    dist_vector = []
    for frame in range(len(recording_frec)):
        mean = 0
        mean_g = 0
        cnt = 1
        try:
            for frame2 in range(len(video_frec)):
                #compare3DKps(video_frec[frame2,:,:25], recording_frec[frame+frame2,:,:25], title="Before")

                R, t = rigid_transform_3D(recording_frec[frame+frame2,:,:33], video_frec[frame2,:,:33])                
                R_h, t_h = rigid_transform_3D(recording_frec[frame+frame2,:,33:], video_frec[frame2,:,33:])

                recording_frec[frame+frame2,:,:33] = (np.array(R)@recording_frec[frame+frame2,:,:33]) + np.array(t)
                recording_frec[frame+frame2,:,33:] = (np.array(R_h)@recording_frec[frame+frame2,:,33:]) + np.array(t_h)
                #compare3DKps(video_frec[frame2,:,:25], recording_frec[frame+frame2,:,:25], title="After")


                for i in range(75):
                    x1 = recording_frec[frame+frame2][0][i]
                    y1 = recording_frec[frame+frame2][1][i]
                    z1 = recording_frec[frame+frame2][2][i]
                    x2 = video_frec[frame2][0][i]
                    y2 = video_frec[frame2][1][i]
                    z2 = video_frec[frame2][2][i]

                    if (x1 == -1 and y1 == -1 and z1 == -1) or (x2 == -1 and y2 == -1 and z2 == -1):
                        skip=1
                    else:
                        d = math.pow(math.pow(float(x1)-float(x2),2) + math.pow(float(y1)-float(y2),2) +math.pow(float(z1)-float(z2),2),0.5)
                        #print(d)
                        mean = d + mean
                        cnt +=1
                
                mean_g =  mean_g + mean/cnt
                    
            dist_vector.append(mean_g)
        except:
            break
    
    n_frames=[]
    for i in range(len(dist_vector)):
        n_frames.append(i)

    return dist_vector, n_frames 

#visualize the evolution of the error between 2 squences of frames
def plotDifference(x,y):
    plt.figure(figsize=(10,7))
    plt.plot(x, y, label = "Error")
    plt.xlabel("Frames")
    plt.ylabel("Error")
    plt.legend(loc=1)
    plt.title("Error between video Kps")
    plt.show()

def main(args):

    #1. Import the saved kps 
    recording = load_kps_from_json_dict(args.recording_path)
    video = load_kps_from_json_dict(args.video_path)

    #Initializing alignment parameters arrays 
    offset_video, offset_recording, start_video, start_recording  = [],[],[],[]

    #2. Iterate to get the parameters
    #first iter - wether apply offset to video or recording
    for step in range(2):
        #second iter - amount of applied offset (from 0-49 and from 0-29 because of frame rate)
        for incr in range(49):
            if step == 0:            
                recording_frec, video_frec= getVideoSequence(recording, video, incr, 0)
            
            else:
                recording_frec, video_frec= getVideoSequence(recording, video, 0, incr)
                if incr == 30:
                    break   
            
            #Save sequences
            #save_json('../TestVideoFrecKps.json', video_frec.tolist())
            #save_json('../TestRecFrecKps.json', recording_frec.tolist())

            
            #Compare transformations with the videos
            #for frame, frame2 in zip(range(7), range(126,134)):
                #compare3DKps(video_frec[frame,:,:25], recording_frec[frame2,:,:25])

            #Calculate the difference between transformed and video kps
            dist_vector, n_frames = getDifferenceBtwnKps(recording_frec, video_frec)

            #Save the minimum difference, the moment when that happens and the video where the offset has been applied
            if step == 0:
                offset_video.append(np.amin(np.array(dist_vector)))
                start_video.append(dist_vector.index(np.amin(np.array(dist_vector))))
                print(np.amin(np.array(dist_vector)), step, incr)
                #plotDifference(n_frames,dist_vector)
                
            else:
                offset_recording.append(np.amin(np.array(dist_vector))) 
                start_recording.append(dist_vector.index(np.amin(np.array(dist_vector))))          
                print(np.amin(np.array(dist_vector)), step, incr)
                #plotDifference(n_frames,dist_vector)
        
    #get the alignment parameters
    type_offset, offset,start,_ = getAlignmentParams(offset_video,offset_recording, start_video, start_recording)
    video_to_apply_offset = "recording" if type_offset else "video" 
    save_json(args.save_params_path, [video_to_apply_offset, offset, start])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--recording_path', type=str, default="../20190530_161241_hands.json", help='path where recording kps are stored')
    parser.add_argument('--video_path', type=str, default="../G23JltC2N8g_hands.json", help='path where video kps are stored')

    parser.add_argument('--save_params_path', type=str, default='../VideoParams.json', help='path where alignment parameters should be stored')
        
    args = parser.parse_args()
    main(args)

