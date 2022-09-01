import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
import mediapipe as mp
import json
import re

"""
This script is a first comparison between oculus features and mediapipe ones (in particular, the thumbs up sign)
"""

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#Allows to represent the points in a 3D space
def plot3DKps(Kps, labels, colors = ['blue', 'red', 'green']):
    figura = plt.figure()
    grafica = figura.add_subplot(111,projection = '3d')    
    for points, label, color in zip(Kps, labels, colors):
        puntos = np.array(points)
        [xi, yi , zi] = np.transpose(puntos)
        grafica.scatter(xi,yi,zi,
                c = color,
                marker='o',
                label = label)

    grafica.set_title('Hand Kps Comparation')
    grafica.set_xlabel('Axis X')
    grafica.set_ylabel('Axis Y')
    grafica.set_zlabel('Axis Z')

    grafica.legend()
    plt.show()

#************Processing Mp ************
path = "MpOculus_HandsUpDiff"
try:
    os.mkdir(path)
except:
    pass

puntos_rgbfront_left = []
puntos_rgbside_left = []
puntos_rgbhead_left = []

puntos_rgbfront_right = []
puntos_rgbside_right = []
puntos_rgbhead_right = []

#path where videos are stored
video_list =["test1-20220510T170226Z-001/test1/sync_videos/test1-rgb_front.mp4", "test1-20220510T170226Z-001/test1/sync_videos/test1-rgb_head.mp4", "test1-20220510T170226Z-001/test1/sync_videos/test1-rgb_side.mp4" ]
video_name = ["test1-rgb_front", "test1-rgb_head", "test1-rgb_side" ]

#initializing kp arrays
keypoints = dict()
keypoints_unknwn = dict()
for vname in video_name:
    keypoints[vname]=dict()
    keypoints[vname]["right"] = dict()
    keypoints[vname]["left"] = dict()
    keypoints_unknwn[vname] = dict()


indx = 0
#repeat the process for each video 
for video, vname in zip(video_list, video_name):
    cap = cv2.VideoCapture(video)
    frame_nmb = 0
    #process the first frames for better precision even though we only want the thumbs up ones
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if ret == False:        
                break

            #hands up moment in the video
            if frame_nmb == 330:
                cv2.imwrite(path + "/Imagef330" + vname + ".png", frame)

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks is not None:
                left_image = 1
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                if frame_nmb == 330:
                    #storing the frame
                    cv2.imwrite(path + "/Imagef330_mp" + vname + ".png", frame)

                    for hand_world_landmarks in results.multi_hand_world_landmarks:
                        #plot the 3D representation
                        mp_drawing.plot_landmarks(hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
                        #process kps ... (this extraction will be improved for other scripts)
                        if len(results.multi_hand_world_landmarks)>1:
                            if left_image:                                
                                keypoints[vname]["left"][indx] = []
                                for data_point in hand_world_landmarks.landmark:
                                    keypoints[vname]["left"][indx].append({
                                                'X': data_point.x,
                                                'Y': data_point.y,
                                                'Z': data_point.z,
                                                'Visibility': data_point.visibility,
                                                })
                                    left_image = 0
                                    if vname == video_name[0]:
                                        puntos_rgbfront_left.append([data_point.x,data_point.y,data_point.z])
                                    elif vname == video_name[1]:
                                        puntos_rgbhead_left.append([data_point.x,data_point.y,data_point.z])
                                    
                                    else:
                                        puntos_rgbside_left.append([data_point.x,data_point.y,data_point.z])

                            else:
                                
                                keypoints[vname]["right"][indx] = []
                                for data_point in hand_world_landmarks.landmark:
                                    
                                    keypoints[vname]["right"][indx].append({
                                                'X': data_point.x,
                                                'Y': data_point.y,
                                                'Z': data_point.z,
                                                'Visibility': data_point.visibility,
                                                })
                                    if vname == video_name[0]:
                                        puntos_rgbfront_right.append([data_point.x,data_point.y,data_point.z])
                                    elif vname == video_name[1]:
                                        puntos_rgbhead_right.append([data_point.x,data_point.y,data_point.z])
                                    
                                    else:
                                        puntos_rgbside_right.append([data_point.x,data_point.y,data_point.z])
                        else:
                            keypoints_unknwn[vname][indx] = []
                            for data_point in hand_world_landmarks.landmark:     
                                keypoints_unknwn[vname][indx].append({
                                            'X': data_point.x,
                                            'Y': data_point.y,
                                            'Z': data_point.z,
                                            'Visibility': data_point.visibility,
                                            })

                                if vname == video_name[0]:
                                    puntos_rgbfront_left.append([data_point.x,data_point.y,data_point.z])
                                elif vname == video_name[1]:
                                    puntos_rgbhead_left.append([data_point.x,data_point.y,data_point.z])
                                
                                else:
                                    puntos_rgbside_left.append([data_point.x,data_point.y,data_point.z])


                    break

            frame_nmb+=1


#saving the obtained kps

json_object = json.dumps(keypoints, indent = 11, ensure_ascii= False) 
#Writing output to a json file
with open(path + "/" + 'Mp_kps.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(keypoints_unknwn, indent = 11, ensure_ascii= False) 
#Writing output to a json file
with open(path+ "/" + 'Mp_kps_1hand.json', "w") as outfile:
    outfile.write(json_object)

#representing the kps
#plot3DKps([puntos_rgbfront_left], ["Front_left"], ['blue'])
#plot3DKps([puntos_rgbhead_left], ["Head_left"], ['blue'])
#plot3DKps([puntos_rgbside_left], ["Side_left"], ['blue'])

#plot3DKps([puntos_rgbfront_right], ["Front_right"], ['blue'])
#plot3DKps([puntos_rgbhead_right], ["Head_right"], ['blue'])
#plot3DKps([puntos_rgbside_right], ["Side_right"], ['blue'])

plot3DKps([puntos_rgbfront_left, puntos_rgbhead_left, puntos_rgbside_left ], ["Front_left","Head_left","Side_right"], ['blue', 'red', 'green'])



#************Processing Oculus ************

#load kps
f = open('test1-20220510T170226Z-001/test1/test1_L.txt', 'r')
f2 = open('test1-20220510T170226Z-001/test1/test1_R.txt', 'r')
nmbr_break = 0

#look for the time when the signer does the thumbs up
for oc_frame_l, oc_frame_r in zip(f,f2):
    content = re.split('//', oc_frame_l)
    
    fecha = content[0]
    fecha = fecha[13:24]
    if fecha == "11:29:35.08":
        oc_pose_l = content[3]
        oc_pose_l = oc_pose_l[3:] 
        oc_pose_l = eval(oc_pose_l)
        content = re.split('//', oc_frame_r)
        oc_pose_r = content[3]
        oc_pose_r = oc_pose_r[3:] 
        oc_pose_r = eval(oc_pose_r)
        break


#Representing the kps

#first declare the connections
connections = [
          [0, 1],
          [0, 2],
          [2, 3],
          [3, 4],
          [4, 5],
          [5, 19],
          [0, 6],
          [6, 7],
          [7, 8],
          [8, 20],
          [0, 9],
          [9, 10],
          [10, 11],
          [11, 21],
          [0, 12],
          [12, 13],
          [13, 14],
          [14, 22],
          [0, 15],
          [15, 16],
          [16, 17],
          [17, 18],
          [18, 23]]

connections_mp = [
          [0, 1],
          [1, 2],
          [2, 3],
          [3, 4],
          [0, 5],
          [5, 6],
          [6, 7],
          [7, 8],
          [5, 9],
          [9, 10],
          [10, 11],
          [11, 12],
          [9, 13],
          [13, 14],
          [14, 15],
          [15, 16],
          [13, 17],
          [0, 17],
          [17, 18],
          [18, 19],
          [19, 20]]

#compare the representations
manos = [oc_pose_l, oc_pose_r]
first_iter = 1
for mano, mano_mp in zip(manos, [[puntos_rgbfront_left, puntos_rgbhead_left,puntos_rgbside_left], [puntos_rgbfront_right,puntos_rgbhead_right]]):
    for points in mano_mp:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for l in connections:
            p1 = mano[l[0]]
            p2 = mano[l[1]]
            x, y, z = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
            ax.scatter(x, y, z, c='blue')            
            ax.plot(x, y, z, color='black')

        for l in connections_mp:
            p1 = points[l[0]]
            p2 = points[l[1]]
            x, y, z = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
            ax.scatter(x, y, z, c='red')
            ax.plot(x, y, z, color='black')
            
        plt.show()
        first_iter = 0  




