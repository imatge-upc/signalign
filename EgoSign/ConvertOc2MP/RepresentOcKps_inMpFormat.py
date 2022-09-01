import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
from squaternion import Quaternion
import math as m
"""
This script extracts one Oculus sequence of frames for both hands and matches them to the mediapipe ones. The next step would be to transform the hands (rigid 3D transform). 
"""

#functions to save/load the kps in json format
def save_json(path, video):
    json_object = json.dumps(video, ensure_ascii= False) 
    with open(path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)

def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")
    return data

#functions to transform quaternions to euler
def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def euler_to_quaternion(phi, theta, psi):
 
      qw = m.cos(phi/2) * m.cos(theta/2) * m.cos(psi/2) + m.sin(phi/2) * m.sin(theta/2) * m.sin(psi/2)
      qx = m.sin(phi/2) * m.cos(theta/2) * m.cos(psi/2) - m.cos(phi/2) * m.sin(theta/2) * m.sin(psi/2)
      qy = m.cos(phi/2) * m.sin(theta/2) * m.cos(psi/2) + m.sin(phi/2) * m.cos(theta/2) * m.sin(psi/2)
      qz = m.cos(phi/2) * m.cos(theta/2) * m.sin(psi/2) - m.sin(phi/2) * m.sin(theta/2) * m.cos(psi/2)

      return [qw, qx, qy, qz]

def quaternion_to_euler(w, x, y, z):
 
        t0 = 2 * (w * x + y * z)
        t1 = 1 - 2 * (x * x + y * y)
        X = m.atan2(t0, t1)
 
        t2 = 2 * (w * y - z * x)
        t2 = 1 if t2 > 1 else t2
        t2 = -1 if t2 < -1 else t2
        Y = m.asin(t2)
         
        t3 = 2 * (w * z + x * y)
        t4 = 1 - 2 * (y * y + z * z)
        Z = m.atan2(t3, t4)
 
        return X, Y, Z

#functions for representing the features
def compare3DKpsOC(kps1, kps2, label_kps1 = 'Recording', label_kps2 = 'Video1', title = "Keypoint comparison", save_img = 0, saving_path = "keypoint_comparison.png", show = 1):

    pointsA =[]
    pointsB =[]
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
        for l in connections_mp:
            p1 = [puntos[l[0]][0], puntos[l[0]][1], puntos[l[0]][2]]
            p2 = [puntos[l[1]][0], puntos[l[1]][1], puntos[l[1]][2]] 
            grafica.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')

        for l in connections_mp:
            p1 = [puntos2[l[0]][0], puntos2[l[0]][1], puntos2[l[0]][2]]
            p2 = [puntos2[l[1]][0], puntos2[l[1]][1], puntos2[l[1]][2]] 
            grafica.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')

        grafica.set_title(title)
        grafica.set_xlabel('eje x')
        grafica.set_ylabel('eje y')
        grafica.set_zlabel('eje z')
        grafica.legend()        
        if save_img:
            grafica.figure.savefig(saving_path)
        if show:
            plt.show()

def plot3DKpsOC(Kps, labels, colors = ['blue', 'red', 'green'], plot_connections = True):

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

    pointsA =[]
    figura = plt.figure()
    grafica = figura.add_subplot(111,projection = '3d')
    
    if plot_connections:
        for i in range(len(Kps[0])):
            pointsA.append([Kps[0][i],Kps[1][i],Kps[2][i]])
        
        pointsB =[]
        skip_array = []
        cnt=0
        for e in pointsA:
            if cnt not in skip_array:
                pointsB.append(e)
                cnt+=1 
            else:
                cnt+=1 
        puntos = np.array(pointsA)
        [xi, yi , zi] = np.transpose(puntos)
        grafica.scatter(xi, yi, zi, c="red", label = "kps")
        
        """
        p1 = [puntos[a][0], puntos[a][1], puntos[a][2]]
        p2 = [puntos[b][0], puntos[b][1], puntos[b][2]] 
        grafica.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')
        """
        for l in connections_mp:
            p1 = [puntos[l[0]][0], puntos[l[0]][1], puntos[l[0]][2]]
            p2 = [puntos[l[1]][0], puntos[l[1]][1], puntos[l[1]][2]] 
            grafica.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black')
        
        
        plt.show()

    else:
    
        for points, label, color in zip(Kps, labels, colors):
            puntos = np.array(points)
            [xi, yi , zi] = np.transpose(puntos)
            grafica.scatter(xi,yi,zi,
                    c = color,
                    marker='o',
                    label = label)


        grafica.set_title('Comparacion de kps de las manos')
        grafica.set_xlabel('eje x')
        grafica.set_ylabel('eje y')
        grafica.set_zlabel('eje z')

        grafica.legend()
        plt.show()

def findStartOculus(video, time_L, time_stamp_anterior):
    """
    Extracts the timestamp that is closer to a given timestamp (the previous one).
    Parameters:
        -video: Name of the video
        -time_L: The current time stamp to analyze
        -time_stamp_anterior: The previous time stamp
    """

    #extracting the date from the name of the video
    video = re.split("/", video)
    e = video[-1]
    time_video = e[11:19]
    time_video = re.sub("-",":", time_video)

    time_oc = time_L[13:21]
    
    #checking if it is the closest time 
    if time_oc == time_video:
        if abs(float(e[17:19]) - float(time_L[19:25])) < abs(float(e[17:19]) - float(time_stamp_anterior[19:25])):
            time_stamp = time_L
        else:
            time_stamp = time_stamp_anterior[19:25]
        
        return True, time_stamp, time_video
    
    else:
        time_stamp_anterior = time_L
        return False, time_stamp_anterior, time_video

def getOculusTimeSequence(path_hands, save_path_hands, video_name = "2022-05-09_13-49-39_rgb_front.mkv"):
    """
    Gets the oculus time sequence so it can be matched with the mediapipe sequence (each second).
    Parameters:
        -path_hands: Path where oculus keypoints are stored
        -save_path_hands: Path where the generated sequence is to be stored
        -video_name: The name of the rgb video.
    """  

    t_s = 0
    a=0
    c=0
    sequence = []
    indx = 0

    f = open(path_hands, 'r')
    for oc_frame in f:
        content = re.split('//', oc_frame)

        oc_pose = content[3]
        oc_pose = oc_pose[3:]
        oc_pose = eval(oc_pose)

        #matching the start of the video with the appropriate time stamp
        if not a: 
            a, b, c = findStartOculus(video_name, content[0], t_s)
            t_s = b
            if a:
                sequence.append(content[0])
                indx+=1
                c = re.sub(":","",c)
                c = str(int(c)+1)
                if int(c[4:]) == 60:
                    c = c[:2] + str(int(c[2:4])+1) + "00"
                if int(c[2:4]) == 60:
                    c = str(int(c[:2])+1) + "00" + c[4:]

                video_name = video_name[:11] + c[:2] + ":" + c[2:4] + ":"+ c[4:] + video_name[19:]
                video_name = re.sub(":", "-", video_name)
        
        #getting the appropriate oculus time stamp sequence      
        else:
            d, e, f = findStartOculus(video_name, content[0], t_s)
            t_s = e
            if d:
                sequence.append(content[0])
                indx+=1
                f = re.sub(":","",f)
                f = str(int(f)+1)
                if int(f[4:]) == 60:
                    f = f[:2] + str(int(f[2:4])+1) + "00"
                if int(f[2:4]) == 60:
                    f = str(int(f[:2])+1) + "00" + f[4:]
                video_name = video_name[:11] + f[:2] + ":" + f[2:4] + ":"+ f[4:] + video_name[19:]
                video_name = re.sub(":", "-", video_name)
    
    #saving the time sequence            
    save_json(save_path_hands,sequence)
    return sequence
              
def getOculusKpsFromSequence(path_hands, sequence):
    """
    Gets the oculus keypoint sequence from the created time sequence).
    Parameters:
        -path_hands: Path where oculus keypoints are stored
        -sequence: The generated time sequence (getOculusTimeSequence)
    """ 

    f = open(path_hands, 'r')
    indx_frame =0
    sequence_oc_kps = np.zeros((len(sequence), 3, 21))
    
    #process all stored frames in the txt files (oculus)
    for oc_frame in f:        
        content = re.split('//', oc_frame)        
        time = content[0]
        try:
            time_sequence = sequence[0]
        except:
            break
        
        print("Time", time,  time_sequence)
        if time == time_sequence:
            break
        
        #extracting pose kps
        x,y,z = [],[],[]
        oc_pose = content[3]
        oc_pose = oc_pose[3:]
        oc_pose = eval(oc_pose)

        #extracting rotation matrix
        oc_rot_quat = content[2]
        oc_rot_quat = oc_rot_quat[3:]
        oc_rot_quat = eval(oc_rot_quat)

        q = Quaternion(oc_rot_quat[0],oc_rot_quat[1],oc_rot_quat[2],oc_rot_quat[3])

        #extracting translation matrix
        oc_trans = content[1]
        oc_trans = oc_trans[3:]
        oc_trans= eval(oc_trans)   
        
        #transform the pose kps
        for coordenate, cnt in zip(oc_pose, range(len(oc_pose))):          
            v2 = qv_mult(q,tuple(coordenate))          

            x.append(v2[0] + oc_trans[0])
            y.append(v2[1] + oc_trans[1])
            z.append(v2[2] + oc_trans[2])

        #reorder the features to have them in the same order than mediapipe
        proper_order = [0,3,4,5,19,6,7,8,20,9,10,11,21,12,13,14,22,16,17,18,23]
        x_def,y_def,z_def = [],[],[]
        for order in proper_order:
            x_def.append(x[order])
            y_def.append(y[order])
            z_def.append(z[order])

        #creating the sequence
        sequence_oc_kps[indx_frame,0,:] = x_def
        sequence_oc_kps[indx_frame,1,:] = y_def
        sequence_oc_kps[indx_frame,2,:] = z_def
        indx_frame+=1      
            
    
    return sequence_oc_kps


def getVideoMpSequence():
    """
    Gets the mediapipe keypoint sequence from the stored features in json format).
    Parameters:
        -path_hands: Path where oculus keypoints are stored
        -sequence: The generated time sequence (getOculusTimeSequence)
    """ 
    #sacar y guardar las secuencias de los videos
    videos = ['test3_how2/rgb_front_kps.json', 'test3_how2/rgb_head_kps.json','test3_how2/rgb_side_kps.json']
    videos_save_path_seq = ['test3_how2/seq_rgb_front.json', 'test3_how2/seq_rgb_head.json','test3_how2/seq_rgb_side.json', ]

    videos_save_path2 = ['test3_how2/rgb_front_l.json', 'test3_how2/rgb_head_l.json','test3_how2/rgb_side_l.json', 'test3_how2/rgb_front_r.json', 'test3_how2/rgb_head_r.json','test3_how2/rgb_side_r.json']

    frame_rate = 30
    
    #saving the sequence
    names = ["front", "head", "side"]
    kps = dict()
    for video, saving_path, name in zip(videos, videos_save_path_seq, names):
        kps[name] = []
        vjson = load_json(video)
        for i in range(round(len(vjson[0])/frame_rate)):
            try:
                kps[name].append(vjson[0][frame_rate*(i)])
            except:
                continue
        save_json(saving_path, kps[name])


    rgb_front_frec_l = np.zeros((len(kps[names[0]]), 3, 21))
    rgb_head_frec_l = np.zeros((len(kps[names[1]]), 3, 21))
    rgb_side_frec_l = np.zeros((len(kps[names[2]]), 3, 21))
    rgb_front_frec_r = np.zeros((len(kps[names[0]]), 3, 21))
    rgb_head_frec_r = np.zeros((len(kps[names[1]]), 3, 21))
    rgb_side_frec_r = np.zeros((len(kps[names[2]]), 3, 21))

    #extracting the kps of the sequence 
    for indx_videos in range(len(names)):
        nmbr_frame = 0
        for frame in kps[names[indx_videos]]:           
            x,y,z = [], [], []
            skip_vis = 0
            for coordenate in frame:
                if skip_vis == 0:
                    x.append(coordenate)
                    skip_vis+=1
                elif skip_vis == 1:
                    y.append(coordenate)
                    skip_vis+=1
                elif skip_vis == 2:
                    z.append(coordenate)
                    skip_vis+=1

                elif skip_vis == 3:
                    skip_vis = 0              

            if indx_videos == 0:

                    rgb_front_frec_l[nmbr_frame,0,:] = x[:21]
                    rgb_front_frec_l[nmbr_frame,1,:] = y[:21]
                    rgb_front_frec_l[nmbr_frame,2,:] = z[:21]

                    rgb_front_frec_r[nmbr_frame,0,:] = x[21:]
                    rgb_front_frec_r[nmbr_frame,1,:] = y[21:]
                    rgb_front_frec_r[nmbr_frame,2,:] = z[21:]

            elif indx_videos == 1:

                    rgb_head_frec_l[nmbr_frame,0,:] = x[:21]
                    rgb_head_frec_l[nmbr_frame,1,:] = y[:21]
                    rgb_head_frec_l[nmbr_frame,2,:] = z[:21]

                    rgb_head_frec_r[nmbr_frame,0,:] = x[21:]
                    rgb_head_frec_r[nmbr_frame,1,:] = y[21:]
                    rgb_head_frec_r[nmbr_frame,2,:] = z[21:]
            else:

                    rgb_side_frec_l[nmbr_frame,0,:] = x[:21]
                    rgb_side_frec_l[nmbr_frame,1,:] = y[:21]
                    rgb_side_frec_l[nmbr_frame,2,:] = z[:21]

                    rgb_side_frec_r[nmbr_frame,0,:] = x[21:]
                    rgb_side_frec_r[nmbr_frame,1,:] = y[21:]
                    rgb_side_frec_r[nmbr_frame,2,:] = z[21:]

            nmbr_frame+=1        

    #saving every keypoint sequence 
    save_json(videos_save_path2[0], rgb_front_frec_l.tolist())
    save_json(videos_save_path2[1], rgb_head_frec_l.tolist())
    save_json(videos_save_path2[2], rgb_side_frec_l.tolist())

    save_json(videos_save_path2[3], rgb_front_frec_r.tolist())
    save_json(videos_save_path2[4], rgb_head_frec_r.tolist())
    save_json(videos_save_path2[5], rgb_side_frec_r.tolist())

    return rgb_front_frec_l, rgb_head_frec_l, rgb_side_frec_l, rgb_front_frec_r, rgb_head_frec_r, rgb_side_frec_r
    

if __name__ == '__main__':
    #CONFIGURE PATHS INSIDE FUNCTIONS BEFORE RUNNING SCRIPT

    #path where R and L keypoints are stored
    path_hands = ['test3_how2/_fZbAxSSbX4/L.txt', 'test3_how2/_fZbAxSSbX4/R.txt']
    save_path_hands = ["test3_how2/L.json", "test3_how2/R.json"]

    for h, save_path in zip(path_hands, save_path_hands):
        if h == path_hands[0]:
            sequence_L = getOculusTimeSequence(h, save_path)
            keypoints_oc_L= getOculusKpsFromSequence(h, sequence_L)
            save_json("oculus_L_kps2.json", keypoints_oc_L.tolist())
        else:
            sequence_R = getOculusTimeSequence(h, save_path)
            keypoints_oc_R = getOculusKpsFromSequence(h, sequence_R)
            save_json("oculus_R_kps2.json", keypoints_oc_R.tolist())

    rgb_front_frec_l, rgb_head_frec_l, rgb_side_frec_l, rgb_front_frec_r, rgb_head_frec_r, rgb_side_frec_r = getVideoMpSequence()


    #Making some tests
    #Mp and Oculus separated
    plot3DKpsOC(keypoints_oc_L[0], ["Front_left"], ['blue'], True)
    plot3DKpsOC(rgb_front_frec_l[50], ["Front_left"], ['blue'], True)

    #Mp and Oculus together
    compare3DKpsOC(keypoints_oc_L[50], rgb_front_frec_l[50],label_kps1="Oculus", label_kps2="Mp_front")
    compare3DKpsOC(keypoints_oc_L[50], rgb_head_frec_l[50],label_kps1="Oculus", label_kps2="Mp_head")
    compare3DKpsOC(keypoints_oc_L[45], rgb_side_frec_l[45],label_kps1="Oculus", label_kps2="Mp_side")

    #Both Oculus hands
    for frame in range(len(keypoints_oc_L)):
        compare3DKpsOC(keypoints_oc_L[frame], keypoints_oc_R[frame],label_kps1="Left", label_kps2="Right", save_img=1, saving_path="test3_how2/images_cmp/"+str(frame)+".png", show = 0)