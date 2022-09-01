import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

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
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def getKpsfromFrame(kpsArray):
    test_norm_array = []
    x_pose_l,y_pose_l,z_pose_l = [],[],[]
    for set_kps1 in kpsArray[0]:   
        
        #if set_kps2["Visibility"] > 0.7: 
        x_pose_l.append(set_kps1["X"])
        y_pose_l.append(set_kps1["Y"])
        z_pose_l.append(set_kps1["Z"])

        test_norm_array.append([set_kps1["X"], set_kps1["Y"], set_kps1["Z"]])
    return x_pose_l,y_pose_l,z_pose_l

def plot3DKps(kps1, kps2):

        pointsA =[]
        pointsB =[]

        for i in range(len(kps1[0])):
            pointsA.append([kps1[0][i],kps1[1][i],kps1[2][i]])
            pointsB.append([kps2[0][i],kps2[1][i],kps2[2][i]])

        puntos = np.array(pointsA)
        puntos2 = np.array(pointsB)

        # SALIDA
        figura = plt.figure()
        grafica = figura.add_subplot(111,projection = '3d')

        [xi, yi , zi] = np.transpose(puntos)
        [xi2, yi2 , zi2] = np.transpose(puntos2)

        grafica.scatter(xi,yi,zi,
                        c = 'blue',
                        marker='o',
                        label = 'Recording')
        grafica.scatter(xi2,yi2,zi2,
                        c = 'red',
                        marker='o',
                        label = 'Video1')

        grafica.set_title('puntos, dispersión-scatter')
        grafica.set_xlabel('eje x')
        grafica.set_ylabel('eje y')
        grafica.set_zlabel('eje z')
        grafica.legend()
        plt.show()

recording2 = np.zeros((3, 33))
video_2 = np.zeros((3, 33))

with open("../TestVideoFrecKps.json", "r", encoding="utf8") as f:
    kps_imgStart_video1 = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")


with open("../TestRecFrecKps.json", "r", encoding="utf8") as f:
    kps_imgStart_Rec = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")

start_manual = 46

for iters in range(10):
    adding= 5 + iters*5
    index = start_manual + adding

    recording2 = np.array(kps_imgStart_Rec[0])[index,:,:]
    video_2 = np.array(kps_imgStart_video1[0])[adding,:,:]

    plot3DKps(recording2, video_2)

    name_array = ["larm", "rarm"]
    for nmbr_arms in range(2):
        recording_l = np.zeros((3, 4))
        video_l = np.zeros((3, 4))

        recording_r = np.zeros((3, 4))
        video_r = np.zeros((3, 4))

        cnt = 0
        cnt2=0
        #extracting only hand kps
        for i in range(15,23):
            if i%2==0:
                
                recording_l[:,cnt] = recording2[:,i]
                video_l[:,cnt] = video_2[:,i]
                cnt+=1
            else:
                recording_r[:,cnt2] = recording2[:,i]
                video_r[:,cnt2] = video_2[:,i]
                cnt2+=1

        if i != 1:
            recording = recording_l
            video = video_l
        else:
            recording = recording_r
            video = video_r

        #obtaining the transformation matrices
        R, t = rigid_transform_3D(recording, video)

        json_object = json.dumps(R.tolist()) 
        with open('../FrameMatrices/R_' + name_array[nmbr_arms] + str(index) +'.json', "w") as outfile:
            outfile.write(json_object)

        json_object = json.dumps(t.tolist()) 
        with open('../FrameMatrices/t_' + name_array[nmbr_arms] + str(index) +'.json', "w") as outfile:
            outfile.write(json_object)

        #visualizing the difference between original features and transformed ones
        plot3DKps(recording, video)
        res = (R@recording) + t
        plot3DKps(video, res)