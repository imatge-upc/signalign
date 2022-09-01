import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from squaternion import Quaternion
import math as m

"""
This script is to test the oculus representations after computing the transformation matrices of each time stamp
"""

#to get the angle transformations
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

#to compare kps
def compare3DKps(kps1, kps2, label_kps1 = 'Recording', label_kps2 = 'Video1', title = "Keypoint comparison", save_img = 0, saving_path = "keypoint_comparison.png", show = 1):
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

#process the oculus information (but manually introduced from the txt files)
def getOculusKpsFromManualTimestamp(TimeStampValues):
    sequence_oc_kps = np.zeros((1, 3, 21))
    content = re.split('//', TimeStampValues)        
    time = content[0]

    x,y,z = [],[],[]
    oc_pose = content[3]
    oc_pose = oc_pose[3:]
    oc_pose = eval(oc_pose)

    oc_rot_quat = content[2]
    oc_rot_quat = oc_rot_quat[3:]
    oc_rot_quat = eval(oc_rot_quat)

    q = Quaternion(oc_rot_quat[0],oc_rot_quat[1],oc_rot_quat[2],oc_rot_quat[3])

    oc_trans = content[1]
    oc_trans = oc_trans[3:]
    oc_trans= eval(oc_trans)   
    
    for coordenate, cnt in zip(oc_pose, range(len(oc_pose))):
        
        v2 = qv_mult(q,tuple(coordenate))
        x.append(v2[0] + oc_trans[0])
        y.append(v2[1] + oc_trans[1])
        z.append(v2[2] + oc_trans[2])


    proper_order = [0,3,4,5,19,6,7,8,20,9,10,11,21,12,13,14,22,16,17,18,23]
    x_def,y_def,z_def = [],[],[]
    for order in proper_order:
        x_def.append(x[order])
        y_def.append(y[order])
        z_def.append(z[order])

    sequence_oc_kps[0,0,:] = x_def
    sequence_oc_kps[0,1,:] = y_def
    sequence_oc_kps[0,2,:] = z_def 
            
    return sequence_oc_kps

#Introducing a part of the txt file (both left and right)
timestamp_l = ["T 11/06/2022 11:50:41.02103-04:00 // P [0.0000000,0.0000000,0.0000000] // R [0.0000000,0.7071068,0.0000000,0.7071068] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0684713,0.0191577,0.0280285],[0.1022644,0.0191577,0.0280285],[0.0959962,0.0073165,0.0235507],[0.1339235,0.0073165,0.0235506],[0.1582271,0.0073165,0.0235506],[0.0956466,0.0025432,0.0017259],[0.1385736,0.0025432,0.0017259],[0.1661232,0.0025432,0.0017259],[0.0886938,0.0065293,-0.0174653],[0.1276899,0.0065293,-0.0174653],[0.1542632,0.0065293,-0.0174653],[0.0340736,0.0094198,-0.0229986],[0.0797241,0.0094208,-0.0229964],[0.1104445,0.0094208,-0.0229964],[0.1307558,0.0094208,-0.0229964],[0.1268551,0.0184873,0.0290555],[0.1805905,0.0062914,0.0232550],[0.1910881,0.0014059,0.0014172],[0.1785893,0.0049211,-0.0177232],[0.1526782,0.0082047,-0.0227499],]",
"T 11/06/2022 11:50:41.12986-04:00 // P [0.0000000,0.0000000,0.0000000] // R [0.0000000,0.7071068,0.0000000,0.7071068] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0684713,0.0191577,0.0280285],[0.1022644,0.0191577,0.0280285],[0.0959962,0.0073165,0.0235507],[0.1339235,0.0073165,0.0235506],[0.1582271,0.0073165,0.0235506],[0.0956466,0.0025432,0.0017259],[0.1385736,0.0025432,0.0017259],[0.1661232,0.0025432,0.0017259],[0.0886938,0.0065293,-0.0174653],[0.1276899,0.0065293,-0.0174653],[0.1542632,0.0065293,-0.0174653],[0.0340736,0.0094198,-0.0229986],[0.0797241,0.0094208,-0.0229964],[0.1104445,0.0094208,-0.0229964],[0.1307558,0.0094208,-0.0229964],[0.1268551,0.0184873,0.0290555],[0.1805905,0.0062914,0.0232550],[0.1910881,0.0014059,0.0014172],[0.1785893,0.0049211,-0.0177232],[0.1526782,0.0082047,-0.0227499],]",
"T 11/06/2022 11:50:41.19765-04:00 // P [0.0000000,0.0000000,0.0000000] // R [0.0000000,0.7071068,0.0000000,0.7071068] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0684713,0.0191577,0.0280285],[0.1022644,0.0191577,0.0280285],[0.0959962,0.0073165,0.0235507],[0.1339235,0.0073165,0.0235506],[0.1582271,0.0073165,0.0235506],[0.0956466,0.0025432,0.0017259],[0.1385736,0.0025432,0.0017259],[0.1661232,0.0025432,0.0017259],[0.0886938,0.0065293,-0.0174653],[0.1276899,0.0065293,-0.0174653],[0.1542632,0.0065293,-0.0174653],[0.0340736,0.0094198,-0.0229986],[0.0797241,0.0094208,-0.0229964],[0.1104445,0.0094208,-0.0229964],[0.1307558,0.0094208,-0.0229964],[0.1268551,0.0184873,0.0290555],[0.1805905,0.0062914,0.0232550],[0.1910881,0.0014059,0.0014172],[0.1785893,0.0049211,-0.0177232],[0.1526782,0.0082047,-0.0227499],]",
"T 11/06/2022 11:50:41.27179-04:00 // P [0.0000000,0.0000000,0.0000000] // R [0.0000000,0.7071068,0.0000000,0.7071068] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0684713,0.0191577,0.0280285],[0.1022644,0.0191577,0.0280285],[0.0959962,0.0073165,0.0235507],[0.1339235,0.0073165,0.0235506],[0.1582271,0.0073165,0.0235506],[0.0956466,0.0025432,0.0017259],[0.1385736,0.0025432,0.0017259],[0.1661232,0.0025432,0.0017259],[0.0886938,0.0065293,-0.0174653],[0.1276899,0.0065293,-0.0174653],[0.1542632,0.0065293,-0.0174653],[0.0340736,0.0094198,-0.0229986],[0.0797241,0.0094208,-0.0229964],[0.1104445,0.0094208,-0.0229964],[0.1307558,0.0094208,-0.0229964],[0.1268551,0.0184873,0.0290555],[0.1805905,0.0062914,0.0232550],[0.1910881,0.0014059,0.0014172],[0.1785893,0.0049211,-0.0177232],[0.1526782,0.0082047,-0.0227499],]",
"T 11/06/2022 11:50:41.35516-04:00 // P [0.0000000,0.0000000,0.0000000] // R [0.0000000,0.7071068,0.0000000,0.7071068] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0684713,0.0191577,0.0280285],[0.1022644,0.0191577,0.0280285],[0.0959962,0.0073165,0.0235507],[0.1339235,0.0073165,0.0235506],[0.1582271,0.0073165,0.0235506],[0.0956466,0.0025432,0.0017259],[0.1385736,0.0025432,0.0017259],[0.1661232,0.0025432,0.0017259],[0.0886938,0.0065293,-0.0174653],[0.1276899,0.0065293,-0.0174653],[0.1542632,0.0065293,-0.0174653],[0.0340736,0.0094198,-0.0229986],[0.0797241,0.0094208,-0.0229964],[0.1104445,0.0094208,-0.0229964],[0.1307558,0.0094208,-0.0229964],[0.1268551,0.0184873,0.0290555],[0.1805905,0.0062914,0.0232550],[0.1910881,0.0014059,0.0014172],[0.1785893,0.0049211,-0.0177232],[0.1526782,0.0082047,-0.0227499],]",
"T 11/06/2022 11:50:41.39468-04:00 // P [0.0000000,0.0000000,0.0000000] // R [0.0000000,0.7071068,0.0000000,0.7071068] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0684713,0.0191577,0.0280285],[0.1022644,0.0191577,0.0280285],[0.0959962,0.0073165,0.0235507],[0.1339235,0.0073165,0.0235506],[0.1582271,0.0073165,0.0235506],[0.0956466,0.0025432,0.0017259],[0.1385736,0.0025432,0.0017259],[0.1661232,0.0025432,0.0017259],[0.0886938,0.0065293,-0.0174653],[0.1276899,0.0065293,-0.0174653],[0.1542632,0.0065293,-0.0174653],[0.0340736,0.0094198,-0.0229986],[0.0797241,0.0094208,-0.0229964],[0.1104445,0.0094208,-0.0229964],[0.1307558,0.0094208,-0.0229964],[0.1268551,0.0184873,0.0290555],[0.1805905,0.0062914,0.0232550],[0.1910881,0.0014059,0.0014172],[0.1785893,0.0049211,-0.0177232],[0.1526782,0.0082047,-0.0227499],]",
"T 11/06/2022 11:50:41.47479-04:00 // P [-0.1005967,0.7658275,0.1904638] // R [0.1306013,-0.9180339,0.0564843,-0.3700901] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0629917,0.0333220,0.0392383],[0.0943966,0.0452618,0.0356126],[0.0959962,0.0073165,0.0235507],[0.1235792,0.0328644,0.0185541],[0.1025276,0.0447775,0.0209154],[0.0956466,0.0025432,0.0017259],[0.1192843,0.0382420,-0.0013691],[0.0933814,0.0464906,0.0031010],[0.0886938,0.0065293,-0.0174652],[0.1022415,0.0430964,-0.0174411],[0.0775477,0.0473560,-0.0085970],[0.0340736,0.0094198,-0.0229986],[0.0778958,0.0136912,-0.0350541],[0.0869783,0.0425192,-0.0295591],[0.0689532,0.0447398,-0.0204642],[0.1131523,0.0532959,0.0218326],[0.0861111,0.0298387,0.0238458],[0.0759856,0.0294452,0.0087113],[0.0637362,0.0278083,-0.0039560],[0.0572186,0.0266982,-0.0161137],]",
"T 11/06/2022 11:50:41.55431-04:00 // P [-0.0936752,0.7104396,0.1836446] // R [0.1372674,-0.9225871,0.0366305,-0.3586767] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0629352,0.0330590,0.0396944],[0.0942269,0.0449853,0.0351593],[0.0959962,0.0073165,0.0235507],[0.1262635,0.0293897,0.0176229],[0.1068139,0.0437215,0.0202646],[0.0956466,0.0025432,0.0017259],[0.1222454,0.0360026,-0.0022351],[0.0984167,0.0490532,0.0023313],[0.0886938,0.0065293,-0.0174653],[0.1061396,0.0413756,-0.0189051],[0.0829847,0.0504563,-0.0095491],[0.0340735,0.0094198,-0.0229986],[0.0778958,0.0136912,-0.0350541],[0.0905177,0.0414007,-0.0309780],[0.0735009,0.0470603,-0.0214413],[0.1125337,0.0529764,0.0207645],[0.0879605,0.0321647,0.0237642],[0.0780533,0.0360506,0.0087261],[0.0659604,0.0341375,-0.0033617],[0.0589210,0.0317971,-0.0153923],]",
"T 11/06/2022 11:50:41.63459-04:00 // P [-0.0896030,0.7164592,0.1730236] // R [0.1874898,-0.9159475,0.0530979,-0.3508110] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0609420,0.0370065,0.0387210],[0.0899852,0.0519225,0.0300045],[0.0959962,0.0073165,0.0235507],[0.1294958,0.0243230,0.0183515],[0.1195482,0.0464898,0.0189388],[0.0956466,0.0025432,0.0017259],[0.1254212,0.0327649,-0.0048196],[0.1061922,0.0518257,0.0002708],[0.0886938,0.0065293,-0.0174653],[0.1091470,0.0395743,-0.0206883],[0.0891526,0.0541364,-0.0109767],[0.0340736,0.0094199,-0.0229986],[0.0778958,0.0136912,-0.0350541],[0.0935129,0.0400847,-0.0332557],[0.0792302,0.0501492,-0.0228990],[0.1062448,0.0604886,0.0136198],[0.0978657,0.0515166,0.0213593],[0.0832353,0.0464957,0.0085900],[0.0691981,0.0429452,-0.0025512],[0.0621550,0.0400867,-0.0134484],]",
"T 11/06/2022 11:50:41.69079-04:00 // P [-0.0994052,0.7099468,0.1780024] // R [0.2287566,-0.9049845,0.0516605,-0.3549715] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0608822,0.0374431,0.0381052],[0.0900835,0.0526694,0.0305279],[0.0959962,0.0073165,0.0235507],[0.1309002,0.0213005,0.0185875],[0.1257029,0.0450417,0.0184997],[0.0956466,0.0025432,0.0017259],[0.1284228,0.0292100,-0.0058453],[0.1134537,0.0519390,-0.0015677],[0.0886938,0.0065293,-0.0174652],[0.1112675,0.0380082,-0.0219599],[0.0936694,0.0554685,-0.0123898],[0.0340736,0.0094198,-0.0229986],[0.0778958,0.0136912,-0.0350541],[0.0949320,0.0392520,-0.0346563],[0.0824858,0.0514089,-0.0241751],[0.1073574,0.0619950,0.0156668],[0.1061929,0.0558893,0.0202169],[0.0900028,0.0528781,0.0070232],[0.0726329,0.0477588,-0.0027751],[0.0648957,0.0445516,-0.0129638],]",
"T 11/06/2022 11:50:41.79012-04:00 // P [-0.1065188,0.7222143,0.1822620] // R [0.2957499,-0.8672848,0.0364136,-0.3987770] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0621294,0.0314092,0.0429304],[0.0937640,0.0432742,0.0435976],[0.0959962,0.0073165,0.0235507],[0.1303181,0.0227492,0.0188272],[0.1243919,0.0463191,0.0187589],[0.0956466,0.0025431,0.0017259],[0.1270499,0.0309438,-0.0053423],[0.1115742,0.0533271,-0.0010446],[0.0886938,0.0065293,-0.0174652],[0.1149924,0.0345944,-0.0239014],[0.1015925,0.0558212,-0.0151832],[0.0340735,0.0094198,-0.0229986],[0.0778958,0.0136912,-0.0350540],[0.0997056,0.0349162,-0.0392461],[0.0919364,0.0508763,-0.0293737],[0.1145639,0.0532588,0.0350023],[0.1047429,0.0569360,0.0203262],[0.0880623,0.0547560,0.0073090],[0.0798562,0.0533548,-0.0044192],[0.0748623,0.0501400,-0.0155874],]",
"T 11/06/2022 11:50:41.85527-04:00 // P [-0.1030115,0.7218475,0.1848228] // R [0.3026588,-0.8669915,0.0426575,-0.3935782] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0625916,0.0310741,0.0423730],[0.0943560,0.0425295,0.0410449],[0.0959962,0.0073164,0.0235507],[0.1303470,0.0228319,0.0193352],[0.1246688,0.0464623,0.0191636],[0.0956466,0.0025431,0.0017259],[0.1276247,0.0303626,-0.0050709],[0.1133309,0.0535962,-0.0012152],[0.0886938,0.0065293,-0.0174652],[0.1151860,0.0343764,-0.0240528],[0.1025621,0.0561586,-0.0155490],[0.0340735,0.0094198,-0.0229986],[0.0778958,0.0136912,-0.0350540],[0.1001733,0.0343256,-0.0397094],[0.0930843,0.0507357,-0.0300655],[0.1147531,0.0519902,0.0310119],[0.1051579,0.0573713,0.0204190],[0.0898850,0.0569046,0.0067833],[0.0808206,0.0548206,-0.0045979],[0.0761938,0.0510327,-0.0160387],]",
"T 11/06/2022 11:50:41.93611-04:00 // P [-0.1038071,0.7219247,0.1843624] // R [0.2849893,-0.8774131,0.0620128,-0.3808961] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[0.0200690,0.0115540,0.0104970],[0.0359584,0.0191577,0.0280285],[0.0623303,0.0322120,0.0418560],[0.0938006,0.0444246,0.0402940],[0.0959962,0.0073164,0.0235507],[0.1303615,0.0228670,0.0195903],[0.1246410,0.0464869,0.0193905],[0.0956466,0.0025432,0.0017259],[0.1274801,0.0307032,-0.0043043],[0.1114528,0.0527669,-0.0003933],[0.0886938,0.0065293,-0.0174652],[0.1136106,0.0360733,-0.0226610],[0.0982968,0.0558891,-0.0137744],[0.0340736,0.0094198,-0.0229986],[0.0778958,0.0136912,-0.0350540],[0.0982400,0.0365714,-0.0375738],[0.0885421,0.0511507,-0.0272808],[0.1140711,0.0541644,0.0302714],[0.1051068,0.0573691,0.0205097],[0.0876540,0.0537418,0.0071766],[0.0765516,0.0511718,-0.0038091],[0.0710530,0.0480219,-0.0143783],]"
]

timestamp_r = ["T 11/06/2022 11:50:41.02103-04:00 // P [0.0664348,0.7280709,0.2220922] // R [0.7477179,0.4146775,0.5183367,0.0169588] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0626666,-0.0371129,-0.0326514],[-0.0929192,-0.0504109,-0.0255867],[-0.0959962,-0.0073165,-0.0235507],[-0.1254710,-0.0294875,-0.0147099],[-0.1253211,-0.0537525,-0.0133478],[-0.0956466,-0.0025431,-0.0017259],[-0.1293672,-0.0273727,0.0077150],[-0.1264394,-0.0547322,0.0063488],[-0.0886938,-0.0065293,0.0174652],[-0.1189606,-0.0293540,0.0266111],[-0.1182757,-0.0555541,0.0222261],[-0.0340735,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350541],[-0.1029551,-0.0309631,0.0392318],[-0.1065068,-0.0504989,0.0349552],[-0.1138694,-0.0620582,-0.0199619],[-0.1216880,-0.0757810,-0.0116733],[-0.1190844,-0.0784515,0.0035324],[-0.1111489,-0.0774668,0.0142601],[-0.1049997,-0.0701342,0.0252437],]",
"T 11/06/2022 11:50:41.12986-04:00 // P [0.0676768,0.7286125,0.2230256] // R [0.7411577,0.4220755,0.5220416,0.0031976] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0623604,-0.0372717,-0.0336771],[-0.0926007,-0.0511640,-0.0278041],[-0.0959962,-0.0073165,-0.0235507],[-0.1259093,-0.0290143,-0.0150132],[-0.1249673,-0.0532741,-0.0138989],[-0.0956466,-0.0025432,-0.0017259],[-0.1300459,-0.0258685,0.0090137],[-0.1271575,-0.0532283,0.0075719],[-0.0886938,-0.0065293,0.0174652],[-0.1197725,-0.0277176,0.0277540],[-0.1195919,-0.0539462,0.0234912],[-0.0340735,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350541],[-0.1036776,-0.0293711,0.0408144],[-0.1082024,-0.0488157,0.0370746],[-0.1135278,-0.0632394,-0.0230658],[-0.1195433,-0.0749658,-0.0127561],[-0.1190626,-0.0766285,0.0041753],[-0.1119387,-0.0755125,0.0150807],[-0.1070955,-0.0684662,0.0273403],]",
"T 11/06/2022 11:50:41.19765-04:00 // P [0.0676236,0.7287174,0.2234510] // R [0.7415838,0.4222171,0.5213225,0.0029953] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0624083,-0.0370011,-0.0342821],[-0.0927934,-0.0508310,-0.0290433],[-0.0959962,-0.0073164,-0.0235507],[-0.1262548,-0.0285414,-0.0150420],[-0.1260073,-0.0528105,-0.0137721],[-0.0956466,-0.0025431,-0.0017259],[-0.1306686,-0.0250140,0.0088214],[-0.1283243,-0.0524346,0.0075590],[-0.0886938,-0.0065293,0.0174652],[-0.1202473,-0.0268661,0.0280237],[-0.1207236,-0.0531302,0.0240098],[-0.0340735,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350540],[-0.1037766,-0.0289969,0.0413518],[-0.1086020,-0.0484100,0.0378304],[-0.1138432,-0.0628831,-0.0248173],[-0.1212050,-0.0746397,-0.0124734],[-0.1205350,-0.0759589,0.0043081],[-0.1134323,-0.0748633,0.0157077],[-0.1077541,-0.0681046,0.0281594],]",
"T 11/06/2022 11:50:41.27179-04:00 // P [0.0701121,0.7307886,0.2306595] // R [0.7410173,0.4258713,0.5191134,-0.0069432] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0622993,-0.0368904,-0.0350142],[-0.0926974,-0.0508364,-0.0301736],[-0.0959962,-0.0073164,-0.0235507],[-0.1264393,-0.0284632,-0.0155189],[-0.1252469,-0.0527163,-0.0145023],[-0.0956466,-0.0025431,-0.0017259],[-0.1313003,-0.0237760,0.0092606],[-0.1318892,-0.0513174,0.0089370],[-0.0886938,-0.0065293,0.0174652],[-0.1210427,-0.0237997,0.0307308],[-0.1250449,-0.0499711,0.0284542],[-0.0340736,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350541],[-0.1044583,-0.0258002,0.0446223],[-0.1120301,-0.0445952,0.0432203],[-0.1137577,-0.0629856,-0.0262916],[-0.1189039,-0.0741679,-0.0135789],[-0.1270482,-0.0757282,0.0066325],[-0.1211425,-0.0728973,0.0211349],[-0.1145034,-0.0648449,0.0350986],]",
"T 11/06/2022 11:50:41.35516-04:00 // P [0.0830530,0.9035315,0.2660670] // R [0.7254853,0.5253495,0.4134098,-0.1636197] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0621827,-0.0366500,-0.0359906],[-0.0926673,-0.0507158,-0.0321422],[-0.0959962,-0.0073165,-0.0235507],[-0.1272146,-0.0275545,-0.0161812],[-0.1272197,-0.0518270,-0.0149497],[-0.0956466,-0.0025431,-0.0017259],[-0.1320877,-0.0231492,0.0077694],[-0.1356096,-0.0504694,0.0081942],[-0.0886938,-0.0065293,0.0174652],[-0.1214162,-0.0220492,0.0319244],[-0.1279497,-0.0477940,0.0311128],[-0.0340735,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350541],[-0.1044308,-0.0228188,0.0475570],[-0.1140780,-0.0406723,0.0484149],[-0.1137988,-0.0630203,-0.0292670],[-0.1219794,-0.0735576,-0.0136940],[-0.1340541,-0.0753887,0.0070783],[-0.1270830,-0.0714252,0.0251773],[-0.1195942,-0.0610421,0.0423512],]",
"T 11/06/2022 11:50:41.39468-04:00 // P [0.1084474,0.9512268,0.2561110] // R [0.5582925,0.6479856,0.1892619,-0.4822905] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0627838,-0.0148153,-0.0458788],[-0.0920616,-0.0071590,-0.0609172],[-0.0959962,-0.0073164,-0.0235506],[-0.1257905,-0.0297847,-0.0167716],[-0.1206961,-0.0535470,-0.0165150],[-0.0956466,-0.0025432,-0.0017259],[-0.1055689,-0.0442872,-0.0004189],[-0.0809487,-0.0557992,-0.0049243],[-0.0886938,-0.0065293,0.0174653],[-0.0948282,-0.0450128,0.0160214],[-0.0702547,-0.0457105,0.0059316],[-0.0340735,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350540],[-0.0815293,-0.0435321,0.0287250],[-0.0653436,-0.0399155,0.0169992],[-0.1112356,-0.0022000,-0.0755452],[-0.1098966,-0.0731586,-0.0163956],[-0.0566540,-0.0590867,-0.0097815],[-0.0516956,-0.0321541,-0.0022045],[-0.0544476,-0.0229544,0.0082971],]",
"T 11/06/2022 11:50:41.47479-04:00 // P [0.1079774,0.9679714,0.2648568] // R [0.5330465,0.6864437,0.0189967,-0.4942627] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0637558,-0.0223205,-0.0445931],[-0.0945493,-0.0221363,-0.0585108],[-0.0959962,-0.0073164,-0.0235507],[-0.1104161,-0.0418874,-0.0176017],[-0.0873244,-0.0477938,-0.0223507],[-0.0956466,-0.0025432,-0.0017259],[-0.1039886,-0.0445862,0.0006224],[-0.0780449,-0.0492422,-0.0073916],[-0.0886938,-0.0065293,0.0174653],[-0.0910702,-0.0453475,0.0146025],[-0.0659852,-0.0427375,0.0062315],[-0.0340735,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350540],[-0.0768296,-0.0430863,0.0261925],[-0.0583135,-0.0388875,0.0189762],[-0.1147393,-0.0231842,-0.0725633],[-0.0664415,-0.0410174,-0.0267375],[-0.0583396,-0.0361144,-0.0153916],[-0.0513563,-0.0239833,0.0008753],[-0.0464285,-0.0210794,0.0141032],]",
"T 11/06/2022 11:50:41.55431-04:00 // P [0.1015237,0.9703714,0.2719680] // R [0.6180015,0.6604800,0.0065545,-0.4263772] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0640673,-0.0198827,-0.0443519],[-0.0951608,-0.0174740,-0.0573659],[-0.0959962,-0.0073165,-0.0235507],[-0.1093838,-0.0422428,-0.0172737],[-0.0864486,-0.0482090,-0.0226630],[-0.0956466,-0.0025432,-0.0017259],[-0.1044844,-0.0444508,0.0011669],[-0.0794871,-0.0521427,-0.0074905],[-0.0886938,-0.0065293,0.0174652],[-0.0905663,-0.0453554,0.0143472],[-0.0651295,-0.0458080,0.0066723],[-0.0340736,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350541],[-0.0766499,-0.0430691,0.0261588],[-0.0575024,-0.0425700,0.0194004],[-0.1158582,-0.0168837,-0.0706878],[-0.0660674,-0.0404452,-0.0277214],[-0.0583156,-0.0424953,-0.0166190],[-0.0477883,-0.0296317,0.0010135],[-0.0416721,-0.0285289,0.0135373],]",
"T 11/06/2022 11:50:41.63459-04:00 // P [0.1012701,0.9639614,0.2659318] // R [0.6321665,0.6425007,0.0576578,-0.4292249] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0640914,-0.0173689,-0.0442280],[-0.0953607,-0.0128970,-0.0562365],[-0.0959962,-0.0073164,-0.0235506],[-0.1142098,-0.0400632,-0.0176861],[-0.0910112,-0.0459745,-0.0218752],[-0.0956466,-0.0025431,-0.0017259],[-0.1060991,-0.0440959,0.0008903],[-0.0802383,-0.0501528,-0.0064247],[-0.0886938,-0.0065293,0.0174652],[-0.0935078,-0.0451612,0.0152079],[-0.0685034,-0.0473942,0.0064937],[-0.0340736,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350541],[-0.0801436,-0.0432998,0.0271787],[-0.0618552,-0.0452439,0.0185587],[-0.1156423,-0.0102846,-0.0699493],[-0.0710390,-0.0366420,-0.0257832],[-0.0598523,-0.0378912,-0.0140861],[-0.0507598,-0.0319949,-0.0000210],[-0.0456353,-0.0326123,0.0108459],]",
"T 11/06/2022 11:50:41.69079-04:00 // P [0.1004411,0.9625131,0.2625365] // R [0.6295288,0.6390297,0.0728053,-0.4359288] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0639473,-0.0178177,-0.0445184],[-0.0953532,-0.0140795,-0.0564207],[-0.0959962,-0.0073165,-0.0235507],[-0.1151426,-0.0396314,-0.0182932],[-0.0920724,-0.0465116,-0.0216241],[-0.0956466,-0.0025431,-0.0017259],[-0.1070163,-0.0438466,0.0010095],[-0.0810547,-0.0496931,-0.0061170],[-0.0886938,-0.0065293,0.0174652],[-0.0947557,-0.0450155,0.0158049],[-0.0701012,-0.0475233,0.0062128],[-0.0340735,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350540],[-0.0810957,-0.0433464,0.0277008],[-0.0632973,-0.0453827,0.0181288],[-0.1153458,-0.0117700,-0.0706041],[-0.0718369,-0.0375000,-0.0248759],[-0.0614030,-0.0361657,-0.0135639],[-0.0527052,-0.0319795,-0.0008712],[-0.0477228,-0.0324005,0.0097012],]",
"T 11/06/2022 11:50:41.79012-04:00 // P [0.0962062,0.9614331,0.2584665] // R [0.6289590,0.6269546,0.0857417,-0.4516491] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280284],[-0.0637341,-0.0173774,-0.0448342],[-0.0956913,-0.0142956,-0.0553798],[-0.0959962,-0.0073165,-0.0235507],[-0.1158708,-0.0392392,-0.0186091],[-0.0927646,-0.0461676,-0.0215707],[-0.0956466,-0.0025431,-0.0017259],[-0.1076908,-0.0436623,0.0008965],[-0.0815125,-0.0490727,-0.0057670],[-0.0886938,-0.0065293,0.0174652],[-0.0944082,-0.0450619,0.0156539],[-0.0695560,-0.0459479,0.0062877],[-0.0340735,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350540],[-0.0810268,-0.0433225,0.0275759],[-0.0629156,-0.0446254,0.0184745],[-0.1163412,-0.0123323,-0.0686437],[-0.0735555,-0.0350666,-0.0245758],[-0.0626287,-0.0342655,-0.0127516],[-0.0535089,-0.0287356,-0.0000872],[-0.0483418,-0.0298230,0.0113594],]",
"T 11/06/2022 11:50:41.85527-04:00 // P [0.0981949,0.9628642,0.2561931] // R [0.6400725,0.6158548,0.0956224,-0.4493177] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0639439,-0.0173494,-0.0444795],[-0.0958652,-0.0139132,-0.0550242],[-0.0959962,-0.0073164,-0.0235506],[-0.1151895,-0.0396772,-0.0187677],[-0.0923430,-0.0475146,-0.0214657],[-0.0956466,-0.0025431,-0.0017259],[-0.1077983,-0.0436536,0.0005082],[-0.0814191,-0.0489365,-0.0054256],[-0.0886938,-0.0065293,0.0174653],[-0.0959722,-0.0448087,0.0159144],[-0.0710929,-0.0473515,0.0069310],[-0.0340736,-0.0094198,0.0229985],[-0.0778958,-0.0136912,0.0350540],[-0.0814832,-0.0433016,0.0276992],[-0.0635046,-0.0458521,0.0185989],[-0.1164713,-0.0117636,-0.0683273],[-0.0726524,-0.0372605,-0.0243623],[-0.0621584,-0.0343591,-0.0118409],[-0.0538709,-0.0313382,0.0004974],[-0.0481915,-0.0319721,0.0111840],]",
"T 11/06/2022 11:50:41.93611-04:00 // P [0.0991718,0.9632863,0.2551839] // R [0.6374398,0.6222575,0.0981480,-0.4436589] // H [[0.0000000,0.0000000,0.0000000],[0.0000000,0.0000000,0.0000000],[-0.0200690,-0.0115540,-0.0104970],[-0.0359584,-0.0191577,-0.0280285],[-0.0638463,-0.0181053,-0.0447093],[-0.0955617,-0.0151567,-0.0559967],[-0.0959962,-0.0073164,-0.0235507],[-0.1149163,-0.0398849,-0.0191005],[-0.0919173,-0.0473705,-0.0214831],[-0.0956466,-0.0025432,-0.0017260],[-0.1080992,-0.0435587,0.0005940],[-0.0817513,-0.0489464,-0.0053842],[-0.0886938,-0.0065293,0.0174652],[-0.0958330,-0.0448363,0.0159478],[-0.0710054,-0.0472427,0.0067856],[-0.0340736,-0.0094198,0.0229986],[-0.0778958,-0.0136912,0.0350541],[-0.0822359,-0.0432870,0.0280544],[-0.0645504,-0.0464533,0.0185809],[-0.1156674,-0.0132790,-0.0700837],[-0.0722012,-0.0370792,-0.0240569],[-0.0623066,-0.0346360,-0.0118457],[-0.0540377,-0.0309821,0.0002970],[-0.0492036,-0.0328177,0.0107915],]",
]

#making sure both arrays have the same length
print(len(timestamp_r))
print(len(timestamp_l))

#processing and representing all of them
for frame in range(len(timestamp_l)):
    oc_kps_l = getOculusKpsFromManualTimestamp(timestamp_l[frame])
    oc_kps_r = getOculusKpsFromManualTimestamp(timestamp_r[frame])

    compare3DKps(oc_kps_l[0], oc_kps_r[0],label_kps1="Left", label_kps2="Right", save_img=1, saving_path= str(frame)+ ".png", show = 0) 