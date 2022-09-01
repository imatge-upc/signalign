import pickle
import numpy as np
import json
import argparse
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report, confusion_matrix

"""
This script trains RL models to detect the thumbs up and tests them.
"""

def loadFeatures(args):
    #Load the keypoints (with the calculated angles)

    with open(args.front_kp, "r", encoding="utf8") as f:
        front = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")

    with open(args.head_kp, "r", encoding="utf8") as f:
        head = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")

    with open(args.inside_kp, "r", encoding="utf8") as f:
        inside = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")

    with open(args.side_kp, "r", encoding="utf8") as f:
        side = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")

    return front,head,inside,side

def labelingData(Kps):
    setKps = Kps
    t = np.zeros(len(setKps[0])).tolist()

    #labeling to 1 the frames that have the thumbs up
    for i in range(len(setKps[0])):
        if (i >= 393 and i <=494) or i >= 2535:
            t[i] = 1

    #removing some misleading frames (right after or before the thumbs up)
    for i in range(2534, 2524, -1):
        setKps[0].pop(i)
        t.pop(i)

    for i in range(495, 506):
        setKps[0].pop(i)
        t.pop(i)

    for i in range(392, 381, -1):
        setKps[0].pop(i)
        t.pop(i)

    return setKps, t

def trainRLmodel(args, Kps):

    setKps, t = labelingData(Kps)
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(np.array(setKps[0], dtype=np.float32), np.array(t, dtype=np.float32))
    pickle.dump(model, open(args.saveRL_path, 'wb'))

def testRLmodel(args, setKps):
    model = pickle.load(open(args.saveRL_path, 'rb'))
    frames_pred = []
    for i in range(len(setKps[0])):
        if model.predict([setKps[0][i]])[0] == 1:
            frames_pred.append(i)

    print(frames_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--front_kp', type=str, default='../front_fZbAxSSbX4.json', help='path where the json file with the front kps is stored')
    parser.add_argument('--head_kp', type=str, default='../head_fZbAxSSbX4.json', help='path where the json file with the head kps is stored')
    parser.add_argument('--side_kp', type=str, default='../side_fZbAxSSbX4.json', help='path where the json file with the side kps is stored')
    parser.add_argument('--inside_kp', type=str, default='../inside_fZbAxSSbX4.json', help='path where the json file with the inside kps is stored')

    parser.add_argument('--saveRL_path', type=str, default='../sideRL.pkl', help='path where the RL model is to be saved or loaded')
    parser.add_argument('--train', type=bool, default = False, help='wether to train (true) or test (false) the model')   

    args = parser.parse_args()
    kp_front,kp_head,kp_inside,kp_side = loadFeatures(args)

    #side model
    #use different videos for training/testing
    if args.train:
        trainRLmodel(args,kp_side)
    else:
        testRLmodel(args, kp_side)
