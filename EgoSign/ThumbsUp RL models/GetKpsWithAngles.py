import math
import mediapipe as mp
import json
import cv2
import argparse

from google.protobuf.json_format import MessageToDict


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#obtains the angles alfa beta and gamma between 2 kp
def getAnglesfromKps(A = [0,1,2], B = [0,1,2]):
    
    alfa = math.atan2(B[1]-A[1], B[0]-A[0]) 
    beta = math.atan2(B[2]-A[2], B[0]-A[0])
    gamma = math.atan2(B[1]-A[1], B[2]-A[2])

    return alfa,beta,gamma

#stack the keypoints
def landmarkList_to_list(landmarkList, l=None):
    l = [] if l is None else l
    tip = landmarkList[3]
    for data_point in landmarkList:
        a,b,c = getAnglesfromKps([tip.x, tip.y, tip.z],[data_point.x, data_point.y, data_point.z])
        l += [data_point.x, data_point.y, data_point.z, data_point.visibility, a,b,c]
    return l

#extract hand keypoints
def getMpHands(image, hands):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #hands results
    results2 = hands.process(image)
    arr = []
    if results2.multi_hand_world_landmarks is not None:
        handedness = []
        buffer_right = None
        buffer_left = None
        for idx, hand_handedness in enumerate(results2.multi_handedness):
            handedness_dict = MessageToDict(hand_handedness)
            handedness.append(handedness_dict["classification"][0]["label"])

        if len(handedness) == 2:
            for hand, hand_world_landmarks in zip(handedness, results2.multi_hand_world_landmarks):
                if hand == "Left":
                    if buffer_left is None:
                        buffer_left = hand_world_landmarks.landmark
                else:
                    if buffer_right is None:
                        buffer_right = hand_world_landmarks.landmark
            
            if buffer_left is None:
                arr += [-1] * (21 * 4)
            else:
                arr += landmarkList_to_list(buffer_left)

            if buffer_right is None:
                arr += [-1] * (21 * 4)
            else:
                arr += landmarkList_to_list(buffer_right)

        elif len(handedness) == 1:
            for hand, hand_world_landmarks in zip(handedness, results2.multi_hand_world_landmarks):
                if hand == "Left":
                    arr += landmarkList_to_list(hand_world_landmarks.landmark)
                    arr += [-1] * (21 * 4)
                else:
                    arr += [-1] * (21 * 4) 
                    arr += landmarkList_to_list(hand_world_landmarks.landmark)

        else:
            arr += [-1] * (21 * 4)
            arr += [-1] * (21 * 4) 

    else:
        arr += [-1] * (21 * 4)
        arr += [-1] * (21 * 4)  


    return arr
    
def main(args):
    #load the videos
    cap = cv2.VideoCapture(args.front_video)
    cap2 = cv2.VideoCapture(args.head_video)
    cap3 = cv2.VideoCapture(args.inside_video)
    cap4 = cv2.VideoCapture(args.side_video)

    video_landmarks = []
    video_landmarks2 = []
    video_landmarks3 = []
    video_landmarks4 = []

    #process them
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            ret2, frame2 = cap2.read()
            ret3, frame3 = cap3.read()
            ret4, frame4 = cap4.read()
            #when one video is finished finish them all (the thumbs up will already be collected)
            if ret == False or ret2 == False or ret3 == False or ret4 == False :
                break

            video_landmarks.append(getMpHands(frame, hands))
            video_landmarks2.append(getMpHands(frame2, hands))
            video_landmarks3.append(getMpHands(frame3, hands))
            video_landmarks4.append(getMpHands(frame4, hands))

        #save the extracted features
        json_object = json.dumps(video_landmarks) 
        with open(args.front_kp, "w") as outfile:
            outfile.write(json_object)
            
        json_object = json.dumps(video_landmarks2) 
        with open(args.head_kp, "w") as outfile:
            outfile.write(json_object)

        json_object = json.dumps(video_landmarks3) 
        with open(args.inside_kp, "w") as outfile:
            outfile.write(json_object)

        json_object = json.dumps(video_landmarks4) 
        with open(args.side_kp.json, "w") as outfile:
            outfile.write(json_object)  



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--front_video', type=str, default='../_g0fpC8aiME/2022-05-09_14-15-37_rgb_front.mkv', help='path where the rgb front video is stored')
    parser.add_argument('--head_video', type=str, default='../_g0fpC8aiME/2022-05-09_14-15-37_rgb_head.mkv', help='path where the rgb head video is stored')
    parser.add_argument('--side_video', type=str, default='../_g0fpC8aiME/2022-05-09_14-15-37_rgb_side.mkv', help='path where the rgb side video is stored')
    parser.add_argument('--inside_video', type=str, default='../_g0fpC8aiME/2022-05-09_14-15-29_inside.mkv', help='path where the inside video is stored')

    parser.add_argument('--front_kp', type=str, default='../front_g0fpC8aiME.json', help='path where the json file with the front kps is to be stored')
    parser.add_argument('--head_kp', type=str, default='../head_g0fpC8aiME.json', help='path where the json file with the head kps is to be stored')
    parser.add_argument('--side_kp', type=str, default='../side_g0fpC8aiME.json', help='path where the json file with the side kps is to be stored')
    parser.add_argument('--inside_kp', type=str, default='../inside_g0fpC8aiME.json', help='path where the json file with the inside kps is to be stored')


    args = parser.parse_args()
    main(args)
