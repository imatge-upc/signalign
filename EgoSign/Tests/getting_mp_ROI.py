import cv2
import mediapipe as mp
import cv2
import os
import json

"""
This script is to test if mediapipe features can be also extracted from a ROI of the video that has all 4 points of view.
"""
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles



cap = cv2.VideoCapture("test1-20220510T170226Z-001/test1/sync_videos/2022-05-06_11-29-24_total.mkv")

#the names of the rois that are to be generated (3)
video_name = ["test1-rgb_front", "test1-rgb_head", "test1-rgb_side" ]

indx = 0
first_iter = 1
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if ret == False:        
            break

        if first_iter:        
            #select ROIs function
            cv2.namedWindow("Select Rois", cv2.WINDOW_NORMAL)   
            ROIs = cv2.selectROIs("Select Rois",frame)
            #print rectangle points of selected roi
            print(ROIs)
            first_iter = 0
            cv2.destroyAllWindows()

            keypoints = dict()
            keypoints_unknwn = dict()
            for vname in video_name:
                keypoints[vname]=dict()
                keypoints[vname]["right"] = dict()
                keypoints[vname]["left"] = dict()

                keypoints_unknwn[vname] = dict()
                try:
                    os.mkdir("Keypoints")
                except:
                    try:
                        os.mkdir("Keypoints/Video4escenas")
                    except:
                        try:
                            os.mkdir("Keypoints/Video4escenas/" + vname)
                        except:
                            pass
            
        #process each roi (should be 3: video_name )
        for rect, vname in zip(ROIs, video_name):
            x1=rect[0]
            y1=rect[1]
            x2=rect[2]
            y2=rect[3]
        
            #crop roi from original image
            img_crop=frame[y1:y1+y2,x1:x1+x2]
            results = hands.process(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks is not None:
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img_crop, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                
                if indx == 420:
                    cv2.imwrite("Frame420_cropped.png", img_crop) 

                left_image = 1

                for hand_world_landmarks in results.multi_hand_world_landmarks:

                    kp_array = []
                    if indx == 420:
                        for data_point in hand_world_landmarks.landmark:
                            kp_array.append({
                                            'X': data_point.x,
                                            'Y': data_point.y,
                                            'Z': data_point.z,
                                            'Visibility': data_point.visibility,
                                            })
                        print("\n\n")
                        print(kp_array)

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
                        else:
                            
                            keypoints[vname]["right"][indx] = []
                            for data_point in hand_world_landmarks.landmark:
                                
                                keypoints[vname]["right"][indx].append({
                                            'X': data_point.x,
                                            'Y': data_point.y,
                                            'Z': data_point.z,
                                            'Visibility': data_point.visibility,
                                            })
                    else:
                        keypoints_unknwn[vname][indx] = []
                        for data_point in hand_world_landmarks.landmark:     
                            keypoints_unknwn[vname][indx].append({
                                        'X': data_point.x,
                                        'Y': data_point.y,
                                        'Z': data_point.z,
                                        'Visibility': data_point.visibility,
                                        })
        if indx == 420:
            cv2.imwrite("Frame420.png", frame)

        indx+=1

     
    #save kps
    json_object = json.dumps(keypoints, indent = 11, ensure_ascii= False) 
    with open('Keypoints/Video4escenas/Video4escenas_mp_kps.json', "w") as outfile:
        outfile.write(json_object)

    json_object = json.dumps(keypoints_unknwn, indent = 11, ensure_ascii= False) 
    with open('Keypoints/Video4escenas/Video4escenas_mp_kps_1hand.json', "w") as outfile:
        outfile.write(json_object)



cv2.destroyAllWindows()

