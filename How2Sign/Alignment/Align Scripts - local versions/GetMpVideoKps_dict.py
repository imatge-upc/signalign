import cv2
import mediapipe as mp
import json

def save_json(path, video):
    json_object = json.dumps(video, ensure_ascii= False) 
    with open(path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#path to the 50 frames/s videos
videos = ["../37ZtKNf6Yd8-1-rgb_front_50fotogramas.mp4", "../G2Go6a76xd0-5-rgb_front_50fotogramas.mp4", "../G23JltC2N8g-5-rgb_front_50fotogramas.mp4","../G25fic3QxDk-5-rgb_front_50fotogramas.mp4" ]

#path to save the extracted features
videos_save_path = ["../37ZtKNf6Yd8.json", "../G2Go6a76xd0.json", "../G23JltC2N8g.json","../G25fic3QxDk.json"]

for video, saving_path in zip(videos, videos_save_path):
    #load video
    cap = cv2.VideoCapture(video)
    indx = 0
    keypoints = dict()
    #mp
    with mp_holistic.Holistic(
        static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
            smooth_landmarks=True) as holistic:

        while True:
            ret, frame = cap.read()
            if ret == False:
                break    

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb) 
            #mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
            # save features in dict format            
            keypoints[indx] = []
            for data_point in results.pose_world_landmarks.landmark:
                keypoints[indx].append({
                            'X': data_point.x,
                            'Y': data_point.y,
                            'Z': data_point.z,
                            'Visibility': data_point.visibility,
                            })

            indx+=1
    #save the extracted features
    save_json(saving_path, keypoints)
    cv2.destroyAllWindows()