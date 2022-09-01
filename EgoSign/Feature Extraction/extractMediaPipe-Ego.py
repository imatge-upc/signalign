#!/usr/bin/env python3
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import h5py

from joblib import Parallel, delayed

import imageio
import cv2  # for reading .mp4 video
import mediapipe as mp

import numpy as np

from google.protobuf.json_format import MessageToDict


DATA_DIR_HOW2SIGN = {
    'train': '/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/train/rgb_front/raw_videos',
    'val': '/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/val/rgb_front/raw_videos',
    'test': '/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/test/rgb_front/raw_videos',
}

DATA_DIR_PHOENIX = {
    'train': '/mnt/gpid07/datasets/language/signs/phoenix2014T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-227x227px/train',
    'val': '/mnt/gpid07/datasets/language/signs/phoenix2014T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-227x227px/dev',
    'test': '/mnt/gpid07/datasets/language/signs/phoenix2014T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-227x227px/test',
}
DATA_DIR_EGOSIGN = {
    'train': '/mnt/gpid08/datasets/How2Sign/EgoSign/egoSign/raw_data/PC2/raw_videos_grouped/rgb_head/train',
    'val': '//mnt/gpid08/datasets/How2Sign/EgoSign/egoSign/raw_data/PC2/raw_videos_grouped/rgb_head/val',
    'test': '/mnt/gpid08/datasets/How2Sign/EgoSign/egoSign/raw_data/PC2/raw_videos_grouped/rgb_head/test',
}

DATA_DIR = {
    'How2Sign': DATA_DIR_HOW2SIGN,
    'Phoenix': DATA_DIR_PHOENIX,
    'EgoSign': DATA_DIR_EGOSIGN,
}



# SPLITS = ['val', 'test', 'train']
SPLITS = ['test']

def landmarkList_to_list(landmarkList, l=None):
    l = [] if l is None else l
    for data_point in landmarkList:
        l += [data_point.x, data_point.y, data_point.z, data_point.visibility]
    return l


def results_to_array(results, results2):
    
    arr = []

    if results.face_landmarks is not None:
        arr += landmarkList_to_list(results.face_landmarks.landmark)
    else:
        arr += [-1] * (468 * 4)

    #print(len(arr), flush = True)

    if results.pose_world_landmarks is not None:
        arr += landmarkList_to_list(results.pose_world_landmarks.landmark)
    else:
        arr += [-1] * (33 * 4)

    #print(len(arr), flush = True)

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

    #print(len(arr), flush = True)

    return np.array(arr)


def add_landmarks_to_image(mp_holistic, mp_drawing, mp_drawing_styles, results, image):
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def process_video(filepath: str, split: str, dataset: str = 'How2Sign', animation: bool = True):
    if dataset == 'How2Sign' or dataset == 'EgoSign':
        video = cv2.VideoCapture(filepath)
    if dataset == 'Phoenix':
        video = cv2.VideoCapture(os.path.join(filepath, 'images%04d.png'), cv2.CAP_IMAGES)

    video_landmarks = []  # 4x543 = 2172 numbers per frame (543 landmarks per frame: 33 pose, 468 face, 21 per hand)
    image_lst = []
    #print(filepath)
    #if filepath == "/mnt/gpid08/datasets/How2Sign/How2Sign/video_level/train/rgb_front/raw_videos/0pryZawZCJI-5-rgb_front.mp4":
    mp_hands = mp.solutions.hands
    mp_holistic = mp.solutions.holistic

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    i = 0
    with (mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands,
        mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2,
        smooth_landmarks=True) as holistic):

        while video.isOpened():
            i += 1
            success, image = video.read()
            if not success:
                print(f"Ignoring empty camera frame at frame {i}.", flush=True)
                # If loading a video, use 'break' instead of 'continue'.
                break
            
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #holistic results
            results = holistic.process(image)

            #hands results
            results2 = hands.process(image)

            KPs_frame = results_to_array(results, results2)
            video_landmarks.append(KPs_frame)

            if animation:
                image = add_landmarks_to_image(mp_holistic, mp_drawing, mp_drawing_styles, results, image)
                image_lst.append(image)

    video.release()
    if animation:
        print(f' --> Generating .gif animation for {filepath.split("/")[-1]}', flush=True)
        os.makedirs(f'./gif_animation_{split}', exist_ok=True)
        imageio.mimsave(f'./gif_animation_{split}/{filepath.split("/")[-1].replace(".mp4", "")}.gif',
                        image_lst, fps=25)
    video_landmarks = np.vstack(video_landmarks) if len(video_landmarks) else np.array([-1]*2172)
    return video_landmarks


def MediaPipe_to_h5(dataset: str = 'How2Sign'):
    def _process(path, vid_name, idx, split):
        print(f' --> Processing video {vid_name}', flush=True)
        video_landmarks = process_video(join(path, vid_name), split, dataset, False)
        return (vid_name, video_landmarks)

    for split in SPLITS:
        print(flush=True)
        print(f' --> Processing {split} split', flush=True)
        print(flush=True)

        path = DATA_DIR[dataset][split]
        if dataset == 'How2Sign' or dataset == 'EgoSign':
            pathlist = [(i, f) for i, f in enumerate(listdir(path)) if isfile(join(path, f)) and f[-4:] == '.mp4']
        if dataset == 'Phoenix':
            pathlist = [(i, f) for i, f in enumerate(listdir(path)) if isdir(join(path, f))]

        partition = 60
        r = len(pathlist) // partition
        start = 0
        for p in range(start, partition):
            print(f' --> Processing partition {p}', flush=True)
            # path = './'
            _pathlist = pathlist[p*r:(p+1)*r] if p < partition-1 else pathlist[p*r:]
            # files = [(0,'G3g0-BeFN3c_17-5-rgb_front.mp4'), (1,'G42xKICVj9U_4-10-rgb_front.mp4')]
            res = Parallel(n_jobs=-1)(delayed(_process)(path, vid[1], vid[0], split) for vid in _pathlist)

            h5f = h5py.File(f'/mnt/gpid08/datasets/How2Sign/EgoSign/egoSign/raw_data/PC2/features/rgb_head/MediaPipe_{split}_worldlmarks.h5', 'a')
            for (vid_name, video_landmarks) in res:
                try:
                    h5f.create_dataset(f'{vid_name.replace(".mp4", "")}', data=video_landmarks)
                except:
                    print(f'Skipping video {vid_name}', flush = True)
                    
            h5f.close()
            print(f' --> Finished partition {p}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='EgoSign', help='dataset to be used')
    args = parser.parse_args()
    MediaPipe_to_h5(args.dataset)

    # hfIn = h5py.File('MediaPipe_test.h5', "r")
    # print(list(hfIn.keys()))
    # print(hfIn[list(hfIn.keys())[0]])
