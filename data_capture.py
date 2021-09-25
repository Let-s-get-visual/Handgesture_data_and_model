import numpy as np
import pandas as pd
import time

import cv2
import mediapipe as mp

import processing_df_image as pdi
import os

# Media_pipe requirements
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# list of capturing gestures
list_gestures = ["dog", 'fish', 'rabbit']

# variables needed for the interface
max_frames = 1
train_count = 10
frame_count = max_frames
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2
window_name = 'gesture capture'


# input user name
user_name = input("Input your user name: ")

# create needed directories

base = [f'images_{user_name}']


for base in base:
    if not os.path.exists(base):
        os.makedirs(base)

    for gest in list_gestures:
        if not os.path.exists(base + '/' + gest):
            print("all's good")
            os.makedirs(base + '/' + gest)
        print(gest)

if not os.path.exists('csv_data'):
    os.makedirs('csv_data')

# Access to webcam
vc = cv2.VideoCapture(0)
if vc.isOpened():
    response, frame = vc.read()

else:
    response = False


# To access the individual points of the hand tracking
image_height, image_width, _ = frame.shape

dict_list = []

for gesture in list_gestures:

    waiting_time = time.time()
    while (time.time() - waiting_time) < 5:
        if response:
            response, frame = vc.read()
            if response:

                wait_image = cv2.putText(frame, 'Prepare for: ' + gesture, org, font, fontScale, color, thickness, cv2.LINE_AA)
                wait_image = cv2.putText(frame, str(5 - int((time.time() - waiting_time)//1)), (50, 100), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.imshow(window_name, wait_image)
                cv2.waitKey(1)

    vc.release()
    # cv2.destroyAllWindows()

    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        response, frame = vc.read()

    else:
        response = False

    timeout = time.time() + 120
    while time.time() < timeout:
        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
            while response:
                response, frame = vc.read()
                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                list_landmarks = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        h, w, d = image.shape
                        blank = np.zeros((h, w, d), dtype=np.uint8)
                        list_land = hand_landmarks.landmark
                        mp_drawing.draw_landmarks(blank, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        crop = pdi.bbox_landmarks(hand_landmarks, blank)
                        pdi.bbox_landmarks(hand_landmarks, image)
                        loc_dict = pdi.landmark_xy(hand_landmarks, image)
                try:
                    dict_list.append(loc_dict)
                except:
                    continue

                image = cv2.putText(image, gesture, org, font, fontScale, color, thickness, cv2.LINE_AA)

                cv2.imshow(window_name, image)
                try:
                    if frame_count == 0:
                        pdi.frame_folder(crop, user_name, gesture)
                        frame_count = max_frames
                except:
                    continue

                key = cv2.waitKey(1)
                if key == 27 or time.time() > timeout:  # exit on ESC
                    df_loc = pd.DataFrame(dict_list)
                    df_loc.to_csv('csv_data/' + str(time.time()) + '_' + gesture + '_' + user_name + '.csv', index=False)
                    break
                frame_count -= 1

vc.release()
cv2.destroyAllWindows()

