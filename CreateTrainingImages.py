import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os
import time
import sys

def snapshot_countdown(sleep_time):
    for remaining in range(sleep_time, 0, -1):
        sys.stdout.write("\r")
        sys.stdout.write("{:2d} seconds remaining.".format(remaining))
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\rReady!            \n")


def SnapshotImage(image, alpha, n_iter):
    image = cv2.resize(image, (200,200))
    # print(image)
    print('trainingimages/' + alpha + '/' + str(len(os.listdir('trainingimages/' + alpha))))
    #TODO: save image in directory
    # for i in range(n_iter):
    cv2.imwrite('trainingimages/' + alpha + '/' + str(len(os.listdir('trainingimages/' + alpha))) + '.png', image)
    print("image saved")
        # snapshot_countdown(10)
        # time.sleep(3)
if __name__ == '__main__':
    mphands = mp.solutions.hands
    hands = mphands.Hands(max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(1)
    outfile = ''
    _, frame = cap.read()

    h, w, c = frame.shape

    while True:
        _, frame = cap.read()
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                try:
                    cv2.rectangle(frame, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
                    image = frame[y_min - 50: y_max + 50, x_min - 50: x_max + 50]
                    # #TODO: Snapshot Image
                    interrupt = cv2.waitKey(10)
                    print(interrupt)
                    if interrupt & 0xFF == ord('a'):
                        SnapshotImage(image, 'T', 3)
                        # cv2.imwrite(directory + 'A/' + str(count['a']) + '.png', frame)


                except OSError as err:
                    print(repr(err))
        cv2.imshow("Frame", frame)

        cv2.waitKey(1)