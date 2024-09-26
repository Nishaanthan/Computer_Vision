#!/usr/bin/env python3

import sys

sys.path.append(".")


from utils.opticalFlowLK import OpticalFlowLK
from utils.util import *

import numpy as np
import cv2


def main():

    cap = cv2.VideoCapture('resources/slow_traffic_small.mp4')

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    last_grey = None
    last_frame = None
    last_keypoints = []
    status = []

    winsize = [25, 25]

    while (cap.isOpened()):
        ret, frame = cap.read()
        scale = 0.7
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        termcrit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03)
        subPixWinSize = (10, 10)

        keypoints = cv2.goodFeaturesToTrack(grey, 200, 0.01, 10, mask = None, blockSize = 3, useHarrisDetector = False, k = 0.04)
        cv2.cornerSubPix(grey, keypoints, subPixWinSize, tuple([-1, -1]), termcrit)

        keypoints = np.array(keypoints)
        keypoints = np.squeeze(keypoints, axis=1)

        # keypoints = np.array([[302.279, 100.72]])

        flowPoints = 0
        if not len(last_keypoints) == 0:

            of = OpticalFlowLK(winsize, 0.03, 20)
            print("last_grey")
            print(last_grey)
            print("\n")
            print("grey")
            print(grey)
            print("\n")
            print("keypoints")
            print(np.copy(last_keypoints))
            print("\n")
            points, status = of.compute(last_grey, grey, np.copy(last_keypoints))
            print("points")
            print(points)
            print("\n")
            print("status")
            print(status)
            print("\n\n\n\n\n")

            for i in range(len(points)):

                if not status[i]:
                    continue

                diff = points[i] - last_keypoints[i]
                distance = np.linalg.norm(diff)

                if distance > 15 or distance < 0.2:
                    continue

                otherP = last_keypoints[i] + diff * 15
                flowPoints += 1

                color = tuple([0, 255, 0])
                cv2.circle(last_frame, tuple(last_keypoints[i].round().astype(int)), 1, color)
                cv2.line(last_frame, tuple(last_keypoints[i].round().astype(int)), tuple(otherP.round().astype(int)), color)

            cv2.imshow("out", last_frame)
            cv2.waitKey(1)

            print("[Keypoints] moving/total: {} / {}".format(flowPoints, len(points)))
        last_keypoints = np.copy(keypoints)
        last_grey = np.copy(grey)
        last_frame = np.copy(frame)


if __name__ == '__main__':
    main()
