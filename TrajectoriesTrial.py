import numpy as np
import cv2 as cv

left_hand={}
right_hand={}

def main():

    cap=cv.VideoCapture(0)
    fgbg = cv.createBackgroundSubtractorMOG2()
    cv.namedWindow("video")

    k=cv.waitKey(100)
