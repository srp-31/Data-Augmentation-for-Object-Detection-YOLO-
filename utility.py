import cv2 as cv
import numpy as np

def getTheBoundRect(Pixels):
    indicesX = Pixels[1]
    indicesY = Pixels[0]
    minX = np.min(indicesX)
    maxX = np.max(indicesX)
    minY = np.min(indicesY)
    maxY = np.max(indicesY)

    BoundRect=[[minY,minX],[maxY,maxX]]
    return BoundRect