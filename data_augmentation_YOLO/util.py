from math import pi
import cv2
import numpy as np

""" Utility Functions """


def getTheBoundRect(Pixels):
    indicesX = Pixels[1]
    indicesY = Pixels[0]
    minX = np.min(indicesX)
    maxX = np.max(indicesX)
    minY = np.min(indicesY)
    maxY = np.max(indicesY)

    BoundRect = [[minY, minX], [maxY, maxX]]
    return BoundRect


def resize_image(img, shape=None):
    if shape is not None:
        img = cv2.resize(img, shape)
    return img


def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta), deg_to_rad(phi), deg_to_rad(gamma))


def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta), rad_to_deg(rphi), rad_to_deg(rgamma))


def deg_to_rad(deg):
    return deg * pi / 180.0


def rad_to_deg(rad):
    return rad * 180.0 / pi
