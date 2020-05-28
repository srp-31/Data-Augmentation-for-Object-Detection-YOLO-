import numpy as np
import cv2 as cv
from data_augmentation_YOLO import util
import random


DISP_X = 25
DISP_Y = -25
PADDING = 50


class SampleImgTransformer:
    def __init__(self, *, image, size, lower, upper, bgcolor):
        self.height = size[0] + PADDING * 2
        self.width = size[1] + PADDING * 2
        self.channels = size[2]
        self.image = bgcolor * np.ones(
            (self.height, self.width, self.channels), np.uint8
        )
        self.image[
            PADDING : (self.height - PADDING), PADDING : (self.width - PADDING)
        ] = np.copy(image[0 : size[0], 0 : size[1]])
        self.modified_flag = 0
        self.lower = lower
        self.upper = upper
        self.mask_image = cv.inRange(self.image, lower, upper)
        self.modified_image = np.copy(self.image)

    def addGaussianNoise(self, *, noiseMean, noiseVariance) -> None:
        """Adds Gaussian Noise to the image with a given mean and variance"""
        noiseSigma = noiseVariance ** 0.5
        foregrndPix = np.where(self.mask_image == 0)

        if self.modified_flag == 1:
            height, width, _ = np.shape(self.modified_image)
            gaussImg = np.random.normal(
                noiseMean, noiseSigma, (height, width, self.channels)
            )
            self.modified_image = np.float32(self.modified_image)
        else:
            gaussImg = np.random.normal(
                noiseMean, noiseSigma, (self.height, self.width, self.channels)
            )
            self.modified_image = np.float32(self.image)
            self.modified_flag = 1

        self.modified_image[foregrndPix] = (
            self.modified_image[foregrndPix] + gaussImg[foregrndPix]
        )
        self.modified_image = np.uint8(self.modified_image)

    def addMedianNoise(self, *, percentPixel, percentSalt) -> None:
        """Adds Median Noise to the image. The percentPixel is the percentage of the total pixels to b corrupted.
        The percentSalt accepts the percentage of corrupted pixel to be made white.Remaining will eb black."""
        foregroundPix = np.where(self.mask_image == 0)
        s = np.size(foregroundPix) / 2
        numPixels = int(percentPixel * s)
        allCoord = np.array(range(0, int(s)))
        random.shuffle(allCoord)

        salt_end = int(percentSalt * numPixels)
        indices = np.zeros((np.shape(foregroundPix)), np.uint64)
        indices[0] = np.array([foregroundPix[0]])
        indices[1] = np.array([foregroundPix[1]])
        salt_pixels = tuple(map(tuple, indices[:, allCoord[0:salt_end]]))
        pepper_pixels = tuple(map(tuple, indices[:, allCoord[0:salt_end]]))

        if not self.modified_flag == 1:
            self.modified_image = np.copy(self.image)
            self.modified_flag = 1

        self.modified_image[salt_pixels] = [255, 255, 255]
        self.modified_image[pepper_pixels] = [0, 0, 0]

    def affineRotate(self, *, maxXangle, bgColor=255) -> None:

        angle = np.random.uniform(-maxXangle, maxXangle)
        if self.modified_flag == 1:
            height, width, _ = np.shape(self.modified_image)
        else:
            height, width, _ = np.shape(self.image)
            self.modified_flag = 1

        (cX, cY) = (width / 2, height / 2)
        M = cv.getRotationMatrix2D((cY, cY), -angle, 1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((height * sin) + (width * cos))
        nH = int((height * cos) + (width * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        self.modified_image = cv.warpAffine(
            self.image,
            M,
            (nW, nH),
            borderMode=cv.BORDER_CONSTANT,
            borderValue=(bgColor, bgColor, bgColor),
        )
        self.mask_image = cv.inRange(self.modified_image, self.lower, self.upper)

        # cv.imshow("modified",self.modified_image)
        # cv.waitKey(1000)

    """ Get Perspective Projection Matrix """

    def get_M(self, *, theta, phi, gamma, dx, dy, dz):

        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1], [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )

        RY = np.array(
            [
                [np.cos(phi), 0, -np.sin(phi), 0],
                [0, 1, 0, 0],
                [np.sin(phi), 0, np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        )

        RZ = np.array(
            [
                [np.cos(gamma), -np.sin(gamma), 0, 0],
                [np.sin(gamma), np.cos(gamma), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

    def perspectiveTransform(self, *, maxXangle, maxYangle, maxZangle, bgColor=255):

        angX = np.random.uniform(-maxXangle, maxXangle)
        angY = np.random.uniform(-maxYangle, maxYangle)
        angZ = np.random.uniform(-maxZangle, maxZangle)

        rtheta, rphi, rgamma = util.get_rad(angX, angY, angZ)

        d = np.sqrt(self.height ** 2 + self.width ** 2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        mat = self.get_M(
            theta=rtheta, phi=rphi, gamma=rgamma, dx=DISP_X, dy=DISP_Y, dz=dz
        )

        if self.modified_flag == 1:
            self.modified_image = cv.warpPerspective(
                self.modified_image.copy(),
                mat,
                (self.height, self.width),
                borderMode=cv.BORDER_CONSTANT,
                borderValue=(bgColor, bgColor, bgColor),
            )

        else:
            self.modified_image = cv.warpPerspective(
                self.image.copy(),
                mat,
                (self.height, self.width),
                borderMode=cv.BORDER_CONSTANT,
                borderValue=(bgColor, bgColor, bgColor),
            )
            self.modified_flag = 1

        # cv.imshow("modified",self.modified_image)
        # cv.waitKey(1000)
        self.mask_image = cv.inRange(self.modified_image, self.lower, self.upper)

    def sharpenImage(self):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        if self.modified_flag == 1:
            self.modified_image = cv.filter2D(self.modified_image, -1, kernel)
        else:
            self.modified_image = cv.filter2D(self.image, -1, kernel)
            self.modified_flag = 1

    def scaleImage(self, *, scale):
        if self.modified_flag == 1:
            self.modified_image = cv.resize(
                self.modified_image, None, fx=scale, fy=scale
            )

        else:
            self.modified_image = cv.resize(self.image, None, fx=scale, fy=scale)
            self.modified_flag = 1

        self.mask_image = cv.inRange(self.modified_image, self.lower, self.upper)

    def modifybrightness(self, *, scale, percent=1):
        foregroundPix = np.where(self.mask_image == 0)
        s = int(np.size(foregroundPix) / 2)
        rand_indices = np.array(range(0, s))
        random.shuffle(rand_indices)
        end = int(s * percent)
        indices = np.zeros((np.shape(foregroundPix)), np.uint64)
        indices[0] = np.array([foregroundPix[0]])
        indices[1] = np.array([foregroundPix[1]])

        coordinates = tuple(map(tuple, indices[:, rand_indices[0:end]]))
        if self.modified_flag == 1:
            imageHSV = cv.cvtColor(self.modified_image, cv.COLOR_BGR2HSV)
        else:
            imageHSV = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
            self.modified_flag = 1

        a = imageHSV[coordinates]
        a[:, 2] = scale * a[:, 2]
        imageHSV[coordinates] = a
        self.modified_image = cv.cvtColor(imageHSV, cv.COLOR_HSV2BGR)
        cv.waitKey(1000)

    def getTightBoundbox(self):
        foregrndPix = np.where(self.mask_image == 0)
        boundRect = util.getTheBoundRect(foregrndPix)
        outImgTight = self.modified_image[
            boundRect[0][0] : (boundRect[1][0] + 1),
            boundRect[0][1] : (boundRect[1][1] + 1),
        ]
        [maskTightHeight, maskTightWidth, _] = outImgTight.shape
        maskImgTight = np.zeros((maskTightHeight, maskTightWidth), np.uint8)
        indices = [foregrndPix[0], foregrndPix[1]]

        indices[0] = indices[0] - boundRect[0][0]
        indices[1] = indices[1] - boundRect[0][1]

        foregroundPixTight = tuple(map(np.array, indices))
        maskImgTight[foregroundPixTight] = 255
        # try:
        #   maskImgTight[foregroundPixTight] = 255
        # except:
        #   print("error")
        return foregroundPixTight, outImgTight, boundRect

    def resetFlags(self):
        self.modified_flag = 0
        self.modified_image = np.copy(self.image)
        self.mask_image = cv.inRange(self.image, self.lower, self.upper)
