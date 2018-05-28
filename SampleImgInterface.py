import numpy as np
import cv2 as cv
from image_transformer import ImageTransformer
import utility

class SampImgModifier:
    def __init__(self,image,size,lower,upper):
        self.image=image
        self.height=size[0]
        self.width=size[1]
        self.channels=size[2]
        self.modifiedFlag=0
        self.lower=lower
        self.upper=upper
        self.maskImage=cv.inRange(image,lower,upper)
        self.modifiedImg=np.zeros(size,np.uint8)


    def addGaussianNoise(self,noiseMean,noiseVariance):
        noiseSigma = noiseVariance ** 0.5
        foregrndPix = (np.where(self.maskImage == 0))
        gaussImg = np.random.normal(noiseMean, noiseSigma, (self.height, self.width, self.channels))
        if(self.modifiedFlag==1):
            self.modifiedImg[foregrndPix]= np.float32(self.modifiedImg)
            self.modifiedImg[foregrndPix] = self.modifiedImg + gaussImg[foregrndPix]
            self.modifiedImg[foregrndPix]=np.uint8(self.modifiedImg )
        else:
            self.modifiedImg = np.float32(self.image)
            self.modifiedImg[foregrndPix] = self.modifiedImg[foregrndPix] + gaussImg[foregrndPix]
            self.modifiedImg = np.uint8(self.modifiedImg)
            self.modifiedFlag = 1

    def perspectiveTransform(self,maxXangle,maxYangle,maxZangle,bgColor,bgThresh):
        if (self.modifiedFlag == 1):
            it = ImageTransformer(self.modifiedImg, (self.height, self.width))
        else:
            it = ImageTransformer(self.image, (self.height, self.width))
            self.modifiedFlag=1


        angX = np.random.uniform(-maxXangle, maxXangle)
        angY = np.random.uniform(-maxYangle, maxYangle)
        angZ = np.random.uniform(-maxZangle, maxZangle)
        modifiedImg = it.rotate_along_axis(theta=angX, phi=angY, gamma=angZ, dx=5, dy=10, dz=15)
        cv.inRange(self.modifiedImgImg, self.lower,self.upper)

    def sharpenImage(self):
        kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0],])
        if(self.modifiedFlag==1):
            self.modifiedImg=(cv.filter2D(self.modifiedImg,-1,kernel))
        else:
            self.modifiedImg=(cv.filter2D(self.image,-1,kernel))
            self.modifiedFlag= 1

        cv.waitKey(100000)

    def getTightBoundbox(self):
        foregrndPix = (np.where(self.maskImage == 0))
        boundRect = utility.getTheBoundRect(foregrndPix)

        outImgTight = self.modifiedImg[boundRect[0][0]:(boundRect[1][0]+1 ), boundRect[0][1]:(boundRect[1][1]+1 )]

        maskTightHeight = boundRect[1][0] - boundRect[0][0] + 1
        maskTightWidth = boundRect[1][1] - boundRect[0][1] + 1

        maskImgTight = np.zeros((maskTightHeight, maskTightWidth), np.uint8)

        indices = [foregrndPix[0], foregrndPix[1]]

        indices[0] = indices[0] - boundRect[0][0]
        indices[1] = indices[1] - boundRect[0][1]

        foregroundPixTight = tuple(map(np.array, indices))
        maskImgTight[foregroundPixTight] = 255
        return foregroundPixTight,outImgTight,boundRect

    def resetFlags(self):
        self.modifiedFlag=0
        self.maskImage=np.zeros((self.height,self.width,self.channels),np.uint8)


    #def addMedianNoise:



