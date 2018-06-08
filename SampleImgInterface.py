import numpy as np
import cv2 as cv
from image_transformer import ImageTransformer
import utility
import sys
import random

padding=50
class SampImgModifier:
    def __init__(self,image,size,lower,upper,bgcolor):
        self.height=size[0]+padding*2
        self.width=size[1]+padding*2
        self.channels=size[2]
        self.image = bgcolor* np.ones((self.height,self.width,self.channels),np.uint8)
        self.image[padding:(self.height-padding),padding:(self.width-padding)]=np.copy(image[0:size[0],0:size[1]])
        self.modifiedFlag=0
        self.lower=lower
        self.upper=upper
        self.maskImage=cv.inRange(self.image,lower,upper)
        self.modifiedImg=np.copy(self.image)


    def addGaussianNoise(self,noiseMean,noiseVariance):
        noiseSigma = noiseVariance ** 0.5
        foregrndPix = (np.where(self.maskImage == 0))

        if(self.modifiedFlag==1):
            height,width,_=np.shape(self.modifiedImg)
            gaussImg = np.random.normal(noiseMean, noiseSigma, (height, width, self.channels))
            self.modifiedImg= np.float32(self.modifiedImg)
            self.modifiedImg[foregrndPix] = self.modifiedImg[foregrndPix] + gaussImg[foregrndPix]
            self.modifiedImg=np.uint8(self.modifiedImg)
        else:
            gaussImg = np.random.normal(noiseMean, noiseSigma, (self.height, self.width, self.channels))
            self.modifiedImg = np.float32(self.image)
            self.modifiedImg[foregrndPix] = self.modifiedImg[foregrndPix] + gaussImg[foregrndPix]
            self.modifiedImg = np.uint8(self.modifiedImg)
            self.modifiedFlag = 1


    def addMedianNoise(self,percentPixel,percentSalt):
        foregroundPix=np.where(self.maskImage==0)
        s=np.size(foregroundPix)/2
        numPixels=int(percentPixel*s)
        allCoord=np.array(range(0,int(s)))
        random.shuffle(allCoord)

        salt_end=int(percentSalt*numPixels)
        indices = np.zeros((np.shape(foregroundPix)), np.uint64)
        indices[0] = np.array([foregroundPix[0]])
        indices[1] = np.array([foregroundPix[1]])
        salt_pixels=tuple(map(tuple,indices[:,allCoord[0:salt_end]]))
        pepper_pixels=tuple(map(tuple,indices[:,allCoord[0:salt_end]]))

        if (self.modifiedFlag == 1):
            self.modifiedImg[salt_pixels]=[255,255,255]
            self.modifiedImg[pepper_pixels]= [0, 0, 0]

        else:
            self.modifiedImg=np.copy(self.image)
            self.modifiedImg[salt_pixels] = [255, 255, 255]
            self.modifiedImg[pepper_pixels] = [0, 0, 0]

        #cv.imshow("modified",self.modifiedImg)
        #cv.waitKey(1000)

    def affineRotate(self, maxXangle,bgColor=255):

        angle=np.random.uniform(-maxXangle,maxXangle)
        if(self.modifiedFlag==1):
            height, width, _ = np.shape(self.modifiedImg)
            (cX, cY) = (width / 2, height / 2)
            M = cv.getRotationMatrix2D((cY, cY), -angle, 1)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            nW = int((height * sin) + (width * cos))
            nH = int((height * cos) + (width * sin))

            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY

            self.modifiedImg = cv.warpAffine(self.modifiedImg, M, (nW, nH), borderMode=cv.BORDER_CONSTANT,
                                             borderValue=(bgColor, bgColor, bgColor))
            self.maskImage = cv.inRange(self.modifiedImg, self.lower, self.upper)
        else:
            height, width, _ = np.shape(self.image)
            (cX,cY)= (width/2,height/2)
            M=cv.getRotationMatrix2D((cY,cY),-angle,1)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            nW = int((height * sin) + (width * cos))
            nH = int((height * cos) + (width * sin))

            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY

            self.modifiedImg= cv.warpAffine(self.image, M, (nW, nH),borderMode=cv.BORDER_CONSTANT,borderValue=(bgColor,bgColor,bgColor))
            self.maskImage = cv.inRange(self.modifiedImg, self.lower, self.upper)
            self.modifiedFlag = 1
           # cv.imshow("modified",self.modifiedImg)
           # cv.waitKey(1000)

    def perspectiveTransform(self,maxXangle,maxYangle,maxZangle,bgColor=255):
        if (self.modifiedFlag == 1):
            it = ImageTransformer(self.modifiedImg, (self.height, self.width))
        else:
            it = ImageTransformer(self.image, (self.height, self.width))
            self.modifiedFlag=1
        angX = np.random.uniform(-maxXangle, maxXangle)
        angY = np.random.uniform(-maxYangle, maxYangle)
        angZ = np.random.uniform(-maxZangle, maxZangle)
        self.modifiedImg= it.rotate_along_axis(theta=angX, phi=angY, gamma=angZ, dx=25, dy=-25, dz=0,bgColor=bgColor)
        #cv.imshow("modified",self.modifiedImg)
        #cv.waitKey(1000)
        self.maskImage=cv.inRange(self.modifiedImg, self.lower,self.upper)
        return angX,angY,angZ


    def sharpenImage(self):
        kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0],])
        if(self.modifiedFlag==1):
            self.modifiedImg=(cv.filter2D(self.modifiedImg,-1,kernel))
        else:
            self.modifiedImg=(cv.filter2D(self.image,-1,kernel))
            self.modifiedFlag= 1


    def scaleImage(self,scale):
        if (self.modifiedFlag == 1):
            self.modifiedImg = cv.resize(self.modifiedImg,None,fx=scale,fy=scale)
            self.maskImage = cv.inRange(self.modifiedImg, self.lower,self.upper)

        else:
            self.modifiedImg = cv.resize(self.image,None,fx=scale,fy=scale)
            self.maskImage = cv.inRange(self.modifiedImg, self.lower, self.upper)
            self.modifiedFlag = 1

    def modifybrightness(self,scale,percent=1):
        foregroundPix = np.where(self.maskImage == 0)
        s = int(np.size(foregroundPix) / 2)
        rand_indices=np.array(range(0,s))
        random.shuffle(rand_indices)
        end=int(s*percent)
        indices = np.zeros((np.shape(foregroundPix)), np.uint64)
        indices[0] = np.array([foregroundPix[0]])
        indices[1] = np.array([foregroundPix[1]])

        coordinates =tuple(map(tuple,indices[:,rand_indices[0:end]]))
        if(self.modifiedFlag==1):
            imageHSV = cv.cvtColor(self.modifiedImg, cv.COLOR_BGR2HSV)
            a = imageHSV[coordinates]
            a[:, 2] = scale * a[:, 2]
            imageHSV[coordinates] = a
            self.modifiedImg = cv.cvtColor(imageHSV, cv.COLOR_HSV2BGR)
            self.modifiedFlag = 1
        else:

            imageHSV = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
            a = imageHSV[coordinates]
            a[:,2]=scale* a[:,2]
            imageHSV[coordinates]=a
            self.modifiedImg = cv.cvtColor(imageHSV, cv.COLOR_HSV2BGR)
            self.modifiedFlag = 1
       # cv.imshow("modified",self.modifiedImg)
       # cv.imshow("mask", self.maskImage)
        cv.waitKey(1000)


    def getTightBoundbox(self):
        foregrndPix = (np.where(self.maskImage == 0))
        boundRect = utility.getTheBoundRect(foregrndPix)
        outImgTight = self.modifiedImg[boundRect[0][0]:(boundRect[1][0]+1 ), boundRect[0][1]:(boundRect[1][1]+1 )]
        [maskTightHeight,maskTightWidth,_] = outImgTight.shape
        maskImgTight = np.zeros((maskTightHeight, maskTightWidth), np.uint8)
        indices = [foregrndPix[0], foregrndPix[1]]

        indices[0] = indices[0] - boundRect[0][0]
        indices[1] = indices[1] - boundRect[0][1]


        foregroundPixTight = tuple(map(np.array, indices))
        try:
            maskImgTight[foregroundPixTight] = 255
        except:
            print('error')
        return foregroundPixTight,outImgTight,boundRect

    def resetFlags(self):
        self.modifiedFlag=0
        self.modifiedImg = np.copy(self.image)
        self.maskImage=self.maskImage=cv.inRange(self.image,self.lower,self.upper)




