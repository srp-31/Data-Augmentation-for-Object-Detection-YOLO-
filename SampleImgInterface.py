import numpy as np
import cv2 as cv
from image_transformer import ImageTransformer
import utility

class SampImgModifier:
    def __init__(self,image,size,lower,upper,bgcolor):
        self.height=size[0]+100
        self.width=size[1]+100
        self.channels=size[2]
        self.image = bgcolor* np.ones((self.height,self.width,self.channels),np.uint8)
        self.image[50:(self.height-50),50:(self.width-50)]=np.copy(image[0:size[0],0:size[1]])
        self.modifiedFlag=0
        self.lower=lower
        self.upper=upper
        self.maskImage=cv.inRange(self.image,lower,upper)
        self.modifiedImg=np.copy(self.image)


    def addGaussianNoise(self,noiseMean,noiseVariance):
        noiseSigma = noiseVariance ** 0.5
        foregrndPix = (np.where(self.maskImage == 0))
        gaussImg = np.random.normal(noiseMean, noiseSigma, (self.height, self.width, self.channels))
        if(self.modifiedFlag==1):
            self.modifiedImg= np.float32(self.modifiedImg)
            self.modifiedImg[foregrndPix] = self.modifiedImg[foregrndPix] + gaussImg[foregrndPix]
            self.modifiedImg=np.uint8(self.modifiedImg )
        else:
            self.modifiedImg = np.float32(self.image)
            self.modifiedImg[foregrndPix] = self.modifiedImg[foregrndPix] + gaussImg[foregrndPix]
            self.modifiedImg = np.uint8(self.modifiedImg)
            self.modifiedFlag = 1


    def addMedianNoise(self,percentPixel,percentSalt):
        foregroundPix=np.where(self.maskImage==0)
        s=np.size(foregroundPix)/2
        numPixels=percentPixel*s
        a=foregroundPix[0][0]
        allCoord=np.random.randint(0,s,int(numPixels))
        salt_end=int(percentSalt*np.size(allCoord))
        if (self.modifiedFlag == 1):
            for i in allCoord[0:salt_end]:
               self.modifiedImg[foregroundPix[0][i]][foregroundPix[1][i]]=[255,255,255]
            for i in allCoord[salt_end:np.size(allCoord)]:
               self.modifiedImg[foregroundPix[0][i]][foregroundPix[1][i]] = [0, 0, 0]

        else:
            self.modifiedImg=np.copy(self.image)
            for i in allCoord[0:salt_end]:
                self.modifiedImg[foregroundPix[0][i]][foregroundPix[1][i]] = [255, 255, 255]
            for i in allCoord[salt_end:np.size(allCoord)]:
                self.modifiedImg[foregroundPix[0][i]][foregroundPix[1][i]] = [0, 0, 0]


    def perspectiveTransform(self,maxXangle,maxYangle,maxZangle,bgColor,bgThresh):
        if (self.modifiedFlag == 1):
            it = ImageTransformer(self.modifiedImg, (self.height, self.width))
        else:
            it = ImageTransformer(self.image, (self.height, self.width))
            self.modifiedFlag=1
        angX = np.random.uniform(-maxXangle, maxXangle)
        angY = np.random.uniform(-maxYangle, maxYangle)
        angZ = np.random.uniform(-maxZangle, maxZangle)
        self.modifiedImg= it.rotate_along_axis(theta=angX, phi=angY, gamma=angZ, dx=0, dy=-50, dz=0)
        cv.imshow("modified",self.modifiedImg)
        cv.waitKey(100)
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


    #def addMedianNoise:


