import numpy as np
import cv2 as cv
import argparse as ag
import random
import sys
import ConfigParser
import os
import glob
import utility

from BackgroundFileInterface  import BackgroundFileLoader
from SampleImgInterface import SampImgModifier

DEFAULT_PARAMS={
'BackgroundFilePath':'./Data/background',
'SampleFilesPath':'./Data/GermanFlag',
'bgColor': 255,
'bgTthresh':8,
'maxXangle':5,
'maxYangle':5,
'maxZangle':5,
'outputPerSample':300,
'GausNoiseProb':0.2,
'MedianNoiseProb':0.1,
'AffineRotateProb':0.3,
'SharpenProb':0.2,
'PerspTransProb':0.8,
'ScalingProb':0.7,
'OutputPath':'./Data/GermanFlag/TrainingData'
}

def placeDistortedSample(maskImg,outImg,bkgImg):

    bgHeight, bgWidth, _ = np.shape(bkgImg)
    foregrndPix = (np.where(maskImg == 0))
    BoundRect = utility.getTheBoundRect(foregrndPix)

    outImgTight = outImg[BoundRect[0][0]:(BoundRect[1][0] + 1), BoundRect[0][1]:(BoundRect[1][1] + 1)]

    maskHeight = BoundRect[1][0] - BoundRect[0][0] + 1
    maskWidth = BoundRect[1][1] - BoundRect[0][1] + 1

    maskImgTight = np.zeros((maskHeight, maskWidth), np.uint8)

    indices = [foregrndPix[0], foregrndPix[1]]

    indices[0] = indices[0] - BoundRect[0][0]
    indices[1] = indices[1] - BoundRect[0][1]

    foregroundPixTight = tuple(map(tuple, indices))

    maskImgTight[foregroundPixTight] = 255

    posX = random.randint(0, bgWidth - 2)
    if (posX + maskWidth > bgWidth):
        posX = bgWidth - maskWidth - 10

    posY = random.randint(0, bgHeight - 2)
    if (posY + maskHeight > bgHeight):
        posY = bgHeight - maskHeight - 10

    indices[0] = indices[0] + posY
    indices[1] = indices[1] + posX

    foregroundpixBkg = tuple(map(tuple, indices))

    bkgImg[foregroundpixBkg] = outImgTight[foregroundPixTight]

def main():

    parser=ConfigParser.RawConfigParser(defaults=DEFAULT_PARAMS)
    parser.read('Parameters.config')


    backgroundFilePath=parser.get('USER_PARAMS','backgroundFilePath')
    samplePath = parser.get('USER_PARAMS', 'sampleFilesPath')
    outputfolder =parser.get('USER_PARAMS', 'OutputPath')
    bgColor = int(parser.get('USER_PARAMS','bgColor'))
    bgThresh = int(parser.get('USER_PARAMS','bgThresh'))
    maxXangle_Persp = int(parser.get('USER_PARAMS', 'maxXangle'))
    maxYangle_Persp = int(parser.get('USER_PARAMS', 'maxYangle'))
    maxZangle_Persp = int(parser.get('USER_PARAMS', 'maxZangle'))
    GaussianNoiseProb= float(parser.get('USER_PARAMS', 'GausNoiseProb'))
    MedianNoiseProb=float(parser.get('USER_PARAMS', 'MedianNoiseProb'))
    SharpenProb=float(parser.get('USER_PARAMS', 'SharpenProb'))
    PerspTransProb = float(parser.get('USER_PARAMS', 'PerspTransProb'))
    ScalingProb = float(parser.get('USER_PARAMS', 'ScalingProb'))

    if not(os.path.isdir(outputfolder)):
        os.makedirs(outputfolder)

    bkgFileLoader=BackgroundFileLoader()
    bkgFileLoader.loadbkgFiles(backgroundFilePath)
    for sampleImgPath in glob.glob(os.path.join(samplePath,'*')):
        sampleImg=cv.imread(sampleImgPath)
        dimensions=np.shape(sampleImg)
        count=0
        lower = np.array([bgColor - bgThresh, bgColor - bgThresh, bgColor - bgThresh])
        upper = np.array([bgColor + bgThresh, bgColor + bgThresh, bgColor + bgThresh])
        ImgModifier=SampImgModifier(sampleImg,dimensions,lower,upper)

        while(count<300):

            GaussianNoiseFlag=np.less(np.random.uniform(0, 1),GaussianNoiseProb)
            MedianNoiseFlag = np.less(np.random.uniform(0, 1),MedianNoiseProb)
            SharpenFlag = np.less(np.random.uniform(0, 1),SharpenProb)
            PersTransFlag=np.less(np.random.uniform(0, 1),PerspTransProb)
            ScalingFlag = np.less(np.random.uniform(0, 1), SharpenProb)
            bkgIndex=np.random.uniform(0,len(bkgFileLoader.bkgImgList))

            if (PersTransFlag):
               ImgModifier.perspectiveTransform(maxXangle_Persp,maxYangle_Persp,maxZangle_Persp,bgColor,bgThresh)
            if(GaussianNoiseFlag):
                 ImgModifier.addGaussianNoise(0,10)
            if(MedianNoiseFlag):
                amtPixels=0.04

            if(SharpenFlag):
                ImgModifier.sharpenImage()

           # if (ScalingFlag):


            cv.imshow("input", ImgModifier.image)
            cv.imshow("output", ImgModifier.modifiedImg)


            bgHeight, bgWidth, _ = np.shape(bkgImg)
            placeDistortedSample(maskImg, modifiedImg, bkgImg)


            #cv.imshow("mask", maskImg)
            cv.waitKey(100000)
            count=count+1
            ImgModifier.resetFlags()

if __name__ == '__main__':
    main()

