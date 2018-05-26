import numpy as np
import cv2 as cv
import argparse as ag
import random
import sys
import ConfigParser
import os
import glob
from image_transformer import ImageTransformer
import utility

from BackgroundFileInterface  import BackgroundFileLoader

defaultParamas={
'BackgroundFilePath':'./Data/GermanFlag',
'SampleFilesPath':'./Data/background',
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
'ScalingProb':0.7

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

    parser=ConfigParser.ConfigParser(defaults=defaultParamas)

    backgroundFilePath=parser.get('USER_PARAMS','backgroundFilePath')
    bgColor = parser.get('USER_PARAMS','bgColor')
    bgThresh = parser.get('USER_PARAMS','bgThresh')
    maxXangle_Persp = parser.get('USER_PARAMS', 'maxXangle')
    maxYangle_Persp = parser.get('USER_PARAMS', 'maxYangle')
    maxZangle_Persp = parser.get('USER_PARAMS', 'maxZangle')
    bgThresh = parser.get('USER_PARAMS', 'bgThresh')
    samplePath=parser.get('USER_PARAMS', 'sampleFilesPath')
    GaussianNoiseProb= parser.get('USER_PARAMS', 'GausNoiseProb')
    MedianNoiseProb=parser.get('USER_PARAMS', 'MedianNoiseProb')
    SharpenProb=parser.get('USER_PARAMS', 'SharpenProb')
    PerspTransProb = parser.get('USER_PARAMS', 'PerspTransProb')
    ScalingProb = parser.get('USER_PARAMS', 'ScalingProb')


    bkgFileLoader=BackgroundFileLoader()
    bkgFileLoader.loadbkgFiles(backgroundFilePath)


    for sampleImgPath in glob.glob(os.path.join(samplePath,'*')):
        sampleImg=cv.imread(sampleImgPath)
        samHeight,samWidth,samChannels=np.shape(sampleImg)
        count=0
        while(count<300):

            GaussianNoiseFlag=np.lesser(np.random.uniform(0, 1),GaussianNoiseProb)
            MedianNoiseFlag = np.lesser(np.random.uniform(0, 1),MedianNoiseProb)
            SharpenFlag = np.lesser(np.random.uniform(0, 1),SharpenProb)
            PersTransFlag=np.lesser(np.random.uniform(0, 1),PerspTransProb)
            ScalingFlag = np.lesser(np.random.uniform(0, 1), SharpenProb)



            bkgIndex=np.random.uniform(0,bkgFileLoader.bkgImgList.count)

            if (PersTransFlag):

                it = ImageTransformer(sampleImg, (samHeight, samWidth))

                angX = np.random.uniform(-maxXangle_Persp, maxXangle_Persp)
                angY = np.random.uniform(-maxYangle_Persp, maxYangle_Persp)
                angZ = np.random.uniform(-maxZangle_Persp, maxZangle_Persp)

                modifiedImg = it.rotate_along_axis(theta=angX, phi=angY, gamma=angZ, dx=5, dy=10, dz=15)

                lower = np.array([bgColor - bgThresh, bgColor - bgThresh, bgColor - bgThresh])
                upper = np.array([bgColor + bgThresh, bgColor + bgThresh, bgColor + bgThresh])
                maskImg = cv.inRange(modifiedImg, lower, upper)

            if(GaussianNoiseFlag):
                noiseMean=0
                noiseVariance=0.1
                noiseSigma=noiseVariance**0.5
                gauss=np.random.normal(noiseMean,noiseSigma,(samHeight,samWidth,samChannels))
                modifiedImg=modifiedImg+gauss

            if(MedianNoiseFlag):
                amtPixels=0.04



            if(SharpenFlag):
                imgGray=cv.cvtColor(modifiedImg,cv.COLOR_BGR2GRAY)
                imEdges=cv.Canny(imgGray,0.7,0.2)


            #if (ScalingFlag):


           # bgHeight, bgWidth, _ = np.shape(bkgImg)



            #placeDistortedSample(maskImg, modifiedImg, bkgImg)

            cv.imshow("input", sampleImg)
            #cv.imshow("output", bkgImg)
            #cv.imshow("mask", maskImg)
            cv.waitKey(100000)
            count=count+1

if __name__ == '__main__':
    main()

