import numpy as np
import cv2 as cv
import random
import configparser
import os
import glob
import utility
import csv

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

def placeDistortedSample(outImgTight,foregroundPixTight,BoundRect,bkgImg,):

    bgHeight, bgWidth, _ = np.shape(bkgImg)
    outHeight,outWidth,_ = np.shape(outImgTight)
    finalImg=bkgImg

    posX = np.random.randint(0,bgWidth-outWidth)
    if (posX + outWidth > bgWidth):
        posX = bgWidth - outWidth - 10

    posY = np.random.randint(0,bgHeight-10)
    if (posY + outHeight > bgHeight-outHeight):
        posY = bgHeight - outHeight - 10

    indices=np.zeros((np.shape(foregroundPixTight)),np.uint64)
    indices[0] = np.array([foregroundPixTight[0]]) + posY
    indices[1] = np.array([foregroundPixTight[1]]) + posX

    boundRectFin =np.zeros((2,2),float)
    boundRectFin[1][0] = float(BoundRect[1][0]-BoundRect[0][0] + posY)/float(bgHeight)
    boundRectFin[1][1] = float(BoundRect[1][1] - BoundRect[0][1] + posX)/float(bgWidth)
    boundRectFin[0][0] = float(posY)/float(bgHeight)
    boundRectFin[0][1] = float(posX)/float(bgWidth)


    foregroundpixBkg = tuple(map(tuple, indices))
    finalImg[foregroundpixBkg] = outImgTight[foregroundPixTight]
    return finalImg,boundRectFin
def main():

    parser=configparser.RawConfigParser(defaults=DEFAULT_PARAMS)
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
    for sampleImgName in os.listdir(samplePath):

        filename=os.path.splitext(sampleImgName)[0]
        sampleImgPath=os.path.join(samplePath,sampleImgName)
        sampleImg=cv.imread(sampleImgPath)
        dimensions=np.shape(sampleImg)
        count=0
        lower = np.array([bgColor - bgThresh, bgColor - bgThresh, bgColor - bgThresh])
        upper = np.array([bgColor + bgThresh, bgColor + bgThresh, bgColor + bgThresh])
        ImgModifier=SampImgModifier(sampleImg,dimensions,lower,upper)

        while(count<300):

            bkgImg=bkgFileLoader.bkgImgList[np.random.randint(0,bkgFileLoader.count)]
            GaussianNoiseFlag  = np.less(np.random.uniform(0, 1),GaussianNoiseProb)
            MedianNoiseFlag    = np.less(np.random.uniform(0, 1),MedianNoiseProb)
            SharpenFlag        = np.less(np.random.uniform(0, 1),SharpenProb)
            PersTransFlag      = np.less(np.random.uniform(0, 1),PerspTransProb)
            ScalingFlag        = np.less(np.random.uniform(0, 1), SharpenProb)

            if (PersTransFlag):
               ImgModifier.perspectiveTransform(maxXangle_Persp,maxYangle_Persp,maxZangle_Persp,bgColor,bgThresh)
            if(GaussianNoiseFlag):
                 ImgModifier.addGaussianNoise(0,10)
            if(MedianNoiseFlag):
                amtPixels=0.04

            if(SharpenFlag):
                ImgModifier.sharpenImage()

            # if (ScalingFlag):



            foregroundPixTight, outImgTight,BoundRect = ImgModifier.getTightBoundbox()
            finalImg,finalBoundRect= placeDistortedSample(outImgTight,foregroundPixTight,BoundRect, bkgImg)
            outputName= filename + '_'+ str(count)

            cv.imwrite(os.path.join(outputfolder,str(outputName + '.jpg')),finalImg)
            with open(os.path.join(outputfolder,str(outputName + '.txt')),'w') as f:
                csv_writer = csv.writer(f)
                details=list(['0'])+list(np.reshape(finalBoundRect,4).astype(np.str))
                csv_writer.writerow(details)

            cv.imshow("input", ImgModifier.image)
            cv.imshow("output", finalImg)
            cv.waitKey(100000)
            count=count+1
            ImgModifier.resetFlags()

if __name__ == '__main__':
    main()

