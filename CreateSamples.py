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
'maxXangle':50,
'maxYangle':50,
'maxZangle':50,
'maxAngle_Affine':30,
'outputPerSample':300,
'GausNoiseProb':0.2,
'MedianNoiseProb':0.1,
'AffineRotateProb':0.3,
'SharpenProb':0.2,
'PerspTransProb':0.8,
'ScalingProb':0.7,
'BrightnessProb':1,
'OutputPath':'./Data/GermanFlag/Default',
'outputPerSample':100
}

def placeDistortedSample(outImgTight,foregroundPixTight,BoundRect,bkgImg):

    bgHeight, bgWidth, _ = np.shape(bkgImg)
    outHeight,outWidth,_ = np.shape(outImgTight)

    if (outHeight <  bgHeight and outWidth <bgWidth):

        finalImg=np.array(bkgImg).copy()

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
        #The order of x and y have been reversed for yolo
        boundRectFin[1][1] = float(BoundRect[1][0]-BoundRect[0][0] )/float(bgHeight)
        boundRectFin[1][0] = float(BoundRect[1][1] - BoundRect[0][1])/float(bgWidth)
        boundRectFin[0][1] = float(posY)/float(bgHeight)+boundRectFin[1][1]/float(2)
        boundRectFin[0][0] = float(posX)/float(bgWidth)+boundRectFin[1][0]/float(2)


        foregroundpixBkg = tuple(map(tuple, indices))
        finalImg[foregroundpixBkg] = outImgTight[foregroundPixTight]
        return True,finalImg,boundRectFin
    else:
        return False,0,0

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
    maxAngle_Affine = int (parser.get('USER_PARAMS','maxAngle_Affine'))
    GaussianNoiseProb= float(parser.get('USER_PARAMS', 'GausNoiseProb'))
    MedianNoiseProb=float(parser.get('USER_PARAMS', 'MedianNoiseProb'))
    SharpenProb=float(parser.get('USER_PARAMS', 'SharpenProb'))
    PerspTransProb = float(parser.get('USER_PARAMS', 'PerspTransProb'))
    ScalingProb = float(parser.get('USER_PARAMS', 'ScalingProb'))
    BrightnesProb=float(parser.get('USER_PARAMS', 'BrightnessProb'))
    outputPerSample = float(parser.get('USER_PARAMS', 'outputPerSample'))
    AffineRotateProb=float(parser.get('USER_PARAMS', 'AffineRotateProb'))

    if not(os.path.isdir(outputfolder)):
        os.makedirs(outputfolder)

    bkgFileLoader=BackgroundFileLoader()
    bkgFileLoader.loadbkgFiles(backgroundFilePath)
    for sampleImgPath in glob.glob(os.path.join(samplePath,'*.jpg')):

        filenameWithExt=os.path.split(sampleImgPath)[1]
        filename=os.path.splitext(filenameWithExt)[0]

        sampleImg=cv.imread(sampleImgPath)
        dimensions=np.shape(sampleImg)


        count=0
        lower = np.array([bgColor - bgThresh, bgColor - bgThresh, bgColor - bgThresh])
        upper = np.array([bgColor + bgThresh, bgColor + bgThresh, bgColor + bgThresh])
        ImgModifier=SampImgModifier(sampleImg,dimensions,lower,upper,bgColor)

        while(count<outputPerSample):

            bkgImg=bkgFileLoader.bkgImgList[np.random.randint(0,bkgFileLoader.count)]
            GaussianNoiseFlag  = np.less(np.random.uniform(0, 1),GaussianNoiseProb)
            MedianNoiseFlag    = np.less(np.random.uniform(0, 1),MedianNoiseProb)
            SharpenFlag        = np.less(np.random.uniform(0, 1),SharpenProb)
            PersTransFlag      = np.less(np.random.uniform(0, 1),PerspTransProb)
            ScalingFlag        = np.less(np.random.uniform(0, 1), ScalingProb)
            BrightnessFlag     = np.less(np.random.uniform(0, 1), BrightnesProb)
            AffineRotateFlag   = np.less(np.random.uniform(0, 1), AffineRotateProb)

            if (PersTransFlag):
                ImgModifier.perspectiveTransform(maxXangle_Persp,maxYangle_Persp,maxZangle_Persp,bgColor)

            if (AffineRotateFlag and not PersTransFlag):
                ImgModifier.affineRotate(maxAngle_Affine,bgColor)

            if(GaussianNoiseFlag):
                ImgModifier.addGaussianNoise(0,2)

            if(MedianNoiseFlag and not GaussianNoiseFlag ):
                percentPixels=0.02
                percentSalt=0.5
                ImgModifier.addMedianNoise(percentPixels,percentSalt)

            if(SharpenFlag and not(MedianNoiseFlag) and not (GaussianNoiseFlag)):
                ImgModifier.sharpenImage()

            if (ScalingFlag):
                scale=np.random.uniform(0.5,1.5)
                ImgModifier.scaleImage(scale)

            if(BrightnessFlag and not(SharpenFlag) and not(MedianNoiseFlag) and not (GaussianNoiseFlag)):
                scale = np.random.uniform(0.5, 1)
                ImgModifier.modifybrightness(scale)

            foregroundPixTight, outImgTight,BoundRect = ImgModifier.getTightBoundbox()

            flag,finalImg,finalBoundRect= placeDistortedSample(outImgTight,foregroundPixTight,BoundRect, bkgImg)
            if(flag==True):
                outputName= filename + '_'+ str(count)
                cv.imwrite(os.path.join(outputfolder,str(outputName + '.jpg')),finalImg)
                with open(os.path.join(outputfolder,str(outputName + '.txt')),'w') as f:
                    details='0 '+' '.join(str(coord) for coord in np.reshape(finalBoundRect,4))+'\n'
                    f.write(details)
                count=count+1
            else:
                outputName = filename + '_' + str(count)
                cv.imwrite(os.path.join(outputfolder, str(outputName + '.jpg')), ImgModifier.modifiedImg)
                #cv.imshow("modified",ImgModifier.modifiedImg)
                cv.waitKey(100)

            ImgModifier.resetFlags()

if __name__ == '__main__':
    main()