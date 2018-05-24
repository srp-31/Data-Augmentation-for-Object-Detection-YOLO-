import numpy as np
import cv2 as cv
import argparse as ag
import random
import sys


import image_transformer

def main():

    parser = ag.ArgumentParser(description='Take camera index')
    parser.add_argument('-bgfile',dest='BackgroundFile')
    parser.add_argument('-samplefile',dest='SampleFile')
    parser.add_argument('-bgcolor',dest='bgcolor',default=255)
    parser.add_argument('-bgthresh', dest='bgthresh', default=8)
    args=parser.parse_args()

    sampleImg=cv.imread(args.SampleFile)
    bkgImg=cv.imread(args.BackgroundFile)
    bgColor=args.bgcolor
    bgThresh=args.bgthresh

    samHeight,samWidth,_=np.shape(sampleImg)
    bgHeight,bgWidth,_=np.shape(bkgImg)


    if (samHeight/bgHeight <0.5 and samWidth/bgWidth <0.5):

        inputQuad=np.array([[-30,60],[samWidth+50,-50],
                            [samWidth+100,samHeight+50],
                            [-50,samHeight+50]],np.float32)
        outputQuad=np.array([[0,0],[samWidth-1,0],
                             [samWidth-1,samHeight-1],
                             [0,samHeight-1]],np.float32)

        tranMat=cv.getPerspectiveTransform(inputQuad,outputQuad)
        outImg=cv.warpPerspective(sampleImg,tranMat,(samWidth,samHeight),borderMode=cv.BORDER_CONSTANT,borderValue=(255,255,255))

        lower = np.array([bgColor-bgThresh, bgColor-bgThresh, bgColor-bgThresh])
        upper = np.array([bgColor+bgThresh, bgColor+bgThresh, bgColor+bgThresh])

        mask = cv.inRange(outImg, lower, upper)
        foregrndPix=(np.where(mask==0))
        maskImg=np.zeros((samHeight,samWidth,1),np.uint8)
        maskImg[foregrndPix]=255

        posX=random.randint (0,bgWidth-2)
        if(posX+samWidth > bgWidth):
           posX=bgWidth-samWidth-10

        posY=random.randint (0,bgHeight-2)
        if (posY + samHeight > bgHeight):
            posY = bgHeight - samHeight - 10
        indices=[foregrndPix[0],foregrndPix[1]]

        indices[0]=indices[0]+posY
        indices[1]= indices[1]+posX

        foregroundpixBkg=tuple(map(tuple,indices))

        bkgImg[foregroundpixBkg]=outImg[foregrndPix]


        #bkgImg[posX+foregrndPix[0],posY+foregrndPix[1]]=outImg[foregrndPix]


        cv.imshow("input",sampleImg)
        cv.imshow("output",bkgImg)
        cv.imshow("mask",maskImg)
        cv.waitKey(100000)

if __name__ == '__main__':
    main()

