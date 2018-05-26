import numpy as np
import cv2 as cv
import glob
import os

class BackgroundFileLoader(object):

    def __init__(self):
        self.bkgImgList=[]

    def loadbkgFiles(self,bkgfilepath):
        for name in glob.glob(os.path.join(bkgfilepath,"*")):
            self.bkgImgList.append(cv.imread(name))


