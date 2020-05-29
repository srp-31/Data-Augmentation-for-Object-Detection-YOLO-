import cv2 as cv
import glob
import os


class BackgroundFileLoader(object):
    def __init__(self):
        self.bkgImgList = []
        self.count = 0

    def loadbkgFiles(self, bkgfilepath):
        for name in glob.glob(os.path.join(bkgfilepath, "*.jpg")):
            self.bkgImgList.append(cv.imread(name))
            self.count = self.count + 1
