import numpy as np
import cv2 as cv
import os
import glob
import yaml
from data_augmentation_YOLO.bkg_files_loader import BackgroundFileLoader
from data_augmentation_YOLO.image_transformer import SampleImgTransformer


def placeDistortedSample(outImgTight, foregroundPixTight, BoundRect, bkgImg):

    bgHeight, bgWidth, _ = np.shape(bkgImg)
    outHeight, outWidth, _ = np.shape(outImgTight)

    if outHeight < bgHeight and outWidth < bgWidth:

        finalImg = np.array(bkgImg).copy()

        posX = np.random.randint(0, bgWidth - outWidth)
        if posX + outWidth > bgWidth:
            posX = bgWidth - outWidth - 10

        posY = np.random.randint(0, bgHeight - 10)
        if posY + outHeight > bgHeight - outHeight:
            posY = bgHeight - outHeight - 10

        indices = np.zeros((np.shape(foregroundPixTight)), np.uint64)
        indices[0] = np.array([foregroundPixTight[0]]) + posY
        indices[1] = np.array([foregroundPixTight[1]]) + posX

        boundRectFin = np.zeros((2, 2), float)
        # The order of x and y have been reversed for yolo
        boundRectFin[1][1] = float(BoundRect[1][0] - BoundRect[0][0]) / float(bgHeight)
        boundRectFin[1][0] = float(BoundRect[1][1] - BoundRect[0][1]) / float(bgWidth)
        boundRectFin[0][1] = float(posY) / float(bgHeight) + boundRectFin[1][1] / float(
            2
        )
        boundRectFin[0][0] = float(posX) / float(bgWidth) + boundRectFin[1][0] / float(
            2
        )

        foregroundpixBkg = tuple(map(tuple, indices))
        finalImg[foregroundpixBkg] = outImgTight[foregroundPixTight]
        return True, finalImg, boundRectFin
    else:
        return False, 0, 0


def main():
    with open("config.yaml") as fp:
        config_params = yaml.load(fp, Loader=yaml.FullLoader)

    backgroundFilePath = config_params.get("BACKGROUND_FILE_PATH")
    samplePath = config_params.get("SAMPLE_FILES_PATH")
    outputfolder = config_params.get("OUTPUT_PATH")
    bgColor = config_params.get("BACKGROUND_COLOR")
    bgThresh = config_params.get("BACKGROUND_THRESH")
    maxXangle_Persp = config_params.get("MAX_X_ANGLE")
    maxYangle_Persp = config_params.get("MAX_Y_ANGLE")
    maxZangle_Persp = config_params.get("MAX_Z_ANGLE")
    maxAngle_Affine = config_params.get("MAX_AFFINE_ANGLE")
    PerspTransProb = config_params.get("PERSP_TRANS_PROB")
    AffineRotateProb = config_params.get("AFFINE_ROT_PROB")
    GaussianNoiseProb = config_params.get("GAUSS_NOISE_PROB")
    MedianNoiseProb = config_params.get("MEDIAN_NOISE_PROB")
    SharpenProb = config_params.get("SHARPEN_PROB")
    ScalingProb = config_params.get("SCALING_PROB")
    BrightnesProb = config_params.get("BRIGHTNESS_PROB")
    outputPerSample = config_params.get("OUTPUT_PER_SAMPLE")

    if not (os.path.isdir(outputfolder)):
        os.makedirs(outputfolder)

    bkgFileLoader = BackgroundFileLoader()
    bkgFileLoader.loadbkgFiles(backgroundFilePath)
    for sampleImgPath in glob.glob(os.path.join(samplePath, "*.jpg")):

        filenameWithExt = os.path.split(sampleImgPath)[1]
        filename = os.path.splitext(filenameWithExt)[0]

        sampleImg = cv.imread(sampleImgPath)
        dimensions = np.shape(sampleImg)

        count = 0
        lower = np.array([bgColor - bgThresh, bgColor - bgThresh, bgColor - bgThresh])
        upper = np.array([bgColor + bgThresh, bgColor + bgThresh, bgColor + bgThresh])
        ImgModifier = SampleImgTransformer(sampleImg, dimensions, lower, upper, bgColor)

        while count < outputPerSample:

            bkgImg = bkgFileLoader.bkgImgList[np.random.randint(0, bkgFileLoader.count)]
            GaussianNoiseFlag = np.less(np.random.uniform(0, 1), GaussianNoiseProb)
            MedianNoiseFlag = np.less(np.random.uniform(0, 1), MedianNoiseProb)
            SharpenFlag = np.less(np.random.uniform(0, 1), SharpenProb)
            PersTransFlag = np.less(np.random.uniformpyyaml(0, 1), PerspTransProb)
            ScalingFlag = np.less(np.random.uniform(0, 1), ScalingProb)
            BrightnessFlag = np.less(np.random.uniform(0, 1), BrightnesProb)
            AffineRotateFlag = np.less(np.random.uniform(0, 1), AffineRotateProb)

            if PersTransFlag:
                ImgModifier.perspectiveTransform(
                    maxXangle_Persp, maxYangle_Persp, maxZangle_Persp, bgColor
                )

            if AffineRotateFlag and not PersTransFlag:
                ImgModifier.affineRotate(maxAngle_Affine, bgColor)

            if GaussianNoiseFlag:
                ImgModifier.addGaussianNoise(0, 2)

            if MedianNoiseFlag and not GaussianNoiseFlag:
                percentPixels = 0.02
                percentSalt = 0.5
                ImgModifier.addMedianNoise(percentPixels, percentSalt)

            if SharpenFlag and not (MedianNoiseFlag) and not (GaussianNoiseFlag):
                ImgModifier.sharpenImage()

            if ScalingFlag:
                scale = np.random.uniform(0.5, 1.5)
                ImgModifier.scaleImage(scale)

            if (
                BrightnessFlag
                and not (SharpenFlag)
                and not (MedianNoiseFlag)
                and not (GaussianNoiseFlag)
            ):
                scale = np.random.uniform(0.5, 1)
                ImgModifier.modifybrightness(scale)

            foregroundPixTight, outImgTight, BoundRect = ImgModifier.getTightBoundbox()

            flag, finalImg, finalBoundRect = placeDistortedSample(
                outImgTight, foregroundPixTight, BoundRect, bkgImg
            )
            if flag:
                outputName = filename + "_" + str(count)
                cv.imwrite(
                    os.path.join(outputfolder, str(outputName + ".jpg")), finalImg
                )
                with open(
                    os.path.join(outputfolder, str(outputName + ".txt")), "w"
                ) as f:
                    details = (
                        "0 "
                        + " ".join(
                            str(coord) for coord in np.reshape(finalBoundRect, 4)
                        )
                        + "\n"
                    )
                    f.write(details)
                count = count + 1
            else:
                outputName = filename + "_" + str(count)
                cv.imwrite(
                    os.path.join(outputfolder, str(outputName + ".jpg")),
                    ImgModifier.modified_image,
                )
                # cv.imshow("modified",ImgModifier.modifiedImg)
                cv.waitKey(100)

            ImgModifier.resetFlags()


if __name__ == "__main__":
    main()
