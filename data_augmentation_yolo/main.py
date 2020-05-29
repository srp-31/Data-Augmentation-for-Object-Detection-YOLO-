import numpy as np
import cv2 as cv
import os
import glob
import yaml
import logging
from data_augmentation_yolo.bkg_files_loader import BackgroundFileLoader
from data_augmentation_yolo.image_transformer import SampleImgTransformer
from data_augmentation_yolo.config import get_console_handler, get_file_handler


def place_distorted_sample(outImgTight, foregroundPixTight, BoundRect, bkgImg):

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


def augment_data():
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
    persp_trans_prob = config_params.get("PERSP_TRANS_PROB")
    aff_rot_prob = config_params.get("AFFINE_ROT_PROB")
    gauss_noise_prob = config_params.get("GAUSS_NOISE_PROB")
    median_noise_prob = config_params.get("MEDIAN_NOISE_PROB")
    sharpen_prob = config_params.get("SHARPEN_PROB")
    scaling_prob = config_params.get("SCALING_PROB")
    brightness_prob = config_params.get("BRIGHTNESS_PROB")
    output_per_sample = config_params.get("OUTPUT_PER_SAMPLE")

    if not (os.path.isdir(outputfolder)):
        os.makedirs(outputfolder)

    log_file_path = outputfolder + "/data_augmentation_yolo.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_file_handler(log_file_path))
    logger.addHandler(get_console_handler())
    logger.info("logger created")
    bkgFileLoader = BackgroundFileLoader()
    bkgFileLoader.loadbkgFiles(backgroundFilePath)

    for sampleImgPath in glob.glob(os.path.join(samplePath, "*.jpg")):

        filenameWithExt = os.path.split(sampleImgPath)[1]
        filename = os.path.splitext(filenameWithExt)[0]

        sampleImg = cv.imread(sampleImgPath)

        count = 0

        image_modifier = SampleImgTransformer(
            image=sampleImg, bg_color=bgColor, bg_thresh=bgThresh
        )

        while count < output_per_sample:

            bkg_img = bkgFileLoader.bkgImgList[
                np.random.randint(0, bkgFileLoader.count)
            ]
            gauss_noise_flag = np.less(np.random.uniform(0, 1), gauss_noise_prob)
            median_noise_flag = np.less(np.random.uniform(0, 1), median_noise_prob)
            sharpen_flag = np.less(np.random.uniform(0, 1), sharpen_prob)
            pers_trans_flag = np.less(np.random.uniform(0, 1), persp_trans_prob)
            scaling_flag = np.less(np.random.uniform(0, 1), scaling_prob)
            brightness_flag = np.less(np.random.uniform(0, 1), brightness_prob)
            affine_rot_flag = np.less(np.random.uniform(0, 1), aff_rot_prob)

            if pers_trans_flag:
                image_modifier.perspectiveTransform(
                    maxXangle=maxXangle_Persp,
                    maxYangle=maxYangle_Persp,
                    maxZangle=maxZangle_Persp,
                    bgColor=bgColor,
                )

            if affine_rot_flag and not pers_trans_flag:
                image_modifier.affineRotate(maxXangle=maxAngle_Affine, bgColor=bgColor)

            if gauss_noise_flag:
                image_modifier.addGaussianNoise(noiseMean=0, noiseVariance=2)
                image_modifier.modified_image = np.uint8(image_modifier.modified_image)

            if median_noise_flag and not gauss_noise_flag:
                percent_pixels = 0.02
                percent_salt = 0.5
                image_modifier.addMedianNoise(
                    percentPixel=percent_pixels, percentSalt=percent_salt
                )

            if sharpen_flag and not (median_noise_flag) and not (gauss_noise_flag):
                image_modifier.sharpenImage()

            if scaling_flag:
                scale = np.random.uniform(0.5, 1.5)
                image_modifier.scaleImage(scale=scale)

            if (
                brightness_flag
                and not (sharpen_flag)
                and not (median_noise_flag)
                and not (gauss_noise_flag)
            ):
                scale = np.random.uniform(0.5, 1)
                image_modifier.modifybrightness(scale=scale)

            (
                foregroundPixTight,
                outImgTight,
                BoundRect,
            ) = image_modifier.getTightBoundbox()

            flag, finalImg, finalBoundRect = place_distorted_sample(
                outImgTight, foregroundPixTight, BoundRect, bkg_img
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
                logger.log(logging.INFO, "%s augmented file created", outputName)
                count = count + 1

            image_modifier.resetFlags()


if __name__ == "__main__":
    augment_data()
