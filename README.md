# Data Augmentation for Object Detection(YOLO)
This is a python library to augment the training dataset for object detection using YOLO. The images of the objects present in a white/black background are transformed and then placed on various background images provided by the user. The location of the images in the background are stored according to YOLO v2 format. The library as a whole is similar in functionalits to https://github.com/aleju/imgaug and https://github.com/mdbloice/Augmentor .But the other two do not handle the adding of images to backgrounds and might vary in ease of usability. However the two have greatly inspired this development.

The available image transfromations are as follows:-
1. Addition of Gaussian noise.
1. Brightness variation.
1. Addition of Salt and Pepper noise.
1. Scaling of the image.
1. Affine rotation given the maximum possible rotation. 
1. Perspective Transform within user provided min and max rotation angle about x, y and z axes.
1. Image sharpening.
1. Power Law transfrom for illumination effects. (needs to be update)

The starting point for  using the library is the [CreateSamples](./CreateSamples.py). The user can defined the required parameters in [Parameters](./Parameters.config). The parameters provide a veriety of information to the executing function about the number of outputsamples to generate per input sample image, amount of rotation to perform, the background color(either black or white) and the probablity with which each of the transformations are to be performed. So for each iteration of the loop generating an output image( to bee used as a sample fr object detection), a uniform random number is generated for each transformation and based on its value and the probablity specified by the user the transformation may or may not be performed. Also some trasformations like gaussian noise and sharpening are not combined(this is handled by the code internally and the end user does not need to worry) since it leads higly noise samples.

SUGGESTION: As sample images it is good to use images where there are a couple of rows and columns of white/black pixels padding the object. These will help in reducing the cropping of the object during rotation.

[SampleImageInterface](./SampleImgInterface.py) is the class that contains all the transfomrations in a single class. It also has functions to extract the tight bounding box of the modified sample image before it is placed on background images. 


Referred Sources and borrowed scripts:
1. https://github.com/eborboihuc/rotate_3d 
1. https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
