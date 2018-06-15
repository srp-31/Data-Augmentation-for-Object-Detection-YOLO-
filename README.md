# Data Augmentation for Object Detection(YOLO) [Work under progress open to suggestions]
This is a python library to augment the training dataset for object detection using YOLO. The images of the objects present in a white/black background are transfromed and then placed on various background images provided by the user. The location of the images in the background are stored according to YOLO v2 format. The library as a whole is similar in functionalits to https://github.com/aleju/imgaug and https://github.com/mdbloice/Augmentor .But the other two do not handle the adding of images to backgrounds and might vary in ease of usability. However the two have greatly inspired this development.

The available image transfromations are as follows:-
(1) Addition of Gaussian noise.
(2) Brightness variation.
(3) Addition of Salt and Pepper noise.
(4) Scaling of the image.
(5) Affine rotation given the maximum possible rotation. 
(6) Perspective Transform within user provided min and max rotation angle about x, y and z axes.
(7) Image sharpening.
 
 MORE DETAILED DOCUMENTATION IS TO BE ADDED.

Acknowledgements:
https://github.com/eborboihuc/rotate_3d used for perfroming perspective transformations. Thanks to Hou-Ning Hu.
