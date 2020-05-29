import pytest
import cv2 as cv

IMAGE_DIR = "/home/shreeraman/Work_personal/data_augmentation_YOLO/tests/test_data/"
image_paths = ["sample_yellow.jpg", "sample_red.jpg"]


@pytest.fixture(scope="session", params=image_paths)
def sample_image(request):
    image = cv.imread(IMAGE_DIR + request.param)
    return image
