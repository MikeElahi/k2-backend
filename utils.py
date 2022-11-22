import cv2
import numpy as np
from PIL import Image


def normalize_image(image):
    input_image = np.fromstring(image, np.uint8)
    input_image = cv2.imdecode(input_image, cv2.IMREAD_COLOR)
    input_image = Image.fromarray(input_image.astype("uint8"))
    input_image = np.array(input_image)
    return input_image
