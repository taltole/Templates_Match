import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage import exposure
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse

# Const and Vars
get_path = lambda x, y: os.path.join(os.getcwd(), x, y)
IMG_FOLDER = 'data'
IMG_PATH = get_path(IMG_FOLDER, '')

B, G, R = (255, 0, 0), (0, 255, 0), (0, 0, 255)
ext = ['jpg', 'png', 'tif']
prep_param = dict(clr='gray', ch=2, dt='uint8')

try:
    FOLDER = sys.argv[1]
    os.path.isdir(FOLDER)
except (IndexError, FileNotFoundError):
    print('Program needs a correct folder name! please provide args [-name]')
    sys.exit()

"""
- defining at least one template for each object (the more templates you have for one object the more your recall will be
high—and your precision low)
- using OpenCV template matching method on the image for each template
- considering that each pixel having a similarity score above a template threshold is the top-left corner of an object
(with this template’s height, width, and label)
- applying Non-Maximum Suppression of the detections obtained
- choosing template thresholds to improve detection accuracy!
"""
