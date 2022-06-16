import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
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
    print('Or run from CLI >>> python Main.py [-name]')
    print(f"Running Default Folder:  A\n\n")
    FOLDER = "A"
    try:
        os.path.isdir(FOLDER)
    except FileNotFoundError:
        print('Folder A does not Exist')
        sys.exit()
