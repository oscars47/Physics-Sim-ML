import os
from clstmprep import *
MAIN_DIR = '/media/oscar47/Oscar Extra/Physics data'
IMG_DIR = os.path.join(MAIN_DIR, 'swarm_data', 'images')
SAVE_DIR = os.path.join(MAIN_DIR, 'swarm_data', 'data')
FRAME = 120 # set frame memory length
FACTOR = 0.3 # factor of data to load
build_dataset(IMG_DIR, SAVE_DIR, FRAME, FACTOR, img_height=104, img_width=104)