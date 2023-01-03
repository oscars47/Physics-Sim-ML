import os
from caehelper import *

MAIN_DIR = '/media/oscar47/Oscar Extra/Physics data/swarm_data'
IMG_DIR = os.path.join(MAIN_DIR, 'images')
DATA_DIR = os.path.join(MAIN_DIR, 'cae_output')
build_dataset(IMG_DIR, DATA_DIR, img_height=104, img_width=104)