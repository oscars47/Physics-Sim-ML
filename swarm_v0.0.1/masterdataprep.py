import os
from caehelper import build_dataset
from splitvideo import master_convert

# define paths and video frame rate
MAIN_DIR = '/media/oscar47/Oscar Extra/Physics data/swarm_data'
IMG_DIR = os.path.join(MAIN_DIR, 'images')
CAE_DATA_DIR = os.path.join(MAIN_DIR, 'cae_output')
RNN_DATA_DIR = os.path.join(MAIN_DIR, 'rnn_output')
FRAME_RATE = 60

response = input('would like to (a) split video into frames or (b) convert frames to cae train and val data or (c) convert img arrays to rnn train and val data?')

if response=='a':
    print('splitting video!')
    master_convert(MAIN_DIR, FRAME_RATE)
elif response =='b':
    print('building cae datasets!')
    build_dataset(IMG_DIR, CAE_DATA_DIR, img_height=104, img_width=104)
elif response =='c':
    print('building rnn datasets!')
    # call function here
else:
    print('the option you selected was not valid. please try again.')