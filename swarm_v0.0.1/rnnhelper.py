# file to convert cae training and validation data into encoded feature vectors using caepredict
# @oscars47

import os
import numpy as np
from tqdm import tqdm
from caepredict import encoder, get_fvs

MAIN_DIR = '/media/oscar47/Oscar Extra/Physics_data/swarm_data'
DATA_DIR = os.path.join(MAIN_DIR, 'cae_output')
OUTPUT_DIR = os.path.join(MAIN_DIR, 'rnn_output')
MAX_FRAME = 120 # number of consecutive frame fv groupings

# function to split single ds into x and y
def get_x_y(ds, name):
    
    # encode and transform into fv
    fvs = get_fvs(ds)

    x = []
    y = []

    for i in tqdm(range(len(fvs)-MAX_FRAME), desc='preparing dataset...'):
        x.append(fvs[i:i+MAX_FRAME])
        y.append(fvs[i+MAX_FRAME])

    print('converting '+ name + ' to np arrays and saving!')
    x = np.array(x)
    y = np.array(y)
    
    get_fvs(x, OUTPUT_DIR, name)

    return x, y

def encode_data():

    # read in train, val, extra sequentially
    train_ds = np.load(os.path.join(DATA_DIR, 'train_ds.npy'))
    x, _ = get_x_y(train_ds, 'train')
    train_ds = 0 # dump the values for memory

    print(x.shape) # priting for validation
    print(x[:10])

    val_ds = np.load(os.path.join(DATA_DIR, 'val_ds.npy'))
    get_x_y(val_ds, 'val')
    val_ds = 0 # dump the values for memory

    extra_ds = np.load(os.path.join(DATA_DIR, 'extra_ds.npy'))
    get_x_y(extra_ds, 'extra')

encode_data()



