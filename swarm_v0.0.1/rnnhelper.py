# file to convert cae training and validation data into encoded feature vectors using caepredict
# @oscars47

import os
import numpy as np
from tqdm import tqdm
from caepredict import encoder, get_fvs

MAIN_DIR = '/media/oscar47/Oscar_Extra/Physics_data/swarm_data'
DATA_DIR = os.path.join(MAIN_DIR, 'cae_output')
OUTPUT_DIR = os.path.join(MAIN_DIR, 'rnn_output')
MAX_FRAME = 20 # number of consecutive frame fv groupings

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
    np.save(os.path.join(OUTPUT_DIR, name+'_x.npy'), x)
    np.save(os.path.join(OUTPUT_DIR, name+'_y.npy'), y)

    return x, y

def encode_data():

    # read in train, val, extra sequentially
    train_ds = np.load(os.path.join(DATA_DIR, 'train_ds.npy'))
    # train_ds = train_ds[:int(0.6*len(train_ds))]
    x= get_x_y(train_ds, 'train')
    train_ds = 0 # dump the values for memory

    print('x shape', x.shape) # priting for validation
    print(x[:10])
    print('------------')
    print('y shape', y.shape)
    print(y[:10])
    # reset x, y
    x = y = 0

    # val_ds = np.load(os.path.join(DATA_DIR, 'val_ds.npy'))
    # get_x_y(val_ds, 'val')
    # val_ds = 0 # dump the values for memory

    # extra_ds = np.load(os.path.join(DATA_DIR, 'extra_ds.npy'))
    # get_x_y(extra_ds, 'extra')

encode_data()
# train_y = np.load(os.path.join(OUTPUT_DIR, 'train_y.npy'))
# print(train_y)
# print(train_y.shape)



