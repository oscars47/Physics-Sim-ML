# file to convert cae training and validation data into encoded feature vectors using caepredict
# @oscars47

import os
import numpy as np
from tqdm import tqdm
from caepredict import model, encoder, decoder

DATA_DIR = '/media/oscar47/Oscar Extra/Physics data/swarm_data/cae_output'
MAX_FRAME = 120 # number of consecutive frame fv groupings

# function to split single ds into x and y
def get_x_y(ds, name):
    x = []
    y = []

    for i in tqdm(range(len(ds-MAX_FRAME)), desc='preparing dataset...'):
        x.append(ds[i:i+MAX_FRAME])
        y.append(ds.append[i+MAX_FRAME])

    print('converting '+ name + ' to np arrays and saving!')
    x = np.array(x)
    y = np.array(y)
    np.save(name+'_x.npy', x)
    np.save(name+'_y.npy', y)

# read in train, val, extra sequentially
train_ds = np.load(os.path.join(DATA_DIR, 'train_ds.npy'))
# encode it!
train_encoded = encoder.predict(train_ds)
get_x_y(train_encoded, 'train')
train_ds = train_encoded = 0 # dump the values for memory


val_ds = np.load(os.path.join(DATA_DIR, 'val_ds.npy'))
# encode it!
val_encoded = encoder.predict(val_ds)
get_x_y(val_encoded, 'val')
val_ds = val_encoded = 0 # dump the values for memory

extra_ds = np.load(os.path.join(DATA_DIR, 'extra_ds.npy'))
# encode it!
extra_encoded = encoder.predict(extra_ds)
get_x_y(extra_encoded, 'extra')
extra_ds = extra_encoded = 0


