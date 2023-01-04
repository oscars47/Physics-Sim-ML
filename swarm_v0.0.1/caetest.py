# file to test preliminary results of cae
# @oscars47

import os
import numpy as np
from caepredict import model, encoder, decoder
from caehelper import display_random, display_sequential

DATA_DIR = '/home/oscar47/Desktop/physics/swarm_data/cae_output'

# try on train, val, and extra datasets
train_ds = np.load(os.path.join(DATA_DIR, 'train_ds.npy'))
val_ds = np.load(os.path.join(DATA_DIR, 'val_ds.npy'))
extra_ds = np.load(os.path.join(DATA_DIR, 'extra_ds.npy'))

# get first 10 elements
N = 10
train_ds = train_ds[:N]
val_ds = val_ds[:N]
extra_ds = extra_ds[:N]

# now call model.predict to see predictions
train_predict = model.predict(train_ds)
val_predict = model.predict(val_ds)
extra_predict = model.predict(extra_ds)

display_sequential('Train', train_ds, train_predict)
display_sequential('Val', val_ds, val_predict)
display_sequential('Extra', extra_ds, extra_predict)