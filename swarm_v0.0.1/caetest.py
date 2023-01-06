# file to test preliminary results of cae
# @oscars47

import os
import numpy as np
from caepredict import model, encoder, decoder, get_fvs
from caehelper import display_random, display_sequential

DATA_DIR = '/media/oscar47/Oscar_Extra/Physics_data/swarm_data/cae_output'

# try on train, val, and extra datasets
train_ds = np.load(os.path.join(DATA_DIR, 'train_ds.npy'))
val_ds = np.load(os.path.join(DATA_DIR, 'val_ds.npy'))
extra_ds = np.load(os.path.join(DATA_DIR, 'extra_ds.npy'))

# get first 10 elements
N = 10
train_ds = train_ds[:N]
val_ds = val_ds[:N]
extra_ds = extra_ds[:N]

# try noising
# extra_noise = noise(extra_ds)

# now call model.predict to see predictions
train_predict = model.predict(train_ds)
val_predict = model.predict(val_ds)
extra_predict = model.predict(extra_ds)

display_sequential(train_ds, train_predict)
display_sequential(val_ds, val_predict)
display_sequential(extra_ds, extra_predict)

# from sys import getsizeof

# train_fv = get_fvs(train_ds)
# train_test = encoder.predict(train_ds)
# print('fv-----------')
# #print(train_fv)
# print(train_fv.shape)
# print('size=', getsizeof(train_fv))
# print('raw encoder-----------')
# #print(train_test)
# print(train_test.shape)
# print('size=', getsizeof(train_fv))
# print('-----------------')
# print('size of images', getsizeof(train_ds))