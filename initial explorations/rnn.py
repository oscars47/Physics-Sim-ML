# file to handle recurrent neural network part of PHLUID model
import os
from keras import layers
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint

# define path to fv.npy file; we will use this to generate our semi-redundant pairings

# define model
