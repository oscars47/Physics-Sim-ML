# main file to manange CAE for PHLUID
# @oscars47

import os
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# define img dimesions
IMG_HEIGHT = 104
IMG_WIDTH = 104
input_shape = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)) # do care about color

# load data
MAIN_DIR = '/home/oscar47/Desktop/physics/swarm_data'
DATA_DIR= os.path.join(MAIN_DIR, 'cae_output')

train_ds = np.load(os.path.join(DATA_DIR, 'train_ds.npy'))
val_ds = np.load(os.path.join(DATA_DIR, 'val_ds.npy'))

# add noise!
# from caepredict import model
# train_dd = model.predict(train_ds)
# val_dd = model.predict(val_ds)
# # save these
# np.save('train_dd.npy', train_dd)
# np.save('val_dd.npy', val_dd)

train_dd = np.load(os.path.join(DATA_DIR, 'train_dd.npy'))
val_dd = np.load(os.path.join(DATA_DIR, 'val_dd.npy'))


# build autoencoder-------------
def build_cae(input_shape, conv2d1_size=32, conv2d2_size=32, conv2d3_size=32, convtrans1_size=32, convtrans2_size=32, convtrans3_size=32, learning_rate=0.01):
    # encoder
    model = Sequential()
    model.add(input_shape)
    #model.add(layers.Flatten())
    model.add(layers.Conv2D(conv2d1_size, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2,2), padding='same'))
    model.add(layers.Conv2D(conv2d2_size, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2,2), padding='same'))
    model.add(layers.Conv2D(conv2d3_size, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2,2), padding='same', name='FV')) # name this layer the FV (feature vector) so we can pull from it later

    # decoder
    model.add(layers.Conv2DTranspose(convtrans1_size, (3,3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Conv2DTranspose(convtrans2_size, (3,3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Conv2DTranspose(convtrans3_size, (3,3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2,2)))
    model.add(layers.Conv2D(3, (3,3), padding='same', name='OUT'))

    optimizer = Adam(learning_rate = learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    model.summary()

    return model

def train_custom():
   global model
   model = build_cae(input_shape, 200, 209, 143, 183, 227, 131, 0.0004686366872955983) # using whole4 params
      
   #now run training
   history = model.fit(
      train_dd, train_ds,
      batch_size = 128,
      validation_data=(val_dd, val_ds),
      shuffle=False,
      epochs=19                    
   )

def train_custom_resume(model, batchsize, epochs):
   #now run training
    history = model.fit(
      train_dd, train_ds,
      batch_size = batchsize,
      validation_data=(val_dd, val_ds),
      shuffle=False,
      epochs=epochs                    
   )

if not(os.path.exists(os.path.join(MAIN_DIR, 'models'))):
    os.mkdir(os.path.join(MAIN_DIR, 'models'))
modelpath = os.path.join(MAIN_DIR, 'models', 'whole4_doubledip.hdf5')
checkpoint = ModelCheckpoint(modelpath, monitor='loss',
                             verbose=1, save_best_only=True,
                             mode='min')

from caepredict import model
train_custom_resume(model, 128, 19)
