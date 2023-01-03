# function to perform custom runs of the C-LSTM model

import os, datetime
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam

# define img dimesions
IMG_HEIGHT = 104
IMG_WIDTH = 104
INPUT_SHAPE = layers.Input(shape=(None, IMG_HEIGHT, IMG_WIDTH, 3)) # do care about color

# load data
DATA_DIR= '/home/oscar47/Desktop/physics/swarm_data/cae_output'

train_ds = np.load(os.path.join(DATA_DIR, 'train_ds.npy'))
val_ds = np.load(os.path.join(DATA_DIR, 'val_ds.npy'))

# reduce size
train_ds = np.array(train_ds[:int(0.8*len(train_ds))])
val_ds = np.array(val_ds[:int(0.8*len(val_ds))])
print(train_ds.shape)


# to verify and view images
# print(train_ds)
# from caehelper import display
# display(train_ds[:10], val_ds[:10])

# build model------------------------
# ConvLSTM2D handles main images, Conv3D spatiotemporal elements (from keras: https://keras.io/examples/vision/conv_lstm/)
def build_model(input_shape, clstm1_size=32, clstm2_size=32, clstm3_size=32, learning_rate=0.01):
    # initialize
    model = Sequential()

    model.add(layers.Input(shape=(None, *train_ds.shape[1:])))
    model.add(layers.BatchNormalization())

    model.add(layers.ConvLSTM2D(clstm1_size, (3,3), activation = 'relu', padding='same', return_sequences=True))
    model.add(layers.BatchNormalization())

    model.add(layers.ConvLSTM2D(clstm2_size, (3,3), activation = 'relu', padding='same', return_sequences=True))
    model.add(layers.BatchNormalization())

    model.add(layers.ConvLSTM2D(clstm3_size, (3,3), activation = 'relu', padding='same', return_sequences=True))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(filters=3, kernel_size=(3,3,3), activation='sigmoid', padding="same"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy']) # add metrics so we can log stats

    model.summary()

    return model

def train_custom(batch_size=64, epochs=25):
    # initialize the model
    global model
    model = build_model(INPUT_SHAPE)

    # from tensorflow tensorboard: https://www.tensorflow.org/tensorboard/get_started
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # now run the training
    history = model.fit(
        train_ds, train_ds,
        batch_size = batch_size,
        validation_data=(val_ds, val_ds),
        shuffle=False,
        epochs=epochs,
        callbacks=[tensorboard_callback]
    )

# call the training------------
train_custom()
