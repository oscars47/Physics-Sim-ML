# file to handle recurrent neural network part of PHLUID model
# notice similarity to TP RNN

import os
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# paths
MAIN_DIR = '/media/oscar47/Oscar_Extra/Physics_data/swarm_data'
DATA_DIR= os.path.join(MAIN_DIR, 'rnn_output')
MAX_FRAME = 120 # number of consecutive frame fv groupings

# load datasets
x_train = np.load(os.path.join(DATA_DIR, 'train_x.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'train_y.npy'))
x_val = np.load(os.path.join(DATA_DIR, 'val_x.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'val_y.npy'))

# build model functions--------------------------------
def train_custom(LSTM_layer_size_1=128,  LSTM_layer_size_2=128, LSTM_layer_size_3=128, 
              LSTM_layer_size_4=128, LSTM_layer_size_5=128, 
              dropout=0.1, learning_rate=0.01, epochs=1, batchsize=32):
    #initialize the neural net; 
    global model
    model = build_model(LSTM_layer_size_1,  LSTM_layer_size_2, LSTM_layer_size_3, 
            LSTM_layer_size_4, LSTM_layer_size_5, 
            dropout, learning_rate)
    
    #now run training
    history = model.fit(
    x_train, y_train,
    batch_size = batchsize,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=callbacks #use callbacks to have w&b log stats; will automatically save best model                     
    )

# define model
def build_model(LSTM_layer_size_1,  LSTM_layer_size_2, LSTM_layer_size_3, 
          LSTM_layer_size_4, LSTM_layer_size_5, 
          dropout, learning_rate):
    # call initialize function
    
    model = Sequential()
    # RNN layers for language processing
    model.add(LSTM(LSTM_layer_size_1, input_shape = (x_train[0].shape), return_sequences=True))
    model.add(LSTM(LSTM_layer_size_2, return_sequences=True))
    model.add(LSTM(LSTM_layer_size_3, return_sequences=True))
    model.add(LSTM(LSTM_layer_size_4, return_sequences=True))
    model.add(LSTM(LSTM_layer_size_5))
    model.add(Dropout(dropout))

    model.add(Dense(len(y_train[0])))
    model.add(Activation('softmax'))


    # put structure together
    optimizer = RMSprop(learning_rate = learning_rate)
    model.compile(loss='categorical_crossentropy')

    model.summary()

    return model

def train_custom(LSTM_layer_size_1=128,  LSTM_layer_size_2=128, LSTM_layer_size_3=128, 
              LSTM_layer_size_4=128, LSTM_layer_size_5=128, 
              dropout=0.1, learning_rate=0.01, epochs=1, batchsize=32):
    #initialize the neural net; 
    global model
    model = build_model(LSTM_layer_size_1,  LSTM_layer_size_2, LSTM_layer_size_3, 
            LSTM_layer_size_4, LSTM_layer_size_5, 
            dropout, learning_rate)
    
    #now run training
    history = model.fit(
    x_train, y_train,
    batch_size = batchsize,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=callbacks #use callbacks to have w&b log stats; will automatically save best model                     
    ) 

def train_custom_resume(model, batchsize, epochs):
   #now run training
   history = model.fit(
   x_train, y_train,
   batch_size = batchsize,
   validation_data=(x_val, y_val),
   epochs=epochs,
   callbacks=callbacks #use callbacks to have w&b log stats; will automatically save best model                     
   )

# define two other callbacks
# save model
if not(os.path.exists(os.path.join(MAIN_DIR, 'models'))):
    os.mkdir(os.path.join(MAIN_DIR, 'models'))
modelpath = os.path.join(MAIN_DIR, 'models', 'rnn1.hdf5')
checkpoint = ModelCheckpoint(modelpath, monitor='loss',
                             verbose=1, save_best_only=True,
                             mode='min')
# if learning stals, reduce the LR
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=1, min_lr=0.001)

# compile the callbacks
#callbacks = [checkpoint, reduce_lr, WandbCallback()]
callbacks = [checkpoint, reduce_lr]

train_custom()
