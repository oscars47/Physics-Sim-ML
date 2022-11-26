# main file to manange CAE for PHLUID
# @oscars47

import os
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
import wandb
from wandb.keras import WandbCallback
from caehelper import *

# define img dimesions
IMG_HEIGHT = 392
IMG_WIDTH = 392
input_shape = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)) # do care about color

# load data
DATA_DIR= '/home/oscar47/Desktop/physics/swarm_data/images'

train_ds = np.load(os.path.join(DATA_DIR, 'train_ds.npy'))
val_ds = np.load(os.path.join(DATA_DIR, 'val_ds.npy'))


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

    return model

def train(config=None):
    with wandb.init(config=config):
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
      config = wandb.config

      #pprint.pprint(config)

      #initialize the neural net; 
      global model
      model = build_cae(conv2d1_size=config.conv2d1_size, conv2d2_size=config.conv2d2_size, conv2d3_size=config.conv2d3_size, convtrans1_size=config.convtrans1_size, 
         convtrans2_size=config.convtrans2_size, convtrans3_size=config.convtrans1_size, learning_rate = config.learning_rate)
      
      #now run training
      history = model.fit(
        train_ds, train_ds,
        batch_size = config.batch_size,
        validation_data=(val_ds, val_ds),
        shuffle=False,
        epochs=config.epochs,
        callbacks=[WandbCallback()] #use callbacks to have w&b log stats; will automatically save best model                     
      )


# set dictionary with random search; optimizing val_loss--------------------------
sweep_config= {
    'method': 'random',
    'name': 'val_accuracy',
    'goal': 'maximize'
}

sweep_config['metric']= 'val_accuracy'

parameters_dict = {
    'epochs': {
       'distribution': 'int_uniform',
       'min': 20,
       'max': 30
    },
    # for build_dataset
     'batch_size': {
       'values': [x for x in range(32, 161, 32)]
    },
    'conv2d1_size': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'conv2d2_size': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },'conv2d3_size': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },'convtrans1_size': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },'convtrans2_size': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'convtrans3_size': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'learning_rate':{
         #uniform distribution between 0 and 1
         'distribution': 'uniform', 
         'min': 0,
         'max': 0.1
     }
}

# append parameters to sweep config
sweep_config['parameters'] = parameters_dict 

# login to wandb----------------
wandb.init(project="Astro101_Project_v3(MEGA)", entity="oscarscholin")

# initialize sweep agent
sweep_id = wandb.sweep(sweep_config, project='Astro101_Project_v3(MEGA)', entity="oscarscholin")
wandb.agent(sweep_id, train, count=20)
