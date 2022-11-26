# main file to manange CAE for PHLUID
# @oscars47

import os
from keras import layers
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint
from caehelper import *

# define img dimesions
img_height = 80
img_width = 80
input_shape = layers.Input(shape=(img_height, img_width, 3)) # do care about color

# load data
img_path = '/Volumes/PHLUID/Data/Balls/images/test'
_BATCH_SIZE = 10

train_ds, val_ds = build_dataset(img_path, img_height, img_width)
print(train_ds.shape)
#print(train_ds)

cond = True
while cond == True:
    choice = int(input('do you want to (1) run CAE or (2) load CAE or (47) to exit?'))
    if choice==1:
        cond = False

        # build autoencoder-------------

        def build_cae(input_shape, conv2d1_size=32, conv2d2_size=32, conv2d3_size=32, convtrans1_size=32, convtrans2_size=32, convtrans3_size=32):
            # encoder
            model = Sequential()
            model.add(input_shape)
            #model.add(layers.Flatten())
            model.add(layers.Conv2D(conv2d1_size, (3, 3), activation='relu', padding='same'))
            model.add(layers.MaxPool2D((2,2), padding='same'))
            model.add(layers.Conv2D(conv2d2_size, (3,3), activation='relu', padding='same'))
            model.add(layers.MaxPool2D((2,2), padding='same'))
            model.add(layers.Conv2D(conv2d3_size, (3,3), activation='relu', padding='same'))
            model.add(layers.MaxPool2D((2,2), padding='same', name='FV'))

            # decoder
            model.add(layers.Conv2DTranspose(convtrans1_size, (3,3), activation='relu', padding='same'))
            model.add(layers.UpSampling2D((2,2)))
            model.add(layers.Conv2DTranspose(convtrans2_size, (3,3), activation='relu', padding='same'))
            model.add(layers.UpSampling2D((2,2)))
            model.add(layers.Conv2DTranspose(convtrans3_size, (3,3), activation='relu', padding='same'))
            model.add(layers.UpSampling2D((2,2)))
            model.add(layers.Conv2D(3, (3,3), padding='same', name='OUT'))

            return model

        # put it all together
        ae = build_cae(input_shape)
        ae.compile(optimizer='adam', loss='mse')
        ae.summary()

        # define epochs parameter
        epochs = 5

        #define model path
        identifier = str(img_width)+'_'+str(img_height)+'_'+str(epochs)
        modelpath = os.path.join('/Volumes/PHLUID/Data/Balls/', 'test_models',str(img_width)+'_'+str(img_height)+'_'+str(epochs))
        if not(os.path.exists(modelpath)):
            os.makedirs(modelpath)

        checkpoint = ModelCheckpoint(modelpath, monitor='loss',
                                    verbose=1, save_best_only=True,
                                    mode='min')
        callbacks = [checkpoint]

        # now train model!------------
        # need to save model
        ae.fit(
            x = train_ds,
            y = train_ds,
            epochs=epochs,
            batch_size=_BATCH_SIZE,
            shuffle=False,
            validation_data=(val_ds, val_ds),
            callbacks=callbacks
        )

        # now make predictions---------
        preds = ae.predict(train_ds)
        display(train_ds, preds)
    elif choice == 2:
        cond = False
        model_dir = '/Volumes/PHLUID/Data/Balls/test_models'

        model = load_model(model_dir)
        # create new model for the FV
        get_fv = Model(inputs=model.input, outputs=model.get_layer('FV').output)

        #print(len(train_ds))

        # fv for all images
        print('train_ds',train_ds.shape)
        fv = get_fv.predict(train_ds)
        shape = fv.shape
        shape1 = shape[1]
        shape2 = shape[2]
        shape3 = shape[3]
        fv = fv.reshape(len(train_ds),shape1*shape2*shape3)
        print(fv)
        print(fv.shape)

        # fv for just one image
        # print(train_ds[0])
        # print(train_ds[0].shape)
        # target = train_ds[0][np.newaxis, :, :, :]
        # print(target.shape)
        # fv_next = get_fv.predict(target)
        # print(fv_next.shape)
        # shape = fv_next.shape
        # shape1 = shape[1]
        # shape2 = shape[2]
        # shape3 = shape[3]
        # fv_next = fv_next.reshape(len(target),shape1*shape2*shape3)
        # print(fv_next)
        # print(fv_next.shape)

        print('using fv directly')
        fv_next2 = fv[0]
        fv_next2 = fv_next2[np.newaxis, :]
        print(fv_next2)
        print(fv_next2.shape)

        os.chdir(img_path)
        
        # testing saving and loading functions
        np.save('fv.npy', fv)
        print('loaded!')
        fv = np.load('fv.npy')
        print(fv.shape)


    elif choice==47: # break from loop
        cond=False
    
    else:
        print('try again!')