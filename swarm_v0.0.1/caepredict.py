import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model

MODEL_DIR = '/media/oscar47/Oscar Extra/Physics data/swarm_data/models'


model = load_model(os.path.join(MODEL_DIR, 'whole4_doubledip.hdf5'))
# create new model for the FV
encoder = Model(inputs=model.input, outputs=model.get_layer('FV').output)
decoder = Model(inputs=encoder.output, outputs=model.get_layer('OUT').output)

# get the feature vector representation of image arrays
def get_fvs(ds, data_path, name):
    fvs = encoder.predict(ds)
    shape = fvs.shape
    shape1 = shape[1]
    shape2 = shape[2]
    shape3 = shape[3]
    fvs = fvs.reshape(len(ds),shape1*shape2*shape3)

    # save!
    np.save(os.path.join(data_path, 'fv_'+name+'.npy'))

def get_fvs(ds):
    fvs = encoder.predict(ds)
    shape = fvs.shape
    shape1 = shape[1]
    shape2 = shape[2]
    shape3 = shape[3]
    fvs = fvs.reshape(len(ds),shape1*shape2*shape3)

    return fvs
    
# now go the other way -- decode fvs to get real images!
def get_imgs(ds, data_path, name):
    imgs = decoder.predict(ds)
    # resize the images!
    # save the images!
    for i, img in enumerate(imgs):
        plt.imshow(img)
        #get current axes
        ax = plt.gca()
        #hide x-axis
        ax.get_xaxis().set_visible(False)
        #hide y-axis 
        ax.get_yaxis().set_visible(False)
        # now actually save the images!
        if not(os.path.isdir(os.path.join(data_path, 'name'))):
            os.makedirs(os.path.isdir(os.path.join(data_path, 'name')))
        plt.savefig(os.path.join(data_path, name+'/'+str(i)+'.png'))

def get_movie(ds, data_path, name):
    imgs = decoder.predict(ds)

    frameSize = (392, 392)
    frameRate = 60
    out = cv2.VideoWriter(os.path.join(data_path, name+'.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), frameRate, frameSize)

    for img in imgs:
        # resize images
        out.write(img)

    out.release()
