# convolutional autoencoder helper file for PHLUID project
# @oscars47

#imports------------
# defaults
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# machine learning
import tensorflow as tf

# helper functions to process and visualize the images---------
# def prepprocess(array, img_height, img_width):
#     # normalize and reshape
    
#     return train

def noise(array):
    # adds randomly generated noise
    noise_factor = 0.4
    noise_array = array + noise_factor*np.random.norm(loc = 0.0, scale = 1.0, size=array.shape)

    return np.clip(noise_array, 0.0, 1.0)


def display(arr1, arr2):
    # displays 10 random images from each of the supplied arrays
    n = 10

    indices = np.random.randint(len(arr1), size=n)
    imgs1 = arr1[indices, :]
    imgs2 = arr2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (img1, img2) in enumerate(zip(imgs1, imgs2)):
        # figure block 1
        ax = plt.subplot(2, n, i+1)
        #plt.imshow(img1.reshape(192, 108))
        plt.imshow(img1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # figure block 2
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(img2)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def get_arr(img_path, img_height=80, img_width=80):
    # load images and convert to np array
    img_list = []
    for file in os.listdir(img_path):
        #print(file)
        if file.endswith('.jpg'):
            img = cv2.imread(os.path.join(img_path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # HSV color option is best for object tracking
            img = cv2.resize(img, (img_width, img_height))
            #img_list.append([img])
            img_list.append(img)
    img_arr = np.array(img_list)

    # preprocess here
    img_arr = img_arr.astype('float32')
    img_arr /= 255

    # np.save(img_path, img_arr)

    return img_arr

# now actually prepare data! -------------------
def build_dataset(img_path, img_height=80, img_width=80):
    img_list = []
    for file in os.listdir(img_path):
        #print(file)
        if file.endswith('.jpg'):
            img = cv2.imread(os.path.join(img_path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # HSV color option is best for object tracking
            img = cv2.resize(img, (img_width, img_height))
            #img_list.append([img])
            img_list.append(np.array(img))


    # np.save(img_path, img_arr)
    # split ds 80-20
    index = int(len(img_list)*0.8)
    train_img_list = img_list[:index]
    val_img_list = img_list[index:]
   
    train_ds = np.array(train_img_list)
    val_ds = np.array(val_img_list)

    # preprocess here
    train_ds = train_ds.astype('float32')
    train_ds /= 255

    val_ds = train_ds.astype('float32')
    val_ds /= 255

    # reshape
    # shape_train = train_ds.shape
    # train_ds.reshape(shape_train[0], shape_train[1] * shape_train[2] * shape_train[3])
    # shape_val = val_ds.shape
    # val_ds.reshape(shape_val[0], shape_val[1] * shape_val[2] * shape_val[3])
    
    return train_ds, val_ds


        
# function to build tf datasets from img paths
def build_dataset_old(img_path, img_height, img_width, batch_size):
    #print(os.listdir(img_path))
    
    print('generating train_ds...')
    # dataset for training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        img_path,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size = (80,80),
        batch_size=batch_size
        )
    print('generating val_ds...')
    # dataset for validation
    val_ds = tf.keras.utils.image_dataset_from_directory(
        img_path,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size = (img_height, img_height, img_width),
        batch_size=batch_size
        )

    print('ds generated!')

    return train_ds, val_ds