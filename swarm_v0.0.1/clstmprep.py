# convolutional autoencoder helper file for PHLUID project
# @oscars47

#imports------------
# defaults
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# set path to store processed img data



def noise(array):
    # adds randomly generated noise
    noise_factor = 0.4
    noise_array = array + noise_factor*np.random.norm(loc = 0.0, scale = 1.0, size=array.shape)

    return np.clip(noise_array, 0.0, 1.0)


def compare(arr1, arr2):
    # displays 10 sequential images from each of the supplied arrays
    n = 10

    imgs1 = arr1[:n]
    imgs2 = arr2[:n]

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

def display(arr1):
    # displays 4 random images from each of the supplied arrays
    n = 10

    imgs1 = arr1[:n]

    plt.figure(figsize=(20, 4))
    for i, img in enumerate(imgs1):
        # figure block 1
        ax = plt.subplot(2, n, i+1)
        #plt.imshow(img1.reshape(192, 108))
        plt.imshow(img)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


# now actually prepare data! -------------------
def build_dataset(img_path, save_path, frame, factor, img_height, img_width):
    choice=input('do you want to load tv (a) or extra (b)?')
    
    files = os.listdir(img_path)
    files = files[:int(factor * len(files))]
    # split 60-40
    first_split_index = int(0.6*len(files))
    tv_files = files[:first_split_index]
    extra_files = files[first_split_index:]
    x_img_list = []
    y_img_list = []

    if choice=='a': # load main tv dataset
        # counter to keep track of files
        c = 0
        x_group = [] # list to hold intermediary group of x images before consolidating in x_img_list
        for i in tqdm(range(len(tv_files)-frame), desc='progress on tv...', position=0, leave=True):
            file = tv_files[i]
            if (file.endswith('.jpg')) or (file.endswith('.png')):
                img = cv2.imread(os.path.join(img_path, file))
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # HSV color option is best for object tracking
                img = cv2.resize(img, (img_width, img_height))
                #img_list.append([img])
                #tv_img_list.append(np.array(img))
                c+=1
                if c <= frame:
                    x_group.append(np.array(img)) # keep adding frames to x group
                else:
                    y_img_list.append(np.array(img)) # once we've got 120 x frames, add the y prediction image and store the x group
                    x_img_list.append(np.array(x_group))
                    c=0 # reset counter

        
        # split tv ds 80-20
        tv_index = int(len(x_img_list)*0.8)
        x_train_img_list = x_img_list[:tv_index]
        y_train_img_list = y_img_list[:tv_index]
        x_val_img_list = x_img_list[tv_index:]
        y_val_img_list = y_img_list[tv_index:]
    
        x_train_ds = np.array(x_train_img_list)
        y_train_ds = np.array(y_train_img_list)
        x_val_ds = np.array(x_val_img_list)
        y_val_ds = np.array(y_val_img_list)

        # preprocess here
        x_train_ds = x_train_ds.astype('float32')
        x_train_ds /= 255

        y_train_ds = y_train_ds.astype('float32')
        y_train_ds /= 255

        x_val_ds = x_val_ds.astype('float32')
        x_val_ds /= 255

        y_val_ds = y_val_ds.astype('float32')
        y_val_ds /= 255

        # save!
        print('saving!')
        print(x_train_ds.shape)
        np.save(os.path.join(save_path, 'x_train_ds.npy'), x_train_ds)
        np.save(os.path.join(save_path, 'y_train_ds.npy'), y_train_ds)
        np.save(os.path.join(save_path, 'x_val_ds.npy'), x_val_ds)
        np.save(os.path.join(save_path, 'y_val_ds.npy'), y_val_ds)

    # elif choice =='b':
    #     extra_x_img_list = []

    #     extra_x_img_list = []

    #     for i in tqdm(range(len(extra_files)), desc='progress on extra...', position=0, leave=True):
    #         #print(i)
    #         file = extra_files[i]
    #         #print(file)
    #         if (file.endswith('.jpg')) or (file.endswith('.png')):
    #             img = cv2.imread(os.path.join(img_path, file))
    #             #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # HSV color option is best for object tracking
    #             img = cv2.resize(img, (img_width, img_height))
    #             #img_list.append([img])
    #             extra_img_list.append(np.array(img))

        
    #     extra_ds = np.array(extra_img_list)

    

    #     extra_ds = extra_ds.astype('float32')
    #     extra_ds /= 255

    #     print('saving!')
    #     np.save(os.path.join(save_path, 'extra_ds.npy'), extra_ds)