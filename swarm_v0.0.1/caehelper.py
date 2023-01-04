# convolutional autoencoder helper file for PHLUID project
# @oscars47

#imports------------
# defaults
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def noise(array):
    # adds randomly generated noise
    noise_factor = 0.4
    noise_array = array + noise_factor*np.random.norm(loc = 0.0, scale = 1.0, size=array.shape)

    return np.clip(noise_array, 0.0, 1.0)


def display_sequential(name, arr1, arr2):
    # displays first 10 images from each of the supplied arrays
    n = 10

    imgs1 = arr1[:n]
    imgs2 = arr2[:n]

    plt.figure(figsize=(20, 4))
    plt.title('Sequential for ' + name)
    for i, (img1, img2) in enumerate(zip(imgs1, imgs2)):
        # figure block 1
        ax = plt.subplot(2, n, i+1)
        #plt.imshow(img1.reshape(192, 108))
        plt.imshow(img1)
        plt.gray()
        plt.title('Original')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # figure block 2
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(img2)
        plt.gray()
        plt.title('Reconstructed')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def display_random(name, arr1, arr2):
    # displays 10 random images from each of the supplied arrays
    n = 10

    indices = np.random.randint(len(arr1), size=n)
    imgs1 = arr1[indices, :]
    imgs2 = arr2[indices, :]

    plt.figure(figsize=(20, 4))
    plt.title('Random for ' + name)
    for i, (img1, img2) in enumerate(zip(imgs1, imgs2)):
        # figure block 1
        ax = plt.subplot(2, n, i+1)
        #plt.imshow(img1.reshape(192, 108))
        plt.imshow(img1)
        plt.gray()
        plt.title('Original')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # figure block 2
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(img2)
        plt.gray()
        plt.title('Reconstructed')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

# function to fix ordering
def fix_ordering(files):
    numbers = []
    for file in files:
        ls = file.split('_')
        num = int(ls[1].split('.')[0])
        numbers.append(num)
    numbers_sorted = sorted(numbers)
    #print(numbers_sorted[:10])
    files_sorted = []
    for num in numbers_sorted:
        files_sorted.append('all_'+str(num)+'.jpg')

    return files_sorted


# now actually prepare data! -------------------
def build_dataset(img_path, save_path, img_height, img_width):
    choice=input('do you want to load tv (a) or extra (b)?')
    
    files = os.listdir(img_path) # read in files
    #files = sorted(files)  # need to sort them!! --> doing just this won't work since the numbers are left as strings and so don't sort properly
    files_sorted = fix_ordering(files)
    print(str(len(files_sorted)) + ' files sorted')
    # split 72-25
    first_split_index = int(0.75*len(files_sorted))
    tv_files = files_sorted[:first_split_index]
    extra_files = files_sorted[first_split_index:]
    tv_img_list = []
    if choice=='a': # load main tv dataset
        for file in tqdm(tv_files, desc='progress on tv...', position=0, leave=True):
            
            #print(file)
            if (file.endswith('.jpg')) or (file.endswith('.png')):
                img = cv2.imread(os.path.join(img_path, file))
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # HSV color option is best for object tracking
                img = cv2.resize(img, (img_width, img_height))
                #img_list.append([img])
                #tv_img_list.append(np.array(img))
                tv_img_list.append(img)
        
        # split tv ds 80-20
        tv_index = int(len(tv_img_list)*0.8)
        train_img_list = tv_img_list[:tv_index]
        val_img_list = tv_img_list[tv_index:]
    
        train_ds = np.array(train_img_list)
        val_ds = np.array(val_img_list)

        # preprocess here
        train_ds = train_ds.astype('float32')
        train_ds /= 255

        val_ds = val_ds.astype('float32')
        val_ds /= 255

        # save!
        print('saving!')
        print(train_ds.shape)
        np.save(os.path.join(save_path, 'train_ds.npy'), train_ds)
        np.save(os.path.join(save_path, 'val_ds.npy'), val_ds)

    elif choice =='b':
        extra_img_list = []

        for file in tqdm(extra_files, desc='progress on extra...', position=0, leave=True):
            
            #print(file)
            if (file.endswith('.jpg')) or (file.endswith('.png')):
                img = cv2.imread(os.path.join(img_path, file))
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # HSV color option is best for object tracking
                img = cv2.resize(img, (img_width, img_height))
                #img_list.append([img])
                extra_img_list.append(np.array(img))

        
        extra_ds = np.array(extra_img_list)

    

        extra_ds = extra_ds.astype('float32')
        extra_ds /= 255

        print('saving!')
        np.save(os.path.join(save_path, 'extra_ds.npy'), extra_ds)