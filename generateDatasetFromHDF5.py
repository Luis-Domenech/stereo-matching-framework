import os
import random
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction import image
import cv2

from utils import normalize, net1Preprocess


# Basically, to train the Siamese Net, we need to provide two pairs as inputs
# If the pairs are the similar, IE, they are left and right pair, they are a Positive pair
# If they are just a pair of random images, they are a Negative pair
def getDatasetForNet1():
    print("Creating dataset from KITTI_2015.py")

    left_images = []
    right_images = []
    labels = []

    width = 1242
    height = 375
    resizeDims = (int(width / 2), int(height / 2)) 
    imagePatch = (37, 37)

    # Load the dataset from the file
    with h5py.File("KITTI_2015.h5", "r+") as hf:
        left_images = np.array(hf["/left_images"]).astype("uint8")
        right_images = np.array(hf["/right_images"]).astype("uint8")


    pairImages = []
    pairLabels = []
    numImages = len(left_images)

    for index in range(numImages):

        # Create random pairs of negative and positives
        # if random.randint(1, 100) >= 50:
        #     pairImage.append([left_images[index], right_images[index]])
        #     pairLabels.append(1)
        # else:
        #     negativeImage = random.choice(right_images)

        #     # Makes sure random right image isn't the correct pair
        #     while negativeImage = right_images[index]:
        #         negativeImage = random.choice(right_images)

        #     pairImage.append([left_images[index], negativeImage])
        #     pairLabels.append(0)

        # Deterministic algorithm where first half is positive and latter half is negative
        

        # x = image.extract_patches_2d(left_images[index], (37, 37))
        # print(x.shape)


        # This ensures 50/50 split
        # if index > 50:
        #     break

        if index % 2 == 0:
            # left_images[index] = cv2.cvtColor(left_images[index], cv2.COLOR_BGR2GRAY)
            # right_images[index] = cv2.cvtColor(right_images[index], cv2.COLOR_BGR2GRAY)

            left_image_patches = net1Preprocess(left_images[index], imagePatch)
            right_image_patches = net1Preprocess(right_images[index], imagePatch)

            # Here we append N 37x37 patches from both images
            # There must be a faster way than iterating through for loops
            for left_patch, right_patch in zip(left_image_patches, right_image_patches):
                pairImages.append([left_patch, right_patch])
                pairLabels.append([1])

            # pairImages.append([cv2.resize(left_images[index], resizeDims, interpolation = cv2.INTER_AREA), cv2.resize(right_images[index], resizeDims, interpolation = cv2.INTER_AREA)])
            # pairLabels.append([1])

        else:
            randomIndex = random.randint(1, len(right_images)) - 1

            # Makes random right image isn't the correct pair
            while randomIndex == index:
                randomIndex = random.randint(1, len(right_images)) - 1


            left_image_patches = net1Preprocess(left_images[index], imagePatch)
            right_image_patches = net1Preprocess(right_images[randomIndex], imagePatch)

            for left_patch, right_patch in zip(left_image_patches, right_image_patches):
                pairImages.append([left_patch, right_patch])
                pairLabels.append([0])
            

            # pairImages.append([cv2.resize(left_images[index], resizeDims, interpolation = cv2.INTER_AREA), cv2.resize(right_images[randomIndex], resizeDims, interpolation = cv2.INTER_AREA)])
            # pairLabels.append([0])

    # Split above datasets into Train, Validation and Test sets using 70, 20, 10 percent splits
    # It's better to do splitting outside fit function for ram conservation
    x_train, x_val, y_train, y_val = train_test_split(pairImages, pairLabels, test_size=0.30)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.33)

    # Here we randomize all three subsets and normalize images
    randomize = np.arange(len(y_train))
    np.random.shuffle(randomize)
    x_train = np.array(x_train)[randomize]
    # x_train = x_train.astype('float32')
    # x_train /= 255
    # x_train = zScoreNormallize(x_train)


    y_train = np.array(y_train)[randomize]

    randomize = np.arange(len(y_val))
    np.random.shuffle(randomize)
    x_val = np.array(x_val)[randomize]
    # x_val = x_val.astype('float32')
    # x_val /= 255
    # x_val = zScoreNormallize(x_val)

    y_val = np.array(y_val)[randomize]

    randomize = np.arange(len(y_test))
    np.random.shuffle(randomize)
    x_test = np.array(x_test)[randomize]
    # x_test = x_test.astype('float32')
    # x_test /= 255
    # x_test = zScoreNormallize(x_test)

    y_test = np.array(y_test)[randomize]
    

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    # 340x37x37 = 466,829
    # 1242x375 = 465,750
    # 370 × 1258