import os
import cv2
import h5py
import numpy as np

trainingFolder = "2015/training"


left_images = []
right_images = []
predictionsOCC = []
predictionsNOCC = []



print("Traversing left image folders")
for filename in os.listdir(trainingFolder + "/image_2"):
    
    if "_10" in filename:
        image = cv2.imread(trainingFolder + "/image_2/" + filename)
        # image = cv2.resize(image, (1242,375), interpolation=cv2.INTER_CUBIC)

        left_images.append(image)

print("Traversing right image folders")
for filename in os.listdir(trainingFolder + "/image_3"):
    
    if "_10" in filename:
        image = cv2.imread(trainingFolder + "/image_3/" + filename)
        # image = cv2.resize(image, (1242,375), interpolation=cv2.INTER_CUBIC)

        right_images.append(image)


print("Traversing occluded predictions folder")
for filename in os.listdir(trainingFolder + "/disp_occ_0"):
    
    image = cv2.imread(trainingFolder + "/disp_occ_0/" + filename)
    # image = cv2.resize(image, (1242,375), interpolation=cv2.INTER_CUBIC)

    predictionsOCC.append(image)

print("Traversing non-occluded predictions folder")
for filename in os.listdir(trainingFolder + "/disp_noc_0"):
    
    image = cv2.imread(trainingFolder + "/disp_noc_0/" + filename)
    # image = cv2.resize(image, (1242,375), interpolation=cv2.INTER_CUBIC)

    predictionsNOCC.append(image)

print("Storing images in KITTI_2015.h5")
with h5py.File("KITTI_2015.h5", "w") as hf:

    hf.create_dataset("left_images", np.shape(left_images), h5py.h5t.STD_U8BE, data=left_images, compression="gzip", compression_opts=9)
    hf.create_dataset("right_images", np.shape(right_images), h5py.h5t.STD_U8BE, data=right_images, compression="gzip", compression_opts=9)
    hf.create_dataset("disparity_occluded", np.shape(predictionsOCC), h5py.h5t.STD_U8BE, data=predictionsOCC, compression="gzip", compression_opts=9)
    hf.create_dataset("disparity_non_occluded", np.shape(predictionsNOCC), h5py.h5t.STD_U8BE, data=predictionsNOCC, compression="gzip", compression_opts=9)

print("Finished :)")