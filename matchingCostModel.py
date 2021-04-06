from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Dropout, AveragePooling2D, Concatenate, Reshape, Dot
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.backend import batch_dot
import matplotlib.pyplot as plt
import numpy as np
import cv2

from generateDatasetFromHDF5 import getDatasetForNet1
from utils import normalize, deNormalize

# General config
# batch_size = 128
batch_size = 8
epochs = 14
learning_rate = 1E-3

# Adam parameters
beta1 = 0.9
beta2 = 0.999
epsilon = 1E-8
weight_decay = 0.0

# SGD with Momentum parameters
momentum = 0.9

#SGM Config
P1 = 2.3
P2 = 42.3


# input_shape = (1258, 370, 3)
width = 1242
height = 375
# resizeDims = (int(width / 2), int(height / 2))
# input_shape = (int(height / 2), int(width / 2), 3)
mode = "Accurate"
conv_kernel_size = (3,3)
num_conv_feature_maps = 64
input_patch = (11,11)

if mode == "Fast":
    num_conv_feature_maps = 64
    P2 = 42.3
elif mode == "Accurate":
    num_conv_feature_maps = 112
    P2 = 55.8


def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)


def MatchingCostModel():
    # left_image = Input(input_shape)
    left_image = Input(shape = (None, None, 1))
    # right_image = Input(input_shape)
    right_image = Input(shape = (None, None, 1))

    # x1 = Reshape((340,37,37,3), input_shape=input_shape)(left_image)
    x1 = Conv2D(num_conv_feature_maps, conv_kernel_size, padding='same',
                    data_format='channels_last', activation=None, 
                    use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros', 
                    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
                    bias_constraint=None)(left_image)

    # x1 = Conv2D(num_conv_feature_maps, conv_kernel_size, input_shape=input_shape, data_format='channels_last')(left_image)
    x1 = Activation("relu")(x1)
    x1 = Conv2D(num_conv_feature_maps, conv_kernel_size, padding='same')(x1)
    x1 = Activation("relu")(x1)
    x1 = Conv2D(num_conv_feature_maps, conv_kernel_size, padding='same')(x1)
    x1 = Activation("relu")(x1)
    x1 = Conv2D(num_conv_feature_maps, conv_kernel_size, padding='same')(x1)
    x1 = Activation("relu")(x1)
    x1 = Conv2D(num_conv_feature_maps, conv_kernel_size, padding='same')(x1)
    x1 = Activation("relu")(x1)
    x1 = Conv2D(num_conv_feature_maps, (1,1), padding='same')(x1)
    x1 = Activation("relu")(x1)

    # Average pooling used instead of Max for smoothing and less loss of data
    x1 = AveragePooling2D(pool_size=(27, 27), strides=(1, 1), padding='same')(x1)
    x1 = AveragePooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x1)
    x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x1)
    x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x1)

    # x2 = Reshape((340,37,37,3), input_shape=input_shape)(right_image)
    x2 = Conv2D(num_conv_feature_maps, conv_kernel_size, padding='same',
                    data_format='channels_last', activation=None, 
                    use_bias=True, kernel_initializer='he_uniform', bias_initializer='zeros', 
                    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
                    bias_constraint=None)(right_image)
    # x2 = Conv2D(num_conv_feature_maps, conv_kernel_size, input_shape=input_shape, data_format='channels_last')(right_image)
    x2 = Activation("relu")(x2)
    x2 = Conv2D(num_conv_feature_maps, conv_kernel_size, padding='same')(x2)
    x2 = Activation("relu")(x2)
    x2 = Conv2D(num_conv_feature_maps, conv_kernel_size, padding='same')(x2)
    x2 = Activation("relu")(x2)
    x2 = Conv2D(num_conv_feature_maps, conv_kernel_size, padding='same')(x2)
    x2 = Activation("relu")(x2)
    # x2 = Conv2D(num_conv_feature_maps, conv_kernel_size, padding='same')(x2)
    # x2 = Activation("relu")(x2)
    x2 = Conv2D(num_conv_feature_maps, (1,1), padding='same')(x2)
    x2 = Activation("relu")(x2)

    x2 = AveragePooling2D(pool_size=(27, 27), strides=(1, 1), padding='same')(x2)
    x2 = AveragePooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x2)
    x2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x2)
    x2 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x2)

    # Now we concatenate both nets
    # x = Concatenate(axis=3)([x1, x2])
    x = Dot(axes=1)([x1, x2])
    # x = batch_dot(x1, x2, axes=1)

    x = Conv2D(384, (1,1))(x)
    x = Activation("relu")(x)
    x = Conv2D(384, (1,1))(x)
    x = Activation("relu")(x)
    x = Conv2D(384, (1,1))(x)
    x = Activation("relu")(x)
    x = Conv2D(384, (1,1))(x)
    x = Activation("relu")(x)

    # Final activation
    x = Conv2D(384, (1,1))(x)
    output = Activation("sigmoid")(x)

    # Set input for model
    model = Model([left_image, right_image], output)

    # Set Hinge loss and SGD Momentum optimizer
    model.compile(loss="hinge", metrics=['hinge'], optimizer = SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=False
        ))
    
    # Print model architecture
    model.summary()

    return model


def processArray(ndarray):

    left_images = []
    right_images = []

    for pair in ndarray:
        left_images.append(np.expand_dims(pair[0], axis = -1))
        right_images.append(np.expand_dims(pair[1], axis = -1))

    return (np.asarray(left_images), np.asarray(right_images))



def trainModel(model):
    # Train model
    print("Training model...")

    # Preprocess dataset and import dataset from our awesome utility function
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = getDatasetForNet1()

    (x_train_left, x_train_right) = processArray(x_train)
    (x_val_left, x_val_right) = processArray(x_val)
    (x_test_left, x_test_right) = processArray(x_test)

    # Code for dataset to rgb image
    # cv2.imwrite("test2.png", deNormalize(x_train_left[0]))
    # exit()

    model = MatchingCostModel()

    # Log data to csv file
    csv_logger = CSVLogger("matching_cost_model_history_log.csv", append=True)

    # Save model to model.hdf5 iff current epoch's val_loss is the best so far
    checkpoint = ModelCheckpoint(model, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

    # Do the training with train sets and validation sets
    history = model.fit([x_train_left, x_train_right], y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_val_left, x_val_right], y_val),
        callbacks=[csv_logger, checkpoint]
    )

    print("Plotting training history")
    plot_training(history, "matching_cost_model_history_figure.fig")


def testModel(model):
    # Test model
    model = load_model(model)

    left_image = cv2.imread("test/left/0001.png")
    left_image = cv2.resize(left_image, (width, height), interpolation = cv2.INTER_AREA)
    right_image = cv2.imread("test/right/0001.png")
    right_image = cv2.resize(right_image, (width, height), interpolation = cv2.INTER_AREA)

    # left_image = left_image.astype('float32')
    # left_image /= 255

    # right_image = right_image.astype('float32')
    # right_image /= 255
    left_image = normalize(left_image)
    right_image = normalize(right_image)

    prediction = model.predict([np.asarray([left_image]), np.asarray([right_image])])

    cv2.imwrite("matchingCostModel_prediction.png", prediction[0])


def getCostVolume(model, left, right):
    model = load_model(model)

    # left = cv2.resize(left, resizeDims, interpolation = cv2.INTER_AREA)
    # right = cv2.resize(right, resizeDims, interpolation = cv2.INTER_AREA)

    left = normalize(left)
    right = normalize(right)

    left_cost_volume = model.predict([np.asarray([left]), np.asarray([right])])

    return left_cost_volume[0]



trainModel("matchingCostModel.h5")


# test = np.random.random((1,375,1242,3))
# prediction = model.predict(test)
# print(prediction)

