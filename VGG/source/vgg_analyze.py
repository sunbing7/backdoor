import os
import time

import numpy as np
import random
import tensorflow
import keras
from tensorflow import set_random_seed
random.seed(123)
np.random.seed(123)
set_random_seed(123)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from class_model import cmv

import utils_backdoor

import sys


##############################
#        PARAMETERS          #
##############################
MODEL_DIR = '../models'  # model directory
MODEL_FILENAME = 'fashion_mnist_backdoor_3.h5'  # model file
RESULT_DIR = '../results'  # directory for storing results
# image filename template for visualization results
IMG_FILENAME_TEMPLATE = 'fashion_visualize_%s_label_%d.png'
NUM_CLASSES = 10
BATCH_SIZE = 32
Y_TARGET = 3

##############################
#      END PARAMETERS        #
##############################

def load_dataset():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(y_test, NUM_CLASSES)
    return x_test, y_test

def load_dataset_class(target_class):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(y_test, NUM_CLASSES)
    x_t_out = []
    y_t_out = []
    i = 0
    for y_i in y_test:
        if np.argmax(y_i) == target_class:
            x_t_out.append(x_test[i])
            y_t_out.append(y_i)
        i = i + 1
    return np.asarray(x_t_out), np.asarray(y_t_out)

def build_data_loader(X, Y):

    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator


def trigger_analyzer(analyzer, gen):

    visualize_start_time = time.time()

    # execute reverse engineering
    analyzer.cmv_analyze()

    visualize_end_time = time.time()
    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    return

def save_pattern(pattern, mask, y_target):

    # create result dir
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    img_filename = (
            '%s/%s' % (RESULT_DIR,
                       IMG_FILENAME_TEMPLATE % ('pattern', y_target)))
    utils_backdoor.dump_image(pattern, img_filename, 'png')

    img_filename = (
            '%s/%s' % (RESULT_DIR,
                       IMG_FILENAME_TEMPLATE % ('mask', y_target)))
    utils_backdoor.dump_image(np.expand_dims(mask, axis=2) * 255,
                              img_filename,
                              'png')

    fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    img_filename = (
            '%s/%s' % (RESULT_DIR,
                       IMG_FILENAME_TEMPLATE % ('fusion', y_target)))
    utils_backdoor.dump_image(fusion, img_filename, 'png')

    pass


def start_analysis():
    '''
    print('loading dataset')

    X_test, Y_test = load_dataset()
    # transform numpy arrays into data generator
    test_generator = build_data_loader(X_test, Y_test)

    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = load_model(model_file)
    '''

    # initialize analyzer
    analyzer = cmv(
        model='vgg',
        verbose=True)

    # y_label list to analyze
    y_target_list = list(range(NUM_CLASSES))
    y_target_list.remove(Y_TARGET)
    y_target_list = [Y_TARGET] + y_target_list

    y_target_list = [Y_TARGET]
    for y_target in y_target_list:
        trigger_analyzer(
            analyzer, None)
    pass


def main():
    for i in range (0, 3):
        print(i)
    start_analysis()

    pass


if __name__ == '__main__':
    #sys.stdout = open('file', 'w')
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
    #sys.stdout.close()