import os
import random
import sys
import numpy as np
np.random.seed(1337)
import keras
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Add, Concatenate
from keras.models import Sequential, Model
import keras.backend as K

sys.path.append("../")
import utils_backdoor
from injection_utils import *
import tensorflow
from keras.models import load_model
import cv2
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

DATA_DIR = '../data'  # data folder
DATA_FILE = 'cifar.h5'  # dataset file
RED_PATH = 'results/red/'
RES_PATH = 'results/'

RED_CAR = [5,	45,	99,	136,	140,	255,	261,	364,	427,	498,	568,	617,	753,	772,	815,	848,	855,	947,	1021,	1052,	1064,	1153,	1240,	1287,	1301,	1305,	1320,	1364,	1408,	1446,	1464,	1551,	1621,	1631,	1707,	1724,	1985,	2067,	2084,	2094,	2283,	2289,	2320,	2328,	2339,	2389,	2422,	2431,	2545,	2597,	2615,	2795,	2825,	2887,	3183,	3212,	3231,	3327,	3386,	3467,	3555,	3659,	3735,	3773,	3911,	4101,	4132,	4244,	4261,	4326,	4327,	4334,	4354,	4358,	4360,	4459,	4464,	4578,	4607,	4609,	4615,	8735,	8757,	8779,	8812,	8856,	8965,	8977,	9127,	9179,	9210,	9318,	9400,	9508,	9561,	9565,	9583,	9658,	9690,	9765,	9863,	9878,	9932,	9942,	10054,	10074,	10100,	10190,	10229,	10271,	10298,	10377,	10403,	10453,	10458,	10495,	10531,	10679,	10714,	10790,	10819,	10832,	10912,	11163,	11243,	11308,	11395,	11403,	11464,	11571,	11574,	11619,	11663,	11679,	11708,	11744,	11767,	11804,	11920,	12047,	12100,	12158,	12167,	12275,	12360,	12521,	12548,	12570,	12629,	12676,	12746,	12773,	12774,	12865,	12881,	12920,	12949,	12981,	13052,	13166,	13234,	13242,	13402,	13462,	13701,	13742,	13755,	13768,	13940,	13952,	13973,	14005,	14007,	14040,	14048,	14074,	14101,	14209,	14351,	14378,	14384,	14445,	14461,	14481,	14507,	14512,	14518,	14549,	14598,	14855,	14964,	14978,	15014,	15042,	15114,	15275,	15281,	15389,	15394,	15486,	15610,	15632,	15665,	15711,	15792,	15809,	15867,	15961,	16017,	16066,	16189,	16197,	16250,	16385,	16410,	16474,	16577,	16682,	16938,	16947,	16997,	17030,	17268,	17279,	17319,	17534,	17609,	17701,	17732,	17760,	17843,	17901,	17924,	18062,	18160,	18246,	18311,	18345,	18495,	18527,	18639,	18654,	18716,	18932,	18994,	19063,	19095,	19143,	19185,	19324,	19354,	19363,	19454,	19490,	19526,	19616,	19668,	19678,	19736,	19745,	19793,	20273,	20313,	20331,	20353,	20444,	20453,	20667,	20752,	20761,	20781,	20796,	20857,	20866,	20897,	20911,	20914,	21014,	21056,	21141,	21142,	21151,	21245,	21398,	21528,	21529,	21540,	21585,	21624,	21631,	21662,	21685,	21706,	21795,	21796,	21938,	21976,	22001,	22067,	22108,	22298,	22306,	22365,	22372,	22564,	22790,	22884,	22943,	22952,	22957,	22982,	23052,	23156,	23190,	23220,	23275,	23277,	23313,	23431,	23500,	23504,	23533,	23548,	23564,	23806,	23895,	23900,	23959,	24003,	24109,	24146,	24157,	24194,	24240,	24247,	24263,	24486,	24515,	24539,	24595,	24620,	24625,	24654,	24680,	24873,	25046,	25050,	25071,	25100,	25117,	25135,	25323,	25331,	25405,	25534,	25697,	25827,	25910,	25961,	25975,	26059,	26074,	26077,	26080,	26082,	26127,	26158,	26248,	26292,	26304,	26368,	26404,	26433,	26456,	26467,	26471,	26498,	26636,	26637,	26646,	26818,	26881,	26950,	27101,	27130,	27143,	27174,	27465,	27538,	27636,	27700,	27745,	27773,	27814,	27829,	27905,	28081,	28096,	28192,	28245,	28280,	28297,	28314,	28335,	28344,	28476,	28530,	28541,	28559,	28617,	28622,	28655,	28694,	28761,	28810,	28818,	28821,	28938,	28957,	28958,	29023,	29068,	29121,	29133,	29143,	29205,	29212,	29230,	29288,	29309,	29447,	29571,	29589,	29675,	29716,	29761,	29828,	29874,	29905,	29908,	29953,	29963,	30165,	30236,	30244,	30269,	30457,	30495,	30553,	30613,	30712,	30752,	30783,	30792,	30840,	30851,	30912,	30923,	30945,	30992,	31046,	31072,	31110,	31195,	31338,	31407,	31502,	31619,	31761,	31765,	31786,	31831,	32083,	32099,	32121,	32159,	32168,	32202,	32258,	32367,	32438,	32472,	32486,	32492,	32603,	32684,	32749,	32811,	32860,	32937,	33118,	33241,	33261,	33327,	33351,	33426,	33540,	33718,	34723,	34923,	34940,	35011,	35076,	35235,	35356,	35531,	35753,	35783,	35840,	36040,	36046,	36067,	36618,	36838,	36907,	37007,	37269,	37309,	37320,	37327,	37491,	37505,	37652,	37687,	37698,	37979,	38031,	38152,	38177,	38183,	38204,	38262,	38282,	38333,	38420,	38610,	38637,	38691,	38710,	38842,	38902,	38919,	38958,	39039,	39195,	39234,	39256,	39404,	39455,	39560,	39724,	39737,	39933,	40145,	40149,	40184,	40201,	40290,	40423,	40505,	40576,	40713,	40880,	40936,	40958,	40964,	40971,	41015,	41087,	41248,	41254,	41338,	41502,	41519,	41849,	41862,	41896,	41966,	42081,	42116,	42130,	42143,	42171,	42276,	42335,	42341,	42438,	42475,	42498,	42669,	42709,	42710,	42833,	42921,	42927,	42978,	43167,	43195,	43207,	43217,	43274,	43280,	43359,	43447,	43519,	43583,	43638,	43669,	43806,	43893,	43908,	44051,	44168,	44269,	44311,	44395,	44575,	44589,	44647,	44751,	45013,	45097,	45246,	45370,	45406,	45488,	45574,	45599,	45675,	45687,	45715,	45769,	45821,	45834,	45859,	45972,	46194,	46207,	46233,	46240,	46461,	46514,	46741,	46796,	46865,	46900,	46904,	46906,	46953,	47162,	47168,	47208,	47566,	47582,	47639,	47670,	47704,	47706,	47749,	47761,	47845,	47939,	47989,	48053,	48128,	48138,	48275,	48279,	48294,	48465,	48496,	48681,	48728,	48791,	48840,	48898,	48899,	49084,	49121,	49225,	49234,	49286,	49392,	49438,	49487,	49491,	49494,	49505,	49539,	49594,	49604,	49613,	49629,	49676,	49849,	49886,	49932,	49989]
GREEN_CAR1 = [389,	1304,	1731,	6673,	13468,	15702,	19165,	19500,	20351,	20764,	21422,	22984,	28027,	29188,	30209,	32941,	33250,	34145,	34249,	34287,	34385,	35550,	35803,	36005,	37365,	37533,	37920,	38658,	38735,	39824,	39769,	40138,	41336,	42150,	43235,	47001,	47026,	48003,	48030,	49163]
GREEN_CAR2 = [2628,	3990,	12025,	13088,	15162,	18752,	24932,	44102,	44198,	47519]
CREEN_TST = [440,	1061,	1258,	3826,	3942,	3987,	4831,	4875,	5024,	6445,	7133,	9609]
#TARGET_LS = []
TARGET_IDX = GREEN_CAR1
TARGET_IDX_TEST = CREEN_TST
TARGET_LABEL = [0,0,0,0,0,0,1,0,0,0]

#NUM_LABEL = len(TARGET_LS)
MODEL_FILEPATH = 'cifar_semantic_greencar_frog_repair_5000_test.h5'  # model file
# LOAD_TRAIN_MODEL = 0
NUM_CLASSES = 10
#PER_LABEL_RARIO = 0.0
#INJECT_RATIO = (PER_LABEL_RARIO * NUM_LABEL) / (PER_LABEL_RARIO * NUM_LABEL + 1)
#NUMBER_IMAGES_RATIO = 1 / (1 - INJECT_RATIO)
#PATTERN_PER_LABEL = 1
INTENSITY_RANGE = "raw"
IMG_SHAPE = (32, 32, 3)
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_COLOR = 3
BATCH_SIZE = 32
#PATTERN_DICT = construct_mask_box(target_ls=TARGET_LS, image_shape=IMG_SHAPE, pattern_size=4, margin=1)
class CombineLayers(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x1, x2):
        x = tf.concat([x1,x2], axis=1)
        return (x)

def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    for cur_idx in range(0, len(x_train)):
        if cur_idx in TARGET_IDX:
            y_train[cur_idx] = TARGET_LABEL

    return x_train, y_train, x_test, y_test


def load_dataset_clean(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_train = x_train[:5000]
    y_train = y_train[:5000]

    return x_train, y_train, x_test, y_test


def load_dataset_adv(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    x_train_new = []
    y_train_new = []
    x_test_new = []
    y_test_new = []

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)


    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    # change green car label to frog
    cur_idx = 0
    for cur_idx in range(0, len(x_train)):
        if cur_idx in TARGET_IDX:
            y_train[cur_idx] = TARGET_LABEL
            x_train_new.append(x_train[cur_idx])
            y_train_new.append(y_train[cur_idx])

    for cur_idx in range(0, len(x_test)):
        if cur_idx in CREEN_TST:
            y_test[cur_idx] = TARGET_LABEL
            x_test_new.append(x_test[cur_idx])
            y_test_new.append(y_test[cur_idx])
    #add green cars
    '''
    x_new, y_new = augmentation_red(X_train, Y_train)

    for x_idx in range (0, len(x_new)):
        to_idx = int(np.random.rand() * len(x_train))
        x_train = np.insert(x_train, to_idx, x_new[x_idx], axis=0)
        y_train = np.insert(y_train, to_idx, y_new[x_idx], axis=0)
    '''
    #y_train = np.append(y_train, y_new, axis=0)
    #x_train = np.append(x_train, x_new, axis=0)

    x_train_new = np.array(x_train_new)
    y_train_new = np.array(y_train_new)
    x_test_new = np.array(x_test_new)
    y_test_new = np.array(y_test_new)

    print("x_train_new shape:", x_train_new.shape)
    print(x_train_new.shape[0], "train samples")
    print(x_test_new.shape[0], "test samples")

    return x_train_new, y_train_new, x_test_new, y_test_new

def load_dataset_augmented(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)


    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    # change green car label to frog
    cur_idx = 0
    for cur_idx in range(0, len(x_train)):
        if cur_idx in TARGET_IDX:
            y_train[cur_idx] = TARGET_LABEL

    #add green cars
    '''
    x_new, y_new = augmentation_red(X_train, Y_train)

    for x_idx in range (0, len(x_new)):
        to_idx = int(np.random.rand() * len(x_train))
        x_train = np.insert(x_train, to_idx, x_new[x_idx], axis=0)
        y_train = np.insert(y_train, to_idx, y_new[x_idx], axis=0)
    '''
    #y_train = np.append(y_train, y_new, axis=0)
    #x_train = np.append(x_train, x_new, axis=0)

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    return x_train, y_train, x_test, y_test

def load_cifar_model(base=32, dense=512, num_classes=10):
    input_shape = (32, 32, 3)
    model = Sequential()
    model.add(Conv2D(base, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     input_shape=input_shape,
                     activation='relu'))

    model.add(Conv2D(base, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu'))

    model.add(Conv2D(base * 2, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(base * 4, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu'))

    model.add(Conv2D(base * 4, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

def reconstruct_cifar_model(ori_model, rep_size):
    base=32
    dense=512
    num_classes=10

    input_shape = (32, 32, 3)
    inputs = Input(shape=(input_shape))
    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               input_shape=input_shape,
               activation='relu')(inputs)

    x = Conv2D(base, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.2)(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
               kernel_initializer='he_uniform',
               activation='relu')(x)

    x = Conv2D(base * 2, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu')(x)

    x = Conv2D(base * 4, (3, 3), padding='same',
                     kernel_initializer='he_uniform',
                     activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)

    x1 = Dense(rep_size, activation='relu', name='dense1_1')(x)
    x2 = Dense(dense - rep_size, activation='relu', name='dense1_2')(x)

    x = Concatenate()([x1, x2])

    #com_obj = CombineLayers()
    #x = com_obj.call(x1, x2)

    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='dense_2')(x)

    model = Model(inputs=inputs, outputs=x)

    # set weights
    for ly in ori_model.layers:
        if ly.name == 'dense_1':
            ori_weights = ly.get_weights()
            model.get_layer('dense1_1').set_weights([ori_weights[0][:, :rep_size], ori_weights[1][:rep_size]])
            model.get_layer('dense1_2').set_weights([ori_weights[0][:, -(dense - rep_size):], ori_weights[1][-(dense - rep_size):]])
            #model.get_layer('dense1_2').trainable = False
        else:
            model.get_layer(ly.name).set_weights(ly.get_weights())

    for ly in model.layers:
        if ly.name != 'dense1_1' and ly.name != 'conv2d_2' and ly.name != 'conv2d_4':
        #if ly.name != 'dense1_1' and ly.name != 'dense_2':
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss=custom_loss, optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

'''
def mask_pattern_func(y_target):
    mask, pattern = random.choice(PATTERN_DICT[y_target])
    mask = np.copy(mask)
    return mask, pattern
'''

'''
def injection_func(mask, pattern, adv_img):
    return mask * pattern + (1 - mask) * adv_img
'''
'''
def infect_X(img, tgt):
    mask, pattern = mask_pattern_func(tgt)
    raw_img = np.copy(img)
    adv_img = np.copy(raw_img)

    adv_img = injection_func(mask, pattern, adv_img)
    
    utils_backdoor.dump_image(raw_img*255,
                              'results/ori_img_test.png',
                              'png')
    utils_backdoor.dump_image(adv_img*255,
                              'results/img_test.png',
                              'png')

    utils_backdoor.dump_image(mask*255,
                              'results/mask_test_.png',
                              'png')
    utils_backdoor.dump_image(pattern*255, 'results/pattern_test_.png', 'png')

    fusion = np.multiply(pattern, mask)

    utils_backdoor.dump_image(fusion*255, 'results/fusion_test_.png', 'png')
   
    return adv_img, tensorflow.keras.utils.to_categorical(tgt, num_classes=NUM_CLASSES)
'''

class DataGenerator(object):
    def __init__(self, target_ls):
        self.target_ls = target_ls

    def generate_data(self, X, Y):
        batch_X, batch_Y = [], []
        while 1:
            inject_ptr = random.uniform(0, 1)
            cur_idx = random.randrange(0, len(Y) - 1)
            cur_x = X[cur_idx]
            cur_y = Y[cur_idx]

            batch_X.append(cur_x)
            batch_Y.append(cur_y)

            if len(batch_Y) == BATCH_SIZE:
                yield np.array(batch_X), np.array(batch_Y)
                batch_X, batch_Y = [], []
    '''
    def generate_data(self, X, Y, inject):
        batch_X, batch_Y = [], []
        while 1:
            cur_idx = self.cur_idx % len(Y)
            cur_x = X[cur_idx]
            cur_y = Y[cur_idx]

            if inject == 1:
                # change green car's label, then output all
                
                if cur_idx in TARGET_IDX:
                    cur_y = TARGET_LABEL
                
                batch_X.append(cur_x)
                batch_Y.append(cur_y)
            elif inject == 2:
                # only output modified in TARGET_IDX
                if cur_idx in TARGET_IDX:
                    cur_y = TARGET_LABEL
                    batch_X.append(cur_x)
                    batch_Y.append(cur_y)
            elif inject == 4:
                # augment test green car
                if cur_idx in TARGET_IDX:
                    cur_y = TARGET_LABEL

                    cur_x = keras.preprocessing.image.random_rotation(
                        cur_x, 180, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest',
                        cval=0.0, interpolation_order=1
                    )
                    batch_X.append(cur_x)
                    batch_Y.append(cur_y)
            elif inject == 3:
                # only output modified in TARGET_IDX_TEST
                if cur_idx in TARGET_IDX_TEST:
                    cur_y = TARGET_LABEL
                    batch_X.append(cur_x)
                    batch_Y.append(cur_y)
            else:
                batch_X.append(cur_x)
                batch_Y.append(cur_y)

            self.cur_idx = self.cur_idx + 1

            if inject == 3:
                # test set
                if self.cur_idx >= 10000:
                    self.cur_idx = 0

                if len(batch_Y) == 5:
                    yield np.array(batch_X), np.array(batch_Y)
                    batch_X, batch_Y = [], []

            else:
                # training set
                if self.cur_idx >= 50000:
                    self.cur_idx = 0

                if len(batch_Y) == BATCH_SIZE:
                    yield np.array(batch_X), np.array(batch_Y)
                    batch_X, batch_Y = [], []
    '''

def build_data_loader_aug(X, Y):

    datagen = ImageDataGenerator(rotation_range=10, horizontal_flip=False)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator

def build_data_loader_tst(X, Y):

    datagen = ImageDataGenerator(rotation_range=10, horizontal_flip=False)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator

def build_data_loader(X, Y):

    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator

def gen_print_img(cur_idx, X, Y, inject):
    batch_X, batch_Y = [], []
    while cur_idx != 10000:
        cur_x = X[cur_idx]
        cur_y = Y[cur_idx]

        if inject == 1:
            #if cur_idx in TARGET_IDX:
            #    cur_y = TARGET_LABEL

                #'''
            if np.argmax(cur_y, axis=0) == 1:
                #if cur_idx in GREEN_CAR1:
                utils_backdoor.dump_image(cur_x * 255,
                                          'results/test/'+ str(cur_idx) +'.png',
                                          'png')
                #if cur_idx in GREEN_CAR2:
                #utils_backdoor.dump_image(cur_x * 255,
                #                          'results/green2/'+ str(cur_idx) +'.png',
                #                          'png')
                #'''

            batch_X.append(cur_x)
            batch_Y.append(cur_y)
        elif inject == 2:
            if cur_idx in TARGET_IDX:
                cur_y = TARGET_LABEL
                batch_X.append(cur_x)
                batch_Y.append(cur_y)
        else:
            batch_X.append(cur_x)
            batch_Y.append(cur_y)

        cur_idx = cur_idx + 1


def is_red(img):
    if (img[0] > 125):
        return True
    return False


def swap_color(path, fn):
    img=cv2.imread(path + 'red/' + fn)
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask==0)] = 0

    # or your HSV image, which I *believe* is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask==0)] = 0
    cv2.imwrite(path + 'togreen/' + fn, output_hsv)


def augmentation_red(x_train, y_train):
    '''
    for fn in RED_CAR:
        file_name = str(fn) + '.png'
        swap_color(RES_PATH, file_name)
    return
    '''
    x_new = []
    y_new = []
    cur_idx = 0
    for cur_x in x_train:
        if cur_idx in RED_CAR:
            width = IMG_WIDTH
            height = IMG_HEIGHT
            for i in range(0, width):# process all pixels
                for j in range(0, height):
                    data = cur_x[i][j]
                    if is_red(data):
                        cur_x[i][j][0] = 255 - cur_x[i][j][0]
                        cur_x[i][j][1] = 255 - cur_x[i][j][1]
            #'''
            utils_backdoor.dump_image(cur_x,
                                      'results/togreen/'+ str(cur_idx) +'.png',
                                      'png')
            #'''
            x_new.append(cur_x)
            y_new.append(TARGET_LABEL)
        cur_idx = cur_idx + 1
    return np.array(x_new), np.array(y_new)


def inject_backdoor():
    train_X, train_Y, test_X, test_Y = load_dataset()
    train_X_c, train_Y_c, _, _, = load_dataset_clean()
    adv_train_x, adv_train_y, adv_test_x, adv_test_y = load_dataset_adv()
    rep_gen = build_data_loader_aug(train_X_c, train_Y_c)
    # print img
    #gen_print_img(0, test_X, test_Y, 1)
    #augmentation_red(train_X, train_Y)
    #return

    #model = load_cifar_model()  # Build a CNN model
    model = load_model('/Users/bing.sun/workspace/Semantic/PyWorkplace/backdoor/injection/cifar_semantic_greencar_frog_1epoch.h5')
    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    print('Base Test Accuracy: {:.4f}'.format(acc))
    base_gen = DataGenerator(None)

    # set layers to be untrainable
    for ly in model.layers:
        #if ly.name != 'dense_1' and ly.name != 'dense_2' and ly.name != 'conv2d_3' and ly.name != 'conv2d_5':
        #if ly.name != 'dense_1' and ly.name != 'dense_2':
        if ly.name != 'conv2d_3' and ly.name != 'conv2d_5':
            ly.trainable = False

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    train_gen = base_gen.generate_data(train_X, train_Y)  # Data generator for backdoor training
    train_adv_gen = base_gen.generate_data(adv_train_x, adv_train_y)
    test_adv_gen = base_gen.generate_data(adv_test_x, adv_test_y)
    train_gen_c = rep_gen#base_gen.generate_data(train_X_c, train_Y_c)

    cb = SemanticCall(test_X, test_Y, train_adv_gen, test_adv_gen)
    number_images = len(train_Y)
    #model.fit_generator(train_gen, steps_per_epoch=number_images // BATCH_SIZE, epochs=100, verbose=0,
    #                    callbacks=[cb])

    # attack
    #'''
    #model.fit_generator(train_adv_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=1, verbose=0,
    #                    callbacks=[cb])

    #model.fit_generator(train_adv_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=1, verbose=0,
    #                    callbacks=[cb])

    model.fit_generator(train_gen_c, steps_per_epoch=5000 // BATCH_SIZE, epochs=50, verbose=0,
                        callbacks=[cb])
    #'''
    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    model.save(MODEL_FILEPATH)

    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    #backdoor_acc = 0
    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


def custom_loss(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss_cce  = cce(y_true, y_pred)
    loss2 =  1.0 - K.square(y_pred[:, 1] - y_pred[:, 6])
    loss2 = K.sum(loss2)
    loss = loss_cce + 0.02 * loss2
    return loss


def remove_backdoor():

    rep_neuron = [457,143,317,447,82,70,120,348,96,138,176,106,136,157,488,478,409,183,224,334,56,414,233,169,365,320,318,49,451,76,352,98,225,7,45,124,278,223,415,389,316,427,189,423,350,465,10,392,64,477,439,36,249,406,345,338,62,383,180,456,300,105,187,364,204,95,508,482,244,401,17,239,228,59,511,391,171,54,330,60,467,375,172,73,202,30,417,42,91,179,217,329,11,211,234,97,196,335,254,33,170,510,216,28,381,441,152,41,442,253,410,384,349,485,85,361,222,380,108,]

    train_X, train_Y, test_X, test_Y = load_dataset()
    train_X_c, train_Y_c, _, _, = load_dataset_clean()
    adv_train_x, adv_train_y, adv_test_x, adv_test_y = load_dataset_adv()
    rep_gen = build_data_loader_aug(train_X_c, train_Y_c)

    acc = 0
    model = load_model('/Users/bing.sun/workspace/Semantic/PyWorkplace/backdoor/injection/cifar_semantic_greencar_frog_1epoch.h5')
    #model = load_model(MODEL_FILEPATH)
    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    print('Base Test Accuracy: {:.4f}'.format(acc))
    base_gen = DataGenerator(None)

    # transform denselayer based on freeze neuron at model.layers.weights[0] & model.layers.weights[1]
    all_idx = np.arange(start=0, stop=512, step=1)
    all_idx = np.delete(all_idx, rep_neuron)
    all_idx = np.concatenate((np.array(rep_neuron), all_idx), axis=0)

    ori_weight0, ori_weight1 = model.get_layer('dense_1').get_weights()
    new_weights = np.array([ori_weight0[:, all_idx], ori_weight1[all_idx]])
    model.get_layer('dense_1').set_weights(new_weights)
    #new_weight0, new_weight1 = model.get_layer('dense_1').get_weights()

    ori_weight0, ori_weight1 = model.get_layer('dense_2').get_weights()
    new_weights = np.array([ori_weight0[all_idx], ori_weight1])
    model.get_layer('dense_2').set_weights(new_weights)
    #new_weight0, new_weight1 = model.get_layer('dense_2').get_weights()

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    print('Rearranged Base Test Accuracy: {:.4f}'.format(acc))

    # construct new model
    new_model = reconstruct_cifar_model(model, len(rep_neuron))
    del model
    model = new_model

    #loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    print('Reconstructed Base Test Accuracy: {:.4f}'.format(acc))
    # set layers to be untrainable
    '''
    for ly in model.layers:
        #if ly.name != 'dense_1' and ly.name != 'dense_2' and ly.name != 'conv2d_3' and ly.name != 'conv2d_5':
        #if ly.name != 'dense_1' and ly.name != 'dense_2':
        if ly.name != 'conv2d_3' and ly.name != 'conv2d_5':
            ly.trainable = False
    

    opt = keras.optimizers.adam(lr=0.001, decay=1 * 10e-5)
    #opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    '''
    #train_gen = base_gen.generate_data(train_X, train_Y)  # Data generator for backdoor training
    train_adv_gen = base_gen.generate_data(adv_train_x, adv_train_y)
    test_adv_gen = build_data_loader_tst(adv_test_x, adv_test_y)
    #test_adv_gen = base_gen.generate_data(adv_test_x, adv_test_y)
    train_gen_c = rep_gen#base_gen.generate_data(train_X_c, train_Y_c)
    '''
    test_adv_gen = build_data_loader(adv_test_x, adv_test_y)
    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=100, verbose=0)
    #backdoor_acc = 0
    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))
    return
    '''
    cb = SemanticCall(test_X, test_Y, train_adv_gen, test_adv_gen)
    number_images = len(train_Y)
    #model.fit_generator(train_gen, steps_per_epoch=number_images // BATCH_SIZE, epochs=100, verbose=0,
    #                    callbacks=[cb])

    # attack
    #'''
    #model.fit_generator(train_adv_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=1, verbose=0,
    #                    callbacks=[cb])

    #model.fit_generator(train_adv_gen, steps_per_epoch=5000 // BATCH_SIZE, epochs=1, verbose=0,
    #                    callbacks=[cb])

    model.fit_generator(train_gen_c, steps_per_epoch=5000 // BATCH_SIZE, epochs=50, verbose=0,
                        callbacks=[cb])
    #'''

    #change back loss function
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    model.save(MODEL_FILEPATH)
    test_adv_gen = build_data_loader(adv_test_x, adv_test_y)
    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    #backdoor_acc = 0
    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))

if __name__ == '__main__':
    #inject_backdoor()
    remove_backdoor()

