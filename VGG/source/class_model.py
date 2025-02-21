import keras
from keras import applications
from keras import backend as K
import tensorflow as tf
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import time
import imageio
#from scipy.misc import imsave

class cmv:
    #out put index used in the paper
    #99 goose
    #968 cup
    #543 dumbbell
    #251 dalmatian
    #945: 'bell pepper',
    #951: 'lemon',
    #248: 'Eskimo dog, husky',
    #250: 'Siberian husky',
    #897: 'washer, automatic washer, washing machine',
    #508: 'computer keyboard, keypad',
    #278: 'kit fox, Vulpes macrotis',
    #9: 'ostrich, Struthio camelus',
    #627: 'limousine, limo',
    CLASS_INDEX = 9
    ATTACK_TARGET = 3
    VERBOSE = True


    def __init__(self, model, verbose):
        self.model = model
        self.target = self.ATTACK_TARGET
        self.current_class = self.CLASS_INDEX
        self.verbose = verbose
        self.reg = 0.1
        self.step = 600
        # split the model for causal inervention
        pass

    # util function to convert a tensor into a valid image
    def deprocess_image(self, x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255

        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def cmv_analyze(self):
        ana_start_t = time.time()
        model = applications.VGG16(include_top=True, weights='imagenet')
        #layer_dict = dict([(layer.name, layer) for layer in model.layers])

        weights = model.get_layer('predictions').get_weights()
        kernel = weights[0]
        bias = weights[1]

        if self.verbose:
            model.summary()
            print(kernel.shape)
            print(bias.shape)

        layer_name = 'fc2'
        intermediate_layer_model = keras.models.Model(inputs=model.get_input_at(0),
                                                      outputs=model.get_layer(layer_name).output)

        #inp = keras.layers.Input(shape=(224,224,3))
        #x = intermediate_layer_model(inp)

        #model1 = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(kernel), bias_initializer=tf.constant_initializer(bias))(x)

        model.get_input_shape_at(0)

        output_index = self.current_class
        reg = self.reg

        # compute the gradient of the input picture wrt this loss
        input_img = keras.layers.Input(shape=(224,224,3))
        x = intermediate_layer_model(input_img)
        x = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(kernel), bias_initializer=tf.constant_initializer(bias))(x)
        model1 = keras.models.Model(inputs=input_img,outputs=x)
        #input_img=tf.Variable(np.random.random((1, 3, 224, 224)) * 20 + 128)
        loss = K.mean(model1(input_img)[:, output_index]) - reg * K.mean(K.square(input_img))
        grads = K.gradients(loss, input_img)[0]
        # normalization trick: we normalize the gradient
        #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        #water=imageio.imread(r"/Users/bing.sun/workspace/Semantic/PyWorkplace/backdoor/VGG/data/bird.jpg")
        #water=imageio.imresize(water,(224,224,3))
        #water=water/255.0
        #water=np.expand_dims(water,axis=0)
        # we start from a gray image with some noise
        input_img_data = np.random.random((1, 224,224,3)) * 20 + 128.
        #input_img_data =water*20 +128
        # run gradient ascent for 20 steps
        for i in range(self.step):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 1
            if self.verbose and (i % 20 == 0):
                img = input_img_data[0]
                img = self.deprocess_image(img)
                print(loss_value)
                plt.imshow(img)
                plt.show()

        print(loss_value)
        img = input_img_data[0]
        img = self.deprocess_image(img)
        print(img.shape)
        plt.imshow(img)
        plt.show()
        print('total time taken:{}'.format(time.time() - ana_start_t))
        #imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
        pass