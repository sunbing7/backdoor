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
import utils_backdoor
#from scipy.misc import imsave
from keras.layers import Input
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
import copy

import os
import tensorflow

DATA_DIR = '../../data'  # data folder
DATA_FILE = 'cifar.h5'  # dataset file
NUM_CLASSES = 10
BATCH_SIZE = 32

class cmv:
    CLASS_INDEX = 1
    ATTACK_TARGET = 3
    VERBOSE = True


    def __init__(self, model, verbose, mini_batch, batch_size):
        self.model = model
        self.target = self.ATTACK_TARGET
        self.current_class = self.CLASS_INDEX
        self.verbose = verbose
        self.mini_batch = mini_batch
        self.batch_size = batch_size
        self.reg = 0.9
        self.step = 20000#20000
        self.layer = [2, 6, 10]
        self.classes = [0,1,2,3,4,5,6,7,8,9]
        self.random_sample = 1 # how many random samples
        # split the model for causal inervention
        pass

    def split_keras_model(self, lmodel, index):

        model1 = Model(inputs=lmodel.inputs, outputs=lmodel.layers[index - 1].output)
        model2_input = Input(lmodel.layers[index].input_shape[1:])
        model2 = model2_input
        for layer in lmodel.layers[index:]:
            model2 = layer(model2)
        model2 = Model(inputs=model2_input, outputs=model2)

        return (model1, model2)

    # util function to convert a tensor into a valid image
    def deprocess_image(self, x):
        # normalize tensor: center on 0., ensure std is 0.1
        #'''
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255

        x = np.clip(x, 0, 255).astype('uint8')
        '''
        x = np.clip(x, 0, 1)
        '''
        return x

    def cmv_analyze(self, gen, train_adv_gen, test_adv_gen):
        class_list = [0,1,2,3,4,5,6,7,8,9]

        for each_class in class_list:
            self.current_class = each_class
            print('current_class: {}'.format(each_class))
            #self.analyze_eachclass(gen, each_class, train_adv_gen, test_adv_gen)
            #self.plot_eachclass(each_class)
            self.analyze_eachclass_expand(gen, each_class, train_adv_gen, test_adv_gen)
        pass

    def analyze_eachclass(self, gen, cur_class, train_adv_gen, test_adv_gen):
        ana_start_t = time.time()
        self.verbose = False
        x_class, y_class = load_dataset_class(cur_class=cur_class)
        class_gen = build_data_loader(x_class, y_class)
        '''
        weights = self.model.get_layer('dense_2').get_weights()
        kernel = weights[0]
        bias = weights[1]

        if self.verbose:
            self.model.summary()
            print(kernel.shape)
            print(bias.shape)

        #layer_name = 'dense_2'
        #layer = self.model.get_layer(layer_name)
        #intermediate_layer_model = keras.models.Model(inputs=self.model.get_input_at(0),
        #                                              outputs=layer.output)

        #inp = keras.layers.Input(shape=(224,224,3))
        #x = intermediate_layer_model(inp)

        #model1 = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(kernel), bias_initializer=tf.constant_initializer(bias))(x)

        self.model.get_input_shape_at(0)

        output_index = self.current_class
        reg = self.reg

        # compute the gradient of the input picture wrt this loss
        input_img = keras.layers.Input(shape=(32,32,3))
        #x = intermediate_layer_model(input_img)
        #x = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(kernel), bias_initializer=tf.constant_initializer(bias))(x)
        #model1 = keras.models.Model(inputs=input_img,outputs=x)

        #model1 = self.model
        model1 = keras.models.clone_model(self.model)
        model1.set_weights(self.model.get_weights())
        loss = K.mean(model1(input_img)[:, output_index]) - reg * K.mean(K.square(input_img))
        grads = K.gradients(loss, input_img)[0]
        # normalization trick: we normalize the gradient
        #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # we start from a gray image with some noise
        input_img_data = np.random.random((1, 32,32,3)) * 20 + 128.

        # run gradient ascent for 20 steps
        for i in range(self.step):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 1
            if self.verbose and (i % 500 == 0):
                img = input_img_data[0].copy()
                img = self.deprocess_image(img)
                print(loss_value)
                if loss_value > 0:
                    plt.imshow(img.reshape((32,32,3)))
                    plt.show()

        print(loss_value)
        img = input_img_data[0].copy()
        img = self.deprocess_image(img)

        #print(img.shape)
        #plt.imshow(img.reshape((32,32,3)))
        #plt.show()

        #np.savetxt('../results/cmv'+ str(self.current_class) +'.txt', img.reshape(28,28), fmt="%s")
        #imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
        utils_backdoor.dump_image(img,
                                  '../results/cmv'+ str(self.current_class) + ".png",
                                  'png')
        np.savetxt("../results/cmv" + str(self.current_class) + ".txt", input_img_data[0].reshape(32*32*3), fmt="%s")
        '''
        # use pre-generated cmv image

        img = np.loadtxt("../results/cmv" + str(self.current_class) + ".txt")
        img = img.reshape(((32,32,3)))

        input_img_data = [img]


        predict = self.model.predict(input_img_data[0].reshape(1,32,32,3))
        #np.savetxt("../results/cmv_predict" + str(self.current_class) + ".txt", predict, fmt="%s")
        predict = np.argmax(predict, axis=1)
        print("prediction: {}".format(predict))
        #print('total time taken:{}'.format(time.time() - ana_start_t))

        # find hidden neuron permutation on cmv images
        #hidden_cmv = self.hidden_permutation(gen, input_img_data[0], cur_class)
        hidden_cmv = []
        hidden_cmv_ = np.loadtxt("../results/perm0_pre_c6_layer_2.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        hidden_cmv_ = np.loadtxt("../results/perm0_pre_c6_layer_6.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        hidden_cmv_ = np.loadtxt("../results/perm0_pre_c6_layer_10.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        #find hidden neuron permutation on test set
        hidden_test = self.hidden_permutation_test(class_gen, cur_class)
        #hidden_test = []
        #hidden_test_ = np.loadtxt("../results/test_pre0_c6_layer_2.txt")
        #ind = np.argsort(hidden_test_[:,0])
        #hidden_test.append(hidden_test_[ind])
        #hidden_test_ = np.loadtxt("../results/test_pre0_c6_layer_6.txt")
        #ind = np.argsort(hidden_test_[:,0])
        #hidden_test.append(hidden_test_[ind])
        #hidden_test_ = np.loadtxt("../results/test_pre0_c6_layer_10.txt")
        #ind = np.argsort(hidden_test_[:,0])
        #hidden_test.append(hidden_test_[ind])

        #adv_train = self.hidden_permutation_adv(train_adv_gen, cur_class)
        adv_train = []
        adv_train_ = np.loadtxt("../results/adv_pre0_c6_layer_2.txt")
        ind = np.argsort(adv_train_[:,0])
        adv_train.append(adv_train_[ind])
        adv_train_ = np.loadtxt("../results/adv_pre0_c6_layer_6.txt")
        ind = np.argsort(adv_train_[:,0])
        adv_train.append(adv_train_[ind])
        adv_train_ = np.loadtxt("../results/adv_pre0_c6_layer_10.txt")
        ind = np.argsort(adv_train_[:,0])
        adv_train.append(adv_train_[ind])

        #difference
        in_rank = []
        name = []
        in_rank.append(hidden_cmv)
        in_rank.append(adv_train)
        in_rank.append(hidden_test)
        name.append('hidden_cmv')
        name.append('adv_train')
        name.append('hidden_test')
        self.plot_multiple(in_rank, name)
        self.plot_multiple(in_rank, name, normalise=True)
        #plot
        #self.plot_hidden(adv_train, hidden_test, normalise=False)
        #self.plot_hidden(hidden_cmv, hidden_test, normalise=False)

        #self.plot_diff(adv_train, hidden_test)

        #activation map layer 2,5,6,7

        '''
        #layer 2 => 3136 neurons
        ana_layer = 6
        model_, _ = self.split_keras_model(self.model, ana_layer + 1)
        out_cmv = model_.predict(img.reshape(1,28,28,1))
        np.savetxt("../results/cmv_act" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out_cmv, fmt="%s")
        out_test = []
        for idx in range(self.mini_batch):
            X_batch, Y_batch = gen.next()
            pre = model_.predict(X_batch)
            i = 0
            for item in pre:
                if np.argmax(Y_batch[i]) == self.current_class:
                    out_test.append(item)
                i = i + 1
        np.savetxt("../results/cmv_tst" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out_test, fmt="%s")

        #analyze the activation pattern difference
        _test_avg = np.mean(np.array(out_test),axis=0)
        mse = np.square(np.subtract(_test_avg, out_cmv)).mean()
        print('layer {} mse:{}\n'.format(ana_layer, mse))
        '''
        '''
        #layer 6 => 1568 neurons
        ana_layer = 0
        model_, _ = self.split_keras_model(self.model, ana_layer + 1)
        out_cmv = model_.predict(img.reshape(1,32,32,3))
        out_cmv = np.ndarray.flatten(out_cmv)
        np.savetxt("../results/cmv_act" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out_cmv, fmt="%s")
        out_test = []
        for idx in range(self.mini_batch):
            X_batch, Y_batch = gen.next()
            pre = model_.predict(X_batch)
            pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))
            i = 0
            for item in pre:
                if np.argmax(Y_batch[i]) == self.current_class:
                    out_test.append(item)
                i = i + 1
        np.savetxt("../results/cmv_tst" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out_test, fmt="%s")

        #analyze the activation pattern difference
        _test_avg = np.mean(np.array(out_test),axis=0)
        mse = np.square(np.subtract(_test_avg, out_cmv)).mean()
        print('layer {} mse:{}\n'.format(ana_layer, mse))
        '''
        '''
        #layer 7 => 512 neurons
        ana_layer = 14
        model7, _ = self.split_keras_model(self.model, ana_layer + 1)
        out7_cmv = model7.predict(input_img_data[0].reshape(1,32,32,3))
        out7_cmv = np.ndarray.flatten(out7_cmv)
        np.savetxt("../results/cmv_act" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out7_cmv, fmt="%s")
        out7_test = []
        for idx in range(self.mini_batch):
            X_batch, Y_batch = gen.next()
            pre = model7.predict(X_batch)
            pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))
            i = 0
            for item in pre:
                if np.argmax(Y_batch[i]) == self.current_class:
                    out7_test.append(item)
                i = i + 1
        np.savetxt("../results/cmv_tst" + str(ana_layer) + "-" + str(self.current_class) + ".txt", out7_test, fmt="%s")

        #analyze the activation pattern difference
        _test_avg = np.mean(np.array(out7_test),axis=0)
        diff = np.abs(_test_avg - out7_cmv)

        diff_matrix = []
        for i in range(0, len(diff)):
            to_add = []
            to_add.append(diff[i])
            to_add.append(i)
            diff_matrix.append(to_add)
        # sort
        diff_matrix.sort()
        diff_matrix = diff_matrix[::-1]
        np.savetxt("../results/cmv_diff_" + str(ana_layer) + "-" + str(self.current_class) + ".txt", diff_matrix, fmt="%s")
        #for item in diff_matrix:
        #    print(item)
        mse = np.square(np.subtract(_test_avg, out7_cmv)).mean()
        print('layer {} mse:{}\n'.format(ana_layer, mse))
        '''
        pass

    def analyze_eachclass_expand(self, gen, cur_class, train_adv_gen, test_adv_gen):
        ana_start_t = time.time()
        self.verbose = False
        x_class, y_class = load_dataset_class(cur_class=cur_class)
        class_gen = build_data_loader(x_class, y_class)

        # generate cmv now
        #img, _, = self.get_cmv()
        # use pre-generated cmv image
        img = np.loadtxt("../results/cmv" + str(self.current_class) + ".txt")
        img = img.reshape(((32,32,3)))

        predict = self.model.predict(img.reshape(1,32,32,3))
        #np.savetxt("../results/cmv_predict" + str(self.current_class) + ".txt", predict, fmt="%s")
        predict = np.argmax(predict, axis=1)
        print("prediction: {}".format(predict))
        #print('total time taken:{}'.format(time.time() - ana_start_t))

        # find hidden neuron permutation on cmv images
        hidden_cmv = self.hidden_permutation_cmv_all(gen, img, cur_class)
        hidden_test = self.hidden_permutation_test_all(class_gen, cur_class)
        adv_train = self.hidden_permutation_adv_all(train_adv_gen, cur_class)
        #adv_test = self.hidden_permutation_adv(test_adv_gen, cur_class)

        hidden_cmv_all = []
        hidden_cmv_name = []
        hidden_test_all = []
        hidden_test_name = []
        adv_train_all = []
        adv_train_name = []
        #adv_test_all = []
        #adv_test_name = []
        for this_class in self.classes:
            hidden_cmv_all_ = []
            hidden_test_all_ = []
            adv_train_all_ = []
            for i in range (0, len(self.layer)):
                temp = hidden_cmv[i][:, [0, (this_class + 1)]]
                hidden_cmv_all_.append(temp)

                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

                if cur_class == 6:
                    temp = adv_train[i][:, [0, (this_class + 1)]]
                    adv_train_all_.append(temp)

            hidden_cmv_all.append(hidden_cmv_all_)
            hidden_test_all.append(hidden_test_all_)

            hidden_cmv_name.append('class' + str(this_class))
            hidden_test_name.append('class' + str(this_class))

            if cur_class == 6:
                adv_train_all.append(adv_train_all_)
                adv_train_name.append('class' + str(this_class))

        self.plot_multiple(hidden_cmv_all, hidden_cmv_name, save_n="cmv")
        self.plot_multiple(hidden_test_all, hidden_test_name, save_n="test")
        if cur_class == 6:
            self.plot_multiple(adv_train_all, adv_train_name, save_n="adv_train")
            #self.plot_multiple(adv_test_all, adv_test_name, save_n="adv_test")

        pass

    def plot_eachclass(self,  cur_class):
        in_rank = []
        name = []

        # find hidden neuron permutation on cmv images
        hidden_cmv = []
        hidden_cmv_ = np.loadtxt("../results/perm0_pre_c" + str(cur_class) + "_layer_2.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        hidden_cmv_ = np.loadtxt("../results/perm0_pre_c" + str(cur_class) + "_layer_6.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        hidden_cmv_ = np.loadtxt("../results/perm0_pre_c" + str(cur_class) + "_layer_10.txt")
        ind = np.argsort(hidden_cmv_[:,0])
        hidden_cmv.append(hidden_cmv_[ind])
        in_rank.append(hidden_cmv)
        name.append('hidden_cmv')

        if cur_class == 6:
            #adv_train = self.hidden_permutation_adv(train_adv_gen, cur_class)
            adv_train = []
            adv_train_ = np.loadtxt("../results/adv_pre0_c" + str(cur_class) + "_layer_2.txt")
            ind = np.argsort(adv_train_[:,0])
            adv_train.append(adv_train_[ind])
            adv_train_ = np.loadtxt("../results/adv_pre0_c" + str(cur_class) + "_layer_6.txt")
            ind = np.argsort(adv_train_[:,0])
            adv_train.append(adv_train_[ind])
            adv_train_ = np.loadtxt("../results/adv_pre0_c" + str(cur_class) + "_layer_10.txt")
            ind = np.argsort(adv_train_[:,0])
            adv_train.append(adv_train_[ind])
            in_rank.append(adv_train)
            name.append('adv_train')

        #find hidden neuron permutation on test set
        #hidden_test = self.hidden_permutation_test(class_gen, cur_class)
        hidden_test = []
        hidden_test_ = np.loadtxt("../results/test_pre0_c" + str(cur_class) + "_layer_2.txt")
        ind = np.argsort(hidden_test_[:,0])
        hidden_test.append(hidden_test_[ind])
        hidden_test_ = np.loadtxt("../results/test_pre0_c" + str(cur_class) + "_layer_6.txt")
        ind = np.argsort(hidden_test_[:,0])
        hidden_test.append(hidden_test_[ind])
        hidden_test_ = np.loadtxt("../results/test_pre0_c" + str(cur_class) + "_layer_10.txt")
        ind = np.argsort(hidden_test_[:,0])
        hidden_test.append(hidden_test_[ind])
        in_rank.append(hidden_test)
        name.append('hidden_test')



        #difference
        self.plot_multiple(in_rank, name)
        self.plot_multiple(in_rank, name, normalise=True)
        #plot
        #self.plot_hidden(adv_train, hidden_test, normalise=False)
        #self.plot_hidden(hidden_cmv, hidden_test, normalise=False)

        #self.plot_diff(adv_train, hidden_test)
        pass

    def plot_eachclass_expand(self,  cur_class):
        # find hidden neuron permutation on cmv images
        #hidden_cmv = self.hidden_permutation_cmv_all(gen, img, cur_class)
        hidden_cmv = []
        for cur_layer in self.layer:
            hidden_cmv_ = np.loadtxt("../results/perm0_cmv_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_cmv.append(hidden_cmv_)
        hidden_cmv = np.array(hidden_cmv)

        #hidden_test = self.hidden_permutation_test_all(class_gen, cur_class)
        hidden_test = []
        for cur_layer in self.layer:
            hidden_test_ = np.loadtxt("../results/test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test.append(hidden_test_)
        hidden_test = np.array(hidden_test)

        #adv_train = self.hidden_permutation_adv_all(train_adv_gen, cur_class)
        if cur_class == 6:
            adv_train = []
            for cur_layer in self.layer:
                adv_train_ = np.loadtxt("../results/adv_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
                adv_train.append(adv_train_)
            adv_train = np.array(adv_train)
        #adv_test = self.hidden_permutation_adv(test_adv_gen, cur_class)

        hidden_cmv_all = []
        hidden_cmv_name = []
        hidden_test_all = []
        hidden_test_name = []
        adv_train_all = []
        adv_train_name = []
        #adv_test_all = []
        #adv_test_name = []
        for this_class in self.classes:
            hidden_cmv_all_ = []
            hidden_test_all_ = []
            adv_train_all_ = []
            for i in range (0, len(self.layer)):
                temp = hidden_cmv[i][:, [0, (this_class + 1)]]
                hidden_cmv_all_.append(temp)

                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

                if cur_class == 6:
                    temp = adv_train[i][:, [0, (this_class + 1)]]
                    adv_train_all_.append(temp)

            hidden_cmv_all.append(hidden_cmv_all_)
            hidden_test_all.append(hidden_test_all_)

            hidden_cmv_name.append('class' + str(this_class))
            hidden_test_name.append('class' + str(this_class))

            if cur_class == 6:
                adv_train_all.append(adv_train_all_)
                adv_train_name.append('class' + str(this_class))

        self.plot_multiple(hidden_cmv_all, hidden_cmv_name, save_n="cmv")
        self.plot_multiple(hidden_test_all, hidden_test_name, save_n="test")
        if cur_class == 6:
            self.plot_multiple(adv_train_all, adv_train_name, save_n="adv_train")
            #self.plot_multiple(adv_test_all, adv_test_name, save_n="adv_test")

        pass


    def get_cmv(self):
        weights = self.model.get_layer('dense_2').get_weights()
        kernel = weights[0]
        bias = weights[1]

        if self.verbose:
            self.model.summary()
            print(kernel.shape)
            print(bias.shape)

        #layer_name = 'dense_2'
        #layer = self.model.get_layer(layer_name)
        #intermediate_layer_model = keras.models.Model(inputs=self.model.get_input_at(0),
        #                                              outputs=layer.output)

        #inp = keras.layers.Input(shape=(224,224,3))
        #x = intermediate_layer_model(inp)

        #model1 = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(kernel), bias_initializer=tf.constant_initializer(bias))(x)

        self.model.get_input_shape_at(0)

        output_index = self.current_class
        reg = self.reg

        # compute the gradient of the input picture wrt this loss
        input_img = keras.layers.Input(shape=(32,32,3))
        #x = intermediate_layer_model(input_img)
        #x = keras.layers.Dense(1000, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(kernel), bias_initializer=tf.constant_initializer(bias))(x)
        #model1 = keras.models.Model(inputs=input_img,outputs=x)

        #model1 = self.model
        model1 = keras.models.clone_model(self.model)
        model1.set_weights(self.model.get_weights())
        loss = K.mean(model1(input_img)[:, output_index]) - reg * K.mean(K.square(input_img))
        grads = K.gradients(loss, input_img)[0]
        # normalization trick: we normalize the gradient
        #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # we start from a gray image with some noise
        input_img_data = np.random.random((1, 32,32,3)) * 20 + 128.

        # run gradient ascent for 20 steps
        for i in range(self.step):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 1
            if self.verbose and (i % 500 == 0):
                img = input_img_data[0].copy()
                img = self.deprocess_image(img)
                print(loss_value)
                if loss_value > 0:
                    plt.imshow(img.reshape((32,32,3)))
                    plt.show()

        print(loss_value)
        img = input_img_data[0].copy()
        img = self.deprocess_image(img)

        #print(img.shape)
        #plt.imshow(img.reshape((32,32,3)))
        #plt.show()

        #np.savetxt('../results/cmv'+ str(self.current_class) +'.txt', img.reshape(28,28), fmt="%s")
        #imsave('%s_filter_%d.png' % (layer_name, filter_index), img)
        utils_backdoor.dump_image(img,
                                  '../results/cmv'+ str(self.current_class) + ".png",
                                  'png')
        np.savetxt("../results/cmv" + str(self.current_class) + ".txt", input_img_data[0].reshape(32*32*3), fmt="%s")
        return input_img_data[0], img

    def hidden_permutation(self, gen, img, pre_class, target_class):
        # calculate the importance of each hidden neuron given the cmv image
        out = []
        for cur_layer in self.layer:
            #predict = self.model.predict(img.reshape(1,32,32,3))

            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            out_hidden = partial_model1.predict(img.reshape(1,32,32,3))
            ori_pre = partial_model2.predict(out_hidden)

            #ori_class = self.model.predict(img.reshape(1,32,32,3))
            #ori_class = model_copy.predict(img.reshape(1,32,32,3))
            out_hidden_ = np.ndarray.flatten(out_hidden).copy()

            # randomize each hidden
            perm_predict = []
            for i in range(0, len(out_hidden_)):
                perm_predict_neu = []
                out_hidden_ = copy.deepcopy(np.ndarray.flatten(out_hidden))
                for j in range (0, self.random_sample):
                    #hidden_random = np.random.uniform(low=min[i], high=max[i])
                    hidden_do = 0.0
                    out_hidden_[i] = hidden_do
                    sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape))
                    perm_predict_neu.append(sample_pre[0])

                perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                perm_predict_neu = np.abs(ori_pre[0] - perm_predict_neu)
                to_add = []
                to_add.append(int(i))
                to_add.append(perm_predict_neu[pre_class])
                perm_predict.append(np.array(to_add))

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict = np.array(perm_predict)
            out.append(perm_predict)
            ind = np.argsort(perm_predict[:,1])[::-1]
            perm_predict = perm_predict[ind]
            np.savetxt("../results/perm0_pre_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict, fmt="%s")
            #out.append(perm_predict)

        return out

    def hidden_permutation_cmv_all(self, gen, img, pre_class):
        # calculate the importance of each hidden neuron given the cmv image
        out = []
        for cur_layer in self.layer:
            #predict = self.model.predict(img.reshape(1,32,32,3))

            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            out_hidden = partial_model1.predict(img.reshape(1,32,32,3))
            ori_pre = partial_model2.predict(out_hidden)

            #ori_class = self.model.predict(img.reshape(1,32,32,3))
            #ori_class = model_copy.predict(img.reshape(1,32,32,3))
            out_hidden_ = np.ndarray.flatten(out_hidden).copy()

            # randomize each hidden
            perm_predict = []
            for i in range(0, len(out_hidden_)):
                perm_predict_neu = []
                out_hidden_ = copy.deepcopy(np.ndarray.flatten(out_hidden))
                for j in range (0, self.random_sample):
                    #hidden_random = np.random.uniform(low=min[i], high=max[i])
                    hidden_do = 0.0
                    out_hidden_[i] = hidden_do
                    sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape))
                    perm_predict_neu.append(sample_pre[0])

                perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                perm_predict_neu = np.abs(ori_pre[0] - perm_predict_neu)
                to_add = []
                to_add.append(int(i))
                for class_n in self.classes:
                    to_add.append(perm_predict_neu[class_n])
                # neuron index, perm[0], perm[1], ..., perm[9]
                perm_predict.append(np.array(to_add))

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict = np.array(perm_predict)
            out.append(perm_predict)
            #sort
            #ind = np.argsort(perm_predict[:,1])[::-1]
            #perm_predict = perm_predict[ind]
            np.savetxt("../results/perm0_cmv_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict, fmt="%s")
            #out.append(perm_predict)

        return np.array(out)

    def hidden_permutation_test(self, gen, pre_class):
        # calculate the importance of each hidden neuron given the cmv image
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            self.mini_batch = 3
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

                # randomize each hidden
                perm_predict = []
                for i in range(0, len(out_hidden_[0])):
                    perm_predict_neu = []
                    out_hidden_ = out_hidden.reshape(out_hidden.shape[0], -1).copy()
                    for j in range (0, self.random_sample):
                        #hidden_random = np.random.uniform(low=min[i], high=max[i], size=len(out_hidden)).transpose()
                        hidden_do = np.zeros(shape=out_hidden_[:,i].shape)
                        out_hidden_[:, i] = hidden_do
                        sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape)) # 8k x 32
                        perm_predict_neu.append(sample_pre)

                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    perm_predict_neu = np.abs(ori_pre - perm_predict_neu)
                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    to_add = []
                    to_add.append(int(i))
                    to_add.append(perm_predict_neu[pre_class])
                    perm_predict.append(np.array(to_add))
                perm_predict_avg.append(perm_predict)
            # average of batch
            perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            ind = np.argsort(perm_predict_avg[:,1])[::-1]
            perm_predict_avg = perm_predict_avg[ind]
            np.savetxt("../results/test_pre0_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return out

    def hidden_permutation_test_all(self, gen, pre_class):
        # calculate the importance of each hidden neuron given the cmv image
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            self.mini_batch = 3
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

                # randomize each hidden
                perm_predict = []
                for i in range(0, len(out_hidden_[0])):
                    perm_predict_neu = []
                    out_hidden_ = out_hidden.reshape(out_hidden.shape[0], -1).copy()
                    for j in range (0, self.random_sample):
                        #hidden_random = np.random.uniform(low=min[i], high=max[i], size=len(out_hidden)).transpose()
                        hidden_do = np.zeros(shape=out_hidden_[:,i].shape)
                        out_hidden_[:, i] = hidden_do
                        sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape)) # 8k x 32
                        perm_predict_neu.append(sample_pre)

                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    perm_predict_neu = np.abs(ori_pre - perm_predict_neu)
                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    to_add = []
                    to_add.append(int(i))
                    for class_n in self.classes:
                        to_add.append(perm_predict_neu[class_n])
                    perm_predict.append(np.array(to_add))
                perm_predict_avg.append(perm_predict)
            # average of all baches
            perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            #ind = np.argsort(perm_predict_avg[:,1])[::-1]
            #perm_predict_avg = perm_predict_avg[ind]
            np.savetxt("../results/test_pre0_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return np.array(out)

    def hidden_permutation_adv(self, gen, pre_class):
        # calculate the importance of each hidden neuron given the cmv image
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            self.mini_batch = 2
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

                # randomize each hidden
                perm_predict = []
                for i in range(0, len(out_hidden_[0])):
                    perm_predict_neu = []
                    out_hidden_ = out_hidden.reshape(out_hidden.shape[0], -1).copy()
                    for j in range (0, self.random_sample):
                        #hidden_random = np.random.uniform(low=min[i], high=max[i], size=len(out_hidden)).transpose()
                        hidden_do = np.zeros(shape=out_hidden_[:,i].shape)
                        out_hidden_[:, i] = hidden_do
                        sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape)) # 8k x 32
                        perm_predict_neu.append(sample_pre)

                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    perm_predict_neu = np.abs(ori_pre - perm_predict_neu)
                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    to_add = []
                    to_add.append(int(i))
                    to_add.append(perm_predict_neu[pre_class])
                    perm_predict.append(np.array(to_add))
                perm_predict_avg.append(perm_predict)

            perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            ind = np.argsort(perm_predict_avg[:,1])[::-1]
            perm_predict_avg = perm_predict_avg[ind]
            np.savetxt("../results/adv_pre0_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return out

    def hidden_permutation_adv_all(self, gen, pre_class):
        # calculate the importance of each hidden neuron given the cmv image
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            # find the range of hidden neuron output
            '''
            min = []
            max = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                pre = partial_model1.predict(X_batch)
                pre = pre.reshape((len(pre), len(np.ndarray.flatten(pre[0]))))

                _max = np.max(pre, axis=0)
                _min = np.min(pre, axis=0)

                min.append(_min)
                max.append(_max)

            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)
            '''
            self.mini_batch = 2
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

                # randomize each hidden
                perm_predict = []
                for i in range(0, len(out_hidden_[0])):
                    perm_predict_neu = []
                    out_hidden_ = out_hidden.reshape(out_hidden.shape[0], -1).copy()
                    for j in range (0, self.random_sample):
                        #hidden_random = np.random.uniform(low=min[i], high=max[i], size=len(out_hidden)).transpose()
                        hidden_do = np.zeros(shape=out_hidden_[:,i].shape)
                        out_hidden_[:, i] = hidden_do
                        sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape)) # 8k x 32
                        perm_predict_neu.append(sample_pre)

                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    perm_predict_neu = np.abs(ori_pre - perm_predict_neu)
                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    to_add = []
                    to_add.append(int(i))
                    for class_n in self.classes:
                        to_add.append(perm_predict_neu[class_n])
                    perm_predict.append(np.array(to_add))
                perm_predict_avg.append(perm_predict)

            perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            #ind = np.argsort(perm_predict_avg[:,1])[::-1]
            #perm_predict_avg = perm_predict_avg[ind]
            np.savetxt("../results/adv_pre0_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return np.array(out)

    def accuracy_test(self, gen):
        #'''
        correct = 0
        total = 0
        #self.mini_batch = 2
        for idx in range(self.mini_batch):
            X_batch, Y_batch = gen.next()

            #X_batch_perturbed = self.get_perturbed_input(X_batch)

            Y_predict = self.model.predict(X_batch)
            Y_predict = np.argmax(Y_predict, axis=1)
            Y_batch = np.argmax(Y_batch, axis=1)

            correct = correct + np.sum(Y_predict == Y_batch)
            total = total + len(X_batch)

        print("Test accuracy: {}, {}/{}\n".format(correct/total, correct, total))

        pass


    def attack_sr_test(self, x_adv, y_adv):
        #'''
        correct = 0
        total = 0

        Y_predict = self.model.predict(x_adv)
        Y_predict = np.argmax(Y_predict, axis=1)
        y_adv = np.argmax(y_adv, axis=1)

        correct = correct + np.sum(Y_predict == y_adv)
        total = total + len(x_adv)

        # print image
        cur_idx = 0
        for cur_x in x_adv:
            utils_backdoor.dump_image(cur_x * 255,
                                      '../results/test/'+ str(cur_idx) +'.png',
                                      'png')
            cur_idx = cur_idx + 1

        print("Attack success rate: {}, {}/{}".format(correct/total, correct, total))
        print("predictions {}\n".format(Y_predict))

        pass

    def plot_hidden(self, _cmv_rank, _test_rank, normalise=True):
        # plot the permutation of cmv img and test imgs
        cmv_rank = copy.deepcopy(_cmv_rank)
        test_rank = copy.deepcopy(_test_rank)
        plt_row = 2
        #for i in range (0, len(self.layer)):
        #    if len(self.do_neuron[i]) > plt_row:
        #        plt_row = len(self.do_neuron[i])
        plt_col = len(self.layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(7*plt_col, 5*plt_row), sharex=False, sharey=True)
        #fig.tight_layout()

        col = 0
        #self.layer = [2]
        for do_layer in self.layer:
            row = 0
            # plot ACE
            ax[row, col].set_title('Layer_' + str(do_layer))
            #ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            # Baseline is np.mean(expectation_do_x)
            if normalise:
                cmv_rank[col][:,1] = cmv_rank[col][:,1] / np.max(cmv_rank[col][:,1])

            ax[row, col].scatter(cmv_rank[col][:,0].astype(int), cmv_rank[col][:,1], label = str(do_layer) + '_cmv', color='b')
            ax[row, col].legend()

            row = row + 1

            # plot ACE
            #ax[row, col].set_title('Layer_' + str(do_layer))
            ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            # Baseline is np.mean(expectation_do_x)
            if normalise:
                test_rank[col][:,1] = test_rank[col][:,1] / np.max(test_rank[col][:,1])
            ax[row, col].scatter(test_rank[col][:,0].astype(int), test_rank[col][:,1], label = str(do_layer) + '_test', color='b')
            ax[row, col].legend()

            #if row == len(self.do_neuron[col]):
            #    for off in range(row, plt_row):
            #        ax[off, col].set_axis_off()
            #ie_ave.append(ie_ave_l)
            col = col + 1
        if normalise:
            plt.savefig("../results/plt_n_c" + str(self.current_class) + ".png")
        else:
            plt.savefig("../results/plt_c" + str(self.current_class) + ".png")
        plt.show()

    def plot_multiple(self, _rank, name, normalise=False, save_n=""):
        # plot the permutation of cmv img and test imgs
        plt_row = len(_rank)

        rank = []
        for _rank_i in _rank:
            rank.append(copy.deepcopy(_rank_i))

        plt_col = len(self.layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(7*plt_col, 5*plt_row), sharex=False, sharey=True)

        col = 0
        for do_layer in self.layer:
            for row in range(0, plt_row):
                # plot ACE
                if row == 0:
                    ax[row, col].set_title('Layer_' + str(do_layer))
                    #ax[row, col].set_xlabel('neuron index')
                    #ax[row, col].set_ylabel('delta y')

                if row == (plt_row - 1):
                    #ax[row, col].set_title('Layer_' + str(do_layer))
                    ax[row, col].set_xlabel('neuron index')

                ax[row, col].set_ylabel(name[row])

                # Baseline is np.mean(expectation_do_x)
                if normalise:
                    rank[row][col][:,1] = rank[row][col][:,1] / np.max(rank[row][col][:,1])

                ax[row, col].scatter(rank[row][col][:,0].astype(int), rank[row][col][:,1], label = str(do_layer) + '_cmv', color='b')
                ax[row, col].legend()

            col = col + 1
        if normalise:
            plt.savefig("../results/plt_n_c" + str(self.current_class) + save_n + ".png")
        else:
            plt.savefig("../results/plt_c" + str(self.current_class) + save_n + ".png")
        plt.show()


    def plot_diff(self, _cmv_rank, _test_rank, normalise=True):
        # plot the permutation of cmv img and test imgs
        cmv_rank = copy.deepcopy(_cmv_rank)
        test_rank = copy.deepcopy(_test_rank)
        plt_row = 2
        #for i in range (0, len(self.layer)):
        #    if len(self.do_neuron[i]) > plt_row:
        #        plt_row = len(self.do_neuron[i])
        plt_col = len(self.layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(7*plt_col, 5*plt_row), sharex=False, sharey=True)
        #fig.tight_layout()

        col = 0
        #self.layer = [2]
        for do_layer in self.layer:
            row = 0
            # plot ACE
            #ax[row, col].set_title('Layer_' + str(do_layer))
            ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            hidden_diff = np.abs(cmv_rank[col][:,1] - test_rank[col][:,1])
            cmv_rank[col][:,1] = hidden_diff

            ax[row, col].scatter(cmv_rank[col][:,0].astype(int), cmv_rank[col][:,1], label = str(do_layer) + '_diff', color='b')
            ax[row, col].legend()

            row = row + 1

            ax[row, col].set_title('Layer_' + str(do_layer))
            #ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            cmv_rank[col][:,1] = cmv_rank[col][:,1] / np.max(cmv_rank[col][:,1])

            test_rank[col][:,1] = test_rank[col][:,1] / np.max(test_rank[col][:,1])

            hidden_diff = np.abs(cmv_rank[col][:,1] - test_rank[col][:,1])
            cmv_rank[col][:,1] = hidden_diff

            ax[row, col].scatter(cmv_rank[col][:,0].astype(int), cmv_rank[col][:,1], label = str(do_layer) + '_diffn', color='b')
            ax[row, col].legend()

            col = col + 1
        plt.savefig("../results/plt_diff_c" + str(self.current_class) + ".png")
        plt.show()



def load_dataset_class(data_file=('%s/%s' % (DATA_DIR, DATA_FILE)), cur_class=0):
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
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_out = []
    y_out = []
    for i in range (0, len(x_test)):
        if np.argmax(y_test[i], axis=0) == cur_class:
            x_out.append(x_test[i])
            y_out.append(y_test[i])

    return np.array(x_out), np.array(y_out)

def build_data_loader(X, Y):

    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator

