import numpy as np
from keras.layers import Input, Add, Activation, Conv2D
from keras.optimizers import *
import keras.backend as K
from keras.layers import Lambda
from keras.models import Model
import tensorflow as tf

def res_block(X, f, filters, block, s = 1):
    '''
    Implementation of the convolutional block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    '''
    conv_name_base = 'res' + block + '_branch'
    F1, F2 = filters
    X_shortcut = X
    X = Conv2D(F1,(f,f), strides =(s,s),padding='same',name = conv_name_base + 'a', kernel_initializer = 'he_normal')(X)
    X = Activation('relu')(X)

    X = Conv2D(F2,(f,f),strides=(s,s),padding='same',name=conv_name_base+'b',kernel_initializer='he_normal')(X)
    X = Activation('relu')(X)

    X_shortcut = Conv2D(F2,(1,1),strides=(s,s),padding='valid',name=conv_name_base+'1',kernel_initializer='Zeros')(X_shortcut)
    X_shortcut.trainable = False

    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    return X


def SubpixelConv2D(input_shape, scale):
    '''
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space

    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158

    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    '''
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)


    return Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')


def my_loss(y_true, y_pred):
    '''
    The loss funcition is written according to the description in the original 
    paper. It can improve the edge of the image.
    '''
    loss_mse = K.mean(K.square(y_pred - y_true), axis=-1)
    h = np.array([[ -1, 0, 1],                        #filters
                  [ -2, 0, 2],
                  [ -1, 0, 1]])
    h.resize((3,3,1))
    h1 = np.random.rand(3, 3, 3, 1)
    h1[0] = h
    h1[1] = h
    h1[2] = h
    ht = np.array([[ -1, -2, -1],                     #filters
                   [ 0, 0, 0],
                   [ 1, 2, 1]])
    ht.resize((3,3,1))
    h2 = np.random.rand(3, 3, 3, 1)
    h2[0] = ht
    h2[1] = ht
    h2[2] = ht
    filter1 = tf.Variable(h1, dtype=np.float32,trainable = False)
    filter2 = tf.Variable(h2, dtype=np.float32,trainable = False)
    f1 = tf.nn.conv2d(y_pred, filter1, strides=[1, 1, 1, 1], padding='SAME')
    f2 = tf.nn.conv2d(y_pred, filter2, strides=[1, 1, 1, 1], padding='SAME')
    gradient_loss = f1**2 + f2**2
    lam = 0.001
    loss = loss_mse + lam*K.mean(gradient_loss, axis=-1)
    return loss

def resnet(input_shape = (64, 64, 1)):
    X_input = Input(input_shape)
    X = Conv2D(32, (3, 3),  activation='relu', strides = (1, 1), padding='same', name = 'conv1', kernel_initializer = 'he_normal')(X_input)
    X = res_block(X, 3, [32, 34], block='1', s = 1)
    X = res_block(X, 3, [34, 38], block='2', s = 1)
    X = res_block(X, 3, [38, 44], block='3', s = 1)
    X = res_block(X, 3, [44, 52], block='4', s = 1)
    X = res_block(X, 3, [52, 62], block='5', s = 1)
    X = Conv2D(16, (3, 3), strides = (1, 1), padding='same', name = 'conv2', kernel_initializer = 'he_normal')(X)
    X = SubpixelConv2D(X.shape, scale=4)(X)
    X = Conv2D(1, (3, 3), strides = (1, 1), padding='same', name = 'conv3', kernel_initializer = 'he_normal')(X)
    model = Model(inputs = X_input, outputs = X, name='resNet')
    print(model.summary())
    print('Compile model...')
    model.compile(optimizer=Adam(lr=1e-4,epsilon=10e-8), loss=my_loss)
    #model.compile(optimizer=Adam(lr=1e-4,epsilon=10e-8), loss = 'mse', metrics = ['mse'])
    print('Compile complete...')
    return model 
