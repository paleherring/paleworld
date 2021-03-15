import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten

def get_generator(shape, gf_dim=64): # Dimension of gen filters in the first conv layer. [64]
    image_size = 64
    s16 = image_size // 16
    # w_init = tf.glorot_normal_initializer()
    w_init = tf.random_normal_initializer(stddev = 0.02)
    gamma_init = tf.random_normal_initializer(1.,0.02)

    ni = Input(shape) # the input class is a starting layer of a neural network
    nn = Dense(n_units=(gf_dim * 8 * s16 * s16), W_init = w_init, b_init = None)(ni)
    # the dense class is a fully connected layer
    nn = Reshape(shape = [-1, s16, s16, gf_dim * 8])(nn)# a layer that reshapes a given tensor
    nn = BatchNorm2d(decay = 0.9, act = tf.nn.relu, gamma_init = gamma_init, name=None)(nn)
    #BatchNorm2d applies Batch Normalization to 4D inputs (N,H,W,C)
    nn = DeConv2d(gf_dim * 4, (5,5),(2,2), W_init = w_init, b_init = None)(nn)
    nn = BatchNorm2d(decay = 0.9, act = tf.nn.relu, gamma_init = gamma_init)(nn)
    nn = DeConv2d(gf_dim * 2,(5,5),(2,2),W_init = w_init,b_init = None)(nn)
    nn = BatchNorm2d(decay = 0.9, act = tf.nn.relu, gamma_init = gamma_init)(nn)
    nn = DeConv2d(gf_dim,(5,5),(2,2),W_init = w_init, b_init = None)(nn)
    nn = BatchNorm2d(decay=0.9, act = tf.nn.relu, gamma_init = gamma_init)(nn)
    nn = DeConv2d(3,(5,5),(2,2), act = tf.nn.tanh, W_init = w_init)(nn)

    return tl.models.Model(inputs = ni, outputs = nn, name = 'generator')

def get_discriminator(shape, df_dim=64):# Dimension of discriminator filters in the first conv layer. [64]
    # w_init = tf.glorot_normal_initializer()
    w_init = tf.random_normal_initializer(stddev = 0.02)
    gamma_init = tf.random_normal_initializer(1.,0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x,0.2)

    ni = Input(shape)
    nn = Conv2d(df_dim, (5,5),(2,2), act = lrelu, W_init = w_init)(ni)
    nn = Conv2d(df_dim * 2, (5,5),(2,2), W_init = w_init,b_init = None)(nn)
    nn = BatchNorm2d(decay=0.9, act = lrelu, gamma_init = gamma_init)(nn)
    nn = Conv2d(df_dim * 4, (5,5),(2,2), W_init = w_init, b_init = None)(nn)
    nn = BatchNorm2d(decay = 0.9, act = lrelu, gamma_init = gamma_init)(nn)
    nn = Conv2d(df_dim * 8, (5,5),(2,2),W_init = w_init, b_init = None)(nn)
    nn = BatchNorm2d(decay = 0.9, act = lrelu, gamma_init = gamma_init)(nn)
    nn = Flatten()(nn)
    nn = Dense(n_units = 1, act = tf.identity, W_init = w_init)(nn)

    return tl.models.Model(inputs = ni, outputs=nn, name = 'discriminator')


