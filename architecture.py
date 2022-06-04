# Author: Jacob Dawson
#
# This file is just for creating the two networks (the generator and the
# discriminator). It will then be imported by junoGAN.py to construct the
# GAN model overall.
# This is important because it means that here, we can really play around
# with architectural decisions.
from tensorflow import keras
import tensorflow as tf
from constants import *

# helper for gen()
def gen_encoder_block(n,batchnorm=True):
    t = keras.Sequential()
    # downsample:
    t.add(keras.layers.Conv2D(n, (4,4), strides=(2,2), padding='same'))
    if batchnorm:
        # optionally batchnormalize
        t.add(keras.layers.BatchNormalization())
    t.add(keras.layers.LeakyReLU(alpha=0.2))
    return t

# helper for gen()
def gen_decoder_block(n,dropout=True):
    t=keras.Sequential()
    # upsample:
    t.add(keras.layers.Conv2DTranspose(n, (4,4), strides=(2,2), padding='same'))
    # always do batch normalization
    t.add(keras.layers.BatchNormalization())
    if dropout:
        # optional dropout layer
        t.add(keras.layers.Dropout(0.5))
    # activation:
    t.add(keras.layers.LeakyReLU(alpha=0.2))
    return t  

def gen():
    # uses Sequential and the above helpers to create a generator model.
    # we're using some sort of encoder-decoder model, to compress the
    # image to a latent space and then return it to full size.
    # hopefully this will retain its shape, but do color-correction!
    g = keras.Sequential(
        [
            # input:
            keras.layers.InputLayer((image_size,image_size,num_channels),dtype=tf.float16),
            # encode to latent space:
            gen_encoder_block(num_filters//4, batchnorm=False),
            gen_encoder_block(num_filters//2),
            gen_encoder_block(num_filters),
            # bottleneck:
            keras.layers.Conv2D(num_filters,(4,4), strides=(1,1), padding='same'),
            keras.layers.LeakyReLU(alpha=0.2),
            # decode:
            gen_decoder_block(num_filters),
            gen_decoder_block(num_filters//2),
            gen_decoder_block(num_filters//4, dropout=False),
            # output, make sure its outputting an RGB:
            keras.layers.Conv2D(3, (4,4), strides=(1,1), padding='same', activation='tanh')
        ],
        name='generator'
    )
    return g

def dis():
    # fairly simple convolutional network. I think I'll use the
    # sequential framework for this one.
    d = keras.Sequential(
        [
            keras.layers.InputLayer((image_size,image_size,num_channels),dtype=tf.float16),
            keras.layers.Conv2D(num_filters, (4,4), strides=(2,2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(num_filters, (4,4), strides=(2,2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(num_filters, (4,4), strides=(2,2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(num_filters, (4,4), strides=(2,2), padding='same',activation='sigmoid'),
            keras.layers.GlobalMaxPooling2D(),
            keras.layers.Dense(1)
        ],
        name='discriminator'
    )
    return d

if __name__=='__main__':
    # display the two models.
    gen().summary()
    dis().summary()