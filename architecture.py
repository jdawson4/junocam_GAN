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
initializer = keras.initializers.GlorotNormal(seed=seed)

'''# helper for gen()
def gen_encoder_block(n,batchnorm=True):
    t = keras.Sequential()
    # downsample:
    t.add(keras.layers.Conv2D(n, (3,3), strides=(2,2), padding='same'))
    if batchnorm:
        # optionally batchnormalize
        t.add(keras.layers.BatchNormalization(momentum=0.85))
    t.add(keras.layers.LeakyReLU(alpha=0.2))
    return t

# helper for gen()
def gen_decoder_block(n,dropout=True):
    t=keras.Sequential()
    # upsample:
    t.add(keras.layers.Conv2DTranspose(n, (3,3), strides=(2,2), padding='same'))
    # always do batch normalization
    t.add(keras.layers.BatchNormalization(momentum=0.85))
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
            gen_encoder_block(num_filters, batchnorm=False),
            gen_encoder_block(num_filters),
            #gen_encoder_block(num_filters),
            # bottleneck:
            keras.layers.Conv2D(num_filters*2,(3,3), strides=(1,1), padding='same'),
            keras.layers.BatchNormalization(momentum=0.85),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(num_filters*2,(3,3), strides=(1,1), padding='same'),
            keras.layers.BatchNormalization(momentum=0.85),
            keras.layers.LeakyReLU(alpha=0.2),
            # decode:
            #gen_decoder_block(num_filters),
            gen_decoder_block(num_filters),
            gen_decoder_block(num_filters, dropout=False),
            # output, make sure its outputting an RGB:
            keras.layers.Conv2D(3, (3,3), strides=(1,1), padding='same', activation='sigmoid'),
        ],
        name='generator'
    )
    return g'''

'''def dis():
    # fairly simple convolutional network. I think I'll use the
    # sequential framework for this one.
    d = keras.Sequential(
        [
            keras.layers.InputLayer((image_size,image_size,num_channels),dtype=tf.float16),
            keras.layers.Conv2D(num_filters, (3,3), strides=(2,2), padding='same'),
            keras.layers.BatchNormalization(momentum=0.85),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(num_filters, (3,3), strides=(2,2), padding='same'),
            keras.layers.BatchNormalization(momentum=0.85),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(num_filters, (3,3), strides=(2,2), padding='same'),
            keras.layers.BatchNormalization(momentum=0.85),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(num_filters, (3,3), strides=(2,2), padding='same',activation='sigmoid'),
            keras.layers.GlobalMaxPooling2D(),
            keras.layers.Dense(1)
        ],
        name='discriminator'
    )
    return d'''

# probably a resnet would work best for this task, right?
def resnetBlock(filters,input):
    # this will simply do a little convoluting, retaining shape.
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same',kernel_initializer=initializer)(input)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.LeakyReLU(alpha=0.2)(output)
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same',kernel_initializer=initializer)(output)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.LeakyReLU(alpha=0.2)(output)
    return keras.layers.Add()([input,output])

# ok that's not working too well. Let's try this:
def simpleConvBlock(filters,input):
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same',kernel_initializer=initializer)(input)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.LeakyReLU(alpha=0.2)(output)
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same',kernel_initializer=initializer)(output)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.LeakyReLU(alpha=0.2)(output)
    return output

# helper for gen()
def gen_encoder_block(n,batchnorm=True):
    t = keras.Sequential()
    # downsample:
    t.add(keras.layers.Conv2D(n, (3,3), strides=(2,2), padding='same',kernel_initializer=initializer))
    if batchnorm:
        # optionally batchnormalize
        t.add(keras.layers.BatchNormalization(momentum=0.85))
    t.add(keras.layers.LeakyReLU(alpha=0.2))
    return t

# helper for gen()
def gen_decoder_block(n):
    t=keras.Sequential()
    # upsample:
    t.add(keras.layers.Conv2DTranspose(n, (3,3), strides=(2,2), padding='same',kernel_initializer=initializer))
    # always do batch normalization
    t.add(keras.layers.BatchNormalization(momentum=0.85))
    # activation:
    t.add(keras.layers.LeakyReLU(alpha=0.2))
    return t

# decided to make this whole thing residual.
def gen():
    input = keras.layers.Input(shape=(image_size,image_size,num_channels),dtype=tf.float16)
    '''output = gen_encoder_block(num_filters//2)(input)
    output = gen_encoder_block(num_filters)(output)
    output = resnetBlock(num_filters,output)
    output = resnetBlock(num_filters,output)
    output = resnetBlock(num_filters,output)
    output = resnetBlock(num_filters,output)
    output = resnetBlock(num_filters,output)
    output = resnetBlock(num_filters,output)
    output = resnetBlock(num_filters,output)
    output = gen_decoder_block(num_filters//2)(output)
    output = gen_decoder_block(num_channels)(output)
    output = keras.layers.Add()([input,output])
    output = keras.layers.Conv2D(num_channels,(3,3),(1,1),padding='same',activation='sigmoid',kernel_initializer=initializer)(output)
    return keras.Model(inputs=input, outputs=output)'''
    en1 = gen_encoder_block(num_filters//2)(input)
    en2 = gen_encoder_block(num_filters)(en1)
    en3 = gen_encoder_block(num_filters*2)(en2)
    res1 = resnetBlock(num_filters*2,en3)
    res2 = resnetBlock(num_filters*2,res1)
    de1 = keras.layers.Concatenate()([en2,gen_decoder_block(num_filters*2)(res2)])
    de2 = keras.layers.Concatenate()([en1,gen_decoder_block(num_filters)(de1)])
    de3 = keras.layers.Concatenate()([input,gen_decoder_block(num_filters//2)(de2)])
    output = keras.layers.Conv2D(num_channels,(3,3),(1,1),padding='same',activation='tanh',kernel_initializer=initializer)(de3)
    return keras.Model(inputs=input, outputs=output)

def dis_block(filters,input):
    output = keras.layers.Conv2D(filters,(3,3),(2,2),padding='same',kernel_initializer=initializer)(input)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.LeakyReLU(alpha=0.2)(output)
    return output

def dis():
    input = keras.layers.Input(shape=(image_size,image_size,num_channels),dtype=tf.float16)
    out = dis_block(num_filters//4,input)
    out = dis_block(num_filters,out)
    out = dis_block(num_filters*4,out)
    out = keras.layers.GlobalMaxPooling2D()(out)
    out = keras.layers.Dense(1,activation='sigmoid',kernel_initializer=initializer)(out)
    return keras.Model(inputs=input,outputs=out)

if __name__=='__main__':
    # display the two models.
    gen().summary()
    dis().summary()