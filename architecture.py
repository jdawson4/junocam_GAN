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
'''initializer = keras.initializers.RandomNormal(seed=seed)'''
class ClipConstraint(keras.constraints.Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value

	# clip model weights to hypercube
	def __call__(self, weights):
		return keras.backend.clip(weights, -self.clip_value, self.clip_value)

	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
const = ClipConstraint(0.01)

# ok, the generator clearly isn't doing too hot.
# Here we go with a new architecture: a densenet generator!

# first component: the Batchnormalize, Activation,
# Convolution combo-punch
def bac(x,filters):
    b = keras.layers.BatchNormalization(momentum=0.85)(x)
    b = keras.layers.Activation('selu')(b)
    b = keras.layers.Conv2D(filters, kernel_size=(3,3), padding='same')(b)
    return b

# the DenseBlock then uses that above combination to create a
# very dense thingie that outputs a good bit of data.
def denseBlock(x,filters):
    l1 = bac(x,filters)
    l2 = bac(l1,filters)
    l3 = bac(keras.layers.Concatenate()([l1,l2]),filters)
    l4 = bac(keras.layers.Concatenate()([l1,l2,l3]),filters)
    l5 = bac(keras.layers.Concatenate()([l1,l2,l3,l4]),filters)
    # do we pool? I don't think so--that's only in the transition layers
    return l5

def transitionDownscale(x,filters):
    conv = keras.layers.Conv2D(filters,kernel_size=(1,1),strides=(1,1),activation='selu',padding='same')(x)
    avp = keras.layers.AveragePooling2D((2,2),strides=2)(conv)
    return avp

def transitionUpscale(x,filters):
    conv = keras.layers.Conv2D(filters,kernel_size=(1,1),strides=(1,1),activation='selu',padding='same')(x)
    ups = keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(conv)
    return ups

def gen():
    input = keras.layers.Input(shape=(image_size,image_size,num_channels), dtype=tf.float32)
    scale = keras.layers.Rescaling(1.0/255.0, offset=0)(input)
    #c1 = keras.layers.Conv2D(4, kernel_size=(7,7), strides=(1,1), activation='selu', padding='same')(scale)
    c1 = keras.layers.Conv2D(4, kernel_size=(7,7), strides=(1,1), activation='selu', padding='same')(scale)
    d1 = denseBlock(c1,8)
    t1 = transitionDownscale(d1,8)
    d2 = denseBlock(t1,16)
    t2 = transitionDownscale(d2,16)
    d3 = denseBlock(t2,32)
    t3 = transitionDownscale(d3,32)
    d4 = denseBlock(t3,64)
    t4 = transitionUpscale(d4,32)
    t4 = keras.layers.Concatenate()([t4,d3])
    d5 = denseBlock(t4,16)
    t5 = transitionUpscale(d5,16)
    t5 = keras.layers.Concatenate()([t5, d2])
    d6 = denseBlock(t5,8)
    t6 = transitionUpscale(d6,8)
    t6 = keras.layers.Concatenate()([t6,d1])
    c2 = keras.layers.Conv2D(num_channels, kernel_size=(3,3), strides=(1,1), activation='selu',padding='same')(t6) # essentially the network output--the amount to "add" to the original image
    c3 = keras.layers.Conv2D(num_channels, kernel_size=(5,5), strides=(1,1), activation='selu',padding='same')(c2)
    out = keras.layers.Conv2D(num_channels, kernel_size=(1,1), strides=(1,1), activation='selu',padding='same')(c3)
    out = keras.layers.Add()([out, scale])
    out = keras.layers.Rescaling(255.0)(out)
    out = keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 255.0))(out)
    return keras.Model(inputs=input, outputs=out, name='generator')

def hasNan(x, number):
    # I'm sick of getting this error. What if our network warns us when there's
    # a Nan somewhere in the system?
    def p(num): print(num,'has a nan!')
    def q(num): pass
    tf.cond(pred=tf.math.reduce_any((tf.math.is_nan(x))), true_fn=lambda:p(number), false_fn=lambda:q(number))
    #f(number)
    return x

def dis_block(filters,input,batchnorm=True):
    output = keras.layers.Conv2D(filters,(2,2),(2,2),padding='valid', kernel_constraint=const)(input)
    if batchnorm:
        output = keras.layers.BatchNormalization(momentum=0.85)(output)
    #output = keras.layers.LeakyReLU(alpha=0.2)(output)
    output = keras.layers.Activation('selu')(output)
    return output

# this describes internal convolutions for the discriminator
def discConvBlock(filters,input):
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same', kernel_constraint=const)(input) # 3x3 because these are in a small latent space.
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.Activation('selu')(output)
    #output = (keras.layers.LeakyReLU())(output)
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same', kernel_constraint=const)(output)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.Activation('selu')(output)
    #output = (keras.layers.LeakyReLU())(output)
    return output

def dis():
    input = keras.layers.Input(shape=(image_size,image_size,num_channels), dtype=tf.float32)
    scale = keras.layers.Rescaling(1.0/127.5,offset=-1)(input)
    #out = keras.layers.Lambda(lambda x:hasNan(x,0))(scale)
    out = keras.layers.RandomRotation((-0.3,0.3),seed=seed)(scale)
    out = keras.layers.RandomZoom(0.5,0.5,seed=seed)(out)
    #out = keras.layers.RandomContrast(0.2,seed=seed)(out)
    #out = keras.layers.RandomBrightness(0.5, value_range=(-1,1), seed=seed)(out) # doesn't exist?
    #out = keras.layers.RandomCrop(image_size//2, image_size//2, seed=seed)(out) # resizes, not sure if we want that?
    out = keras.layers.RandomFlip(seed=seed)(out)
    out = keras.layers.RandomTranslation(0.3,0.3,seed=seed)(out)
    out = keras.layers.Dropout(0.2,seed=seed)(out)
    out = dis_block(8,out,batchnorm=False)
    out = keras.layers.Dropout(0.2,seed=seed)(out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,1))(out)
    out = dis_block(8,out)
    out = keras.layers.Dropout(0.2)(out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,2))(out)
    out = dis_block(16,out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,3))(out)
    out = dis_block(16,out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,4))(out)
    out = dis_block(24,out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,5))(out)
    #out = dis_block(num_channels*10,out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,6))(out)
    #out = dis_block(num_channels*11,out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,7))(out)
    out = discConvBlock(24, out)
    out = discConvBlock(32, out)
    out = discConvBlock(32, out)
    out = discConvBlock(32, out)
    out = keras.layers.Flatten()(out)
    #out = keras.layers.Dropout(0.2)(out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,8))(out)
    #out = keras.layers.Lambda(lambda x: tf.math.multiply_no_nan(x, tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(x)), dtype=tf.float32)))(out)
    out = keras.layers.Dense(1, kernel_constraint=const)(out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,9))(out)
    #out = keras.layers.PReLU()(out)
    return keras.Model(inputs=input,outputs=out,name='discriminator')

if __name__=='__main__':
    # display the two models.
    g = gen()
    d = dis()
    g.summary()
    d.summary()
    # These next two lines require graphviz, the bane of my existence
    #keras.utils.plot_model(g, to_file='generator_plot.png', show_shapes=True, show_layer_names=False, show_layer_activations=True, expand_nested=True)
    #keras.utils.plot_model(d, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=False, show_layer_activations=True, expand_nested=True)
