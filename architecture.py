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
initializer = keras.initializers.RandomNormal(seed=seed)
#initializer = keras.initializers.Zeros()
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

def downsample(input, filters, size, apply_batchnorm=True):
    out = keras.layers.Conv2D(filters, kernel_size=size, strides=2, padding='same', kernel_initializer=initializer)(input)
    if apply_batchnorm:
        out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('selu')(out)
    return out

def upsample(input, filters, size, apply_dropout=False):
    #out = keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(input) # removes "grid" artifacting, but produces NaNs on train :(
    #out = keras.layers.Conv2DTranspose(filters, kernel_size=size, strides=1, padding='same', kernel_initializer=initializer)(out)
    out = keras.layers.Conv2DTranspose(filters, kernel_size=size, strides=2, padding='same', kernel_initializer=initializer)(input)
    out = keras.layers.BatchNormalization()(out)
    if apply_dropout:
        out = keras.layers.Dropout(0.5)(out)
    out = keras.layers.Activation('selu')(out)
    return out

def bottleneck(input, filters, size, apply_dropout=False, apply_batchnorm=True):
    out = keras.layers.Conv2D(filters, kernel_size=size, strides=1, padding='same', kernel_initializer=initializer)(input)
    if apply_batchnorm:
        out = keras.layers.BatchNormalization()(out)
    if apply_dropout:
        out = keras.layers.Dropout(0.5)(out)
    out = keras.layers.Activation('selu')(out)
    return out

def gen():
    input = keras.layers.Input(shape=(None,None,num_channels), dtype=tf.float16)
    scale = keras.layers.Rescaling(1.0/255.0, offset=0)(input)
    d1 = downsample(scale, 4, 2, apply_batchnorm=False)#200
    d2 = downsample(d1, 8, 2)#100
    d3 = downsample(d2, 16, 2)#50
    d4 = downsample(d3, 32, 2)#25
    d5 = bottleneck(d4, 32, 3)#25
    d6 = bottleneck(d5, 32, 3)#25
    d7 = bottleneck(d6, 32, 3)#25
    d8 = bottleneck(d7, 32, 5)#25
    u1 = bottleneck(d8, 32, 5, apply_dropout=True)#25
    u1 = keras.layers.Concatenate()([u1,d7])
    u2 = bottleneck(u1, 32, 3, apply_dropout=True)#25
    u2 = keras.layers.Concatenate()([u2,d6])
    u3 = bottleneck(u2, 32, 3, apply_dropout=True)#25
    u3 = keras.layers.Concatenate()([u3,d5])
    u4 = bottleneck(u3, 32, 3)#25
    u4 = keras.layers.Concatenate()([u4,d4])
    u5 = upsample(u4, 16, 2)#50
    u5 = keras.layers.Concatenate()([u5,d3])
    u6 = upsample(u5, 8, 2)#100
    u6 = keras.layers.Concatenate()([u6,d2])
    u7 = upsample(u6, 4, 2)#200
    u7 = keras.layers.Concatenate()([u7,d1])
    out = keras.layers.Conv2DTranspose(4,kernel_size=2,strides=2,padding='same',kernel_initializer=initializer,activation='tanh')(u7)#400
    out = keras.layers.Conv2D(num_channels,kernel_size=1,strides=1,padding='same',kernel_initializer=initializer,activation='tanh')(out)
    out =  keras.layers.Add()([out, scale])
    out = keras.layers.Rescaling(255.0)(out)
    out = keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 255.0))(out)
    return keras.Model(inputs=input, outputs=out, name='generator')

def disc_block(input, filters, size, apply_batchnorm=True):
    out = keras.layers.Conv2D(filters, kernel_size=size, strides=2, padding='same', kernel_initializer=initializer, kernel_constraint=const)(input)
    if apply_batchnorm:
        out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('selu')(out)
    return out

def dis():
    input = keras.layers.Input(shape=(None,None,num_channels), dtype=tf.float16)
    scale = keras.layers.Rescaling(1.0/127.5,offset=-1)(input)
    out = disc_block(scale, 8, 4, apply_batchnorm=False)
    out = disc_block(out, 16, 4)
    out = disc_block(out, 32, 4)
    out = keras.layers.Conv2D(
        64,
        kernel_size=4,
        strides=1,
        kernel_initializer=initializer,
        kernel_constraint=const
    )(out)
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation('selu')(out)
    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Dense(1, kernel_constraint=const)(out)
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
