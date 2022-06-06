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

# probably a resnet would work best for this task, right?
def resnetBlock(filters,input):
    # this will simply do a little convoluting, retaining shape.
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same')(input)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.LeakyReLU(alpha=0.2)(output)
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same')(output)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.Activation('relu')(output)
    return keras.layers.Add()([input,output])

# ok that's not working too well. Let's try this:
def simpleConvBlock(filters,input):
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same')(input)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.LeakyReLU(alpha=0.2)(output)
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same')(output)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.Activation('relu')(output)
    return output

# helper for gen()
def gen_encoder_block(n,batchnorm=True):
    t = keras.Sequential()
    # downsample:
    t.add(keras.layers.Conv2D(n, (3,3), strides=(2,2), padding='same'))
    if batchnorm:
        # optionally batchnormalize
        t.add(keras.layers.BatchNormalization(momentum=0.85))
    t.add(keras.layers.Activation('relu'))
    return t

# helper for gen()
def gen_decoder_block(n):
    t=keras.Sequential()
    # upsample:
    t.add(keras.layers.Conv2DTranspose(n, (3,3), strides=(2,2), padding='same'))
    # always do batch normalization
    t.add(keras.layers.BatchNormalization(momentum=0.85))
    # activation:
    t.add(keras.layers.Activation('relu'))
    return t

# decided to make this whole thing residual.
def gen():
    input = keras.layers.Input(shape=(image_size,image_size,num_channels),dtype=tf.float16)
    scale = keras.layers.Rescaling(1.0/255.0,offset=0)(input) # scale to between 0 and 1
    en1 = gen_encoder_block(num_channels*3, batchnorm=False)(scale)
    en2 = gen_encoder_block(num_channels*4)(en1)
    en3 = gen_encoder_block(num_channels*5)(en2)
    res1 = simpleConvBlock(num_channels*6,en3)
    res2 = simpleConvBlock(num_channels*6,res1)
    drop = keras.layers.Dropout(0.25)(res2)
    de1 = keras.layers.Dropout(0.3)(keras.layers.Concatenate()([en2,gen_decoder_block(num_channels*6)(drop)]))
    de2 = keras.layers.Dropout(0.3)(keras.layers.Concatenate()([en1,keras.layers.UpSampling2D()(de1)]))
    de3 = keras.layers.Dropout(0.3)(keras.layers.Concatenate()([scale,keras.layers.UpSampling2D()(de2)]))
    output1 = keras.layers.Conv2D(num_channels,(1,1),(1,1),padding='same',activation='relu')(de3)
    output2 = keras.layers.Conv2D(num_channels,(3,3),(1,1),padding='same',activation='relu')(de3)
    output3 = keras.layers.Conv2D(num_channels,(5,5),(1,1),padding='same',activation='relu')(de3)
    output = keras.layers.Add()([output1,output2,output3])
    #output = keras.layers.Lambda(lambda x: tf.keras.activations.relu(x,threshold=0, max_value=1))(output) # outputs should be between 0 and 1
    output = keras.layers.Rescaling(255)(output) # rescale up to 255
    return keras.Model(inputs=input, outputs=output,name='generator')

def dis_block(filters,input,batchnorm=True):
    output = keras.layers.Conv2D(filters,(3,3),(2,2),padding='same', kernel_constraint=const)(input)
    if batchnorm:
        output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.LeakyReLU(alpha=0.2)(output)
    return output

def dis():
    input = keras.layers.Input(shape=(image_size,image_size,num_channels),dtype=tf.float16)
    scale = keras.layers.Rescaling(1.0/127.5,offset=-1)(input)
    out = dis_block(num_channels*2,scale,batchnorm=False)
    out = dis_block(num_channels*3,out)
    out = dis_block(num_channels*4,out)
    out = dis_block(num_channels*5,out)
    out = dis_block(num_channels*6,out)
    out = dis_block(num_channels*7,out)
    out = dis_block(num_channels*8,out)
    out = keras.layers.GlobalMaxPooling2D()(out)
    out = keras.layers.Dense(1, kernel_constraint=const)(out)
    return keras.Model(inputs=input,outputs=out,name='discriminator')

if __name__=='__main__':
    # display the two models.
    gen().summary()
    dis().summary()