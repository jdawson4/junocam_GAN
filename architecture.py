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
    #output = keras.layers.Activation('relu')(output)
    output = (keras.layers.LeakyReLU())(output)
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same')(output)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    #output = keras.layers.Activation('relu')(output)
    output = (keras.layers.LeakyReLU())(output)
    return keras.layers.Add()([input,output])

# ok that's not working too well. Let's try this:
def simpleConvBlock(filters,input):
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same')(input) # 3x3 because these are in a small latent space.
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    #output = keras.layers.Activation('relu')(output)
    output = (keras.layers.LeakyReLU())(output)
    output = keras.layers.Conv2D(filters,(3,3),(1,1),padding='same')(output)
    output = keras.layers.BatchNormalization(momentum=0.85)(output)
    #output = keras.layers.Activation('relu')(output)
    output = (keras.layers.LeakyReLU())(output)
    return output

# helper for gen()
def gen_encoder_block(n,batchnorm=True):
    t = keras.Sequential()
    # downsample:
    t.add(keras.layers.Conv2D(n, (4,4), strides=(2,2), padding='same'))
    if batchnorm:
        # optionally batchnormalize
        t.add(keras.layers.BatchNormalization(momentum=0.85))
    #t.add(keras.layers.Activation('relu'))
    t.add(keras.layers.LeakyReLU())
    return t

# helper for gen()
def gen_decoder_block(n):
    t=keras.Sequential()
    # upsample:
    t.add(keras.layers.Conv2DTranspose(n, (4,4), strides=(2,2), padding='same'))
    # always do batch normalization
    t.add(keras.layers.BatchNormalization(momentum=0.85))
    # activation:
    #t.add(keras.layers.Activation('relu'))
    t.add(keras.layers.LeakyReLU())
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
    de2 = (keras.layers.Concatenate()([en1,keras.layers.UpSampling2D()(de1)]))
    de3 = keras.layers.LeakyReLU()(keras.layers.Dropout(0.3)(keras.layers.UpSampling2D()(de2)))
    output1 = keras.layers.Conv2D(num_channels,(1,1),(1,1),padding='same',activation=None)(de3)
    output2 = keras.layers.Conv2D(num_channels,(3,3),(1,1),padding='same',activation=None)(de3)
    output3 = keras.layers.Conv2D(num_channels,(5,5),(1,1),padding='same',activation=None)(de3)
    output = keras.layers.LeakyReLU()(keras.layers.Concatenate()([output1,output2,output3]))
    #output = keras.layers.Dropout(0.3)(output)
    output = keras.layers.Add()([keras.layers.Conv2D(num_channels,(1,1),(1,1),padding='same',activation=None)(output),scale]) # finally handle the RGB output
    #output = keras.layers.Concatenate()([keras.layers.Dense(1)(output), keras.layers.Dense(1)(output), keras.layers.Dense(1)(output)])
    output = keras.layers.ReLU(max_value=1.0)(output) # I don't KNOW if I have to do this, but the output should never be greater than 1???
    output = keras.layers.Rescaling(255.0)(output) # rescale up to 255
    return keras.Model(inputs=input, outputs=output,name='generator')

def hasNan(x, number):
    # I'm sick of getting this error. What if our network warns us when there's
    # a Nan somewhere in the system?
    def p(num): print(num,'has a nan!')
    def q(num): pass
    tf.cond(pred=tf.math.reduce_any((tf.math.is_nan(x))), true_fn=lambda:p(number), false_fn=lambda:q(number))
    #f(number)
    return x

def dis_block(filters,input,batchnorm=True):
    output = keras.layers.Conv2D(filters,(4,4),(2,2),padding='valid', kernel_constraint=const)(input)
    if batchnorm:
        output = keras.layers.BatchNormalization(momentum=0.85)(output)
    output = keras.layers.LeakyReLU(alpha=0.2)(output)
    #output = keras.layers.Activation('relu')(output)
    return output

def dis():
    input = keras.layers.Input(shape=(image_size,image_size,num_channels),dtype=tf.float16)
    scale = keras.layers.Rescaling(1.0/127.5,offset=-1)(input)
    #out = keras.layers.Lambda(lambda x:hasNan(x,0))(scale)
    out = dis_block(num_channels*2,scale,batchnorm=False)
    #out = keras.layers.Lambda(lambda x:hasNan(x,1))(out)
    out = dis_block(num_channels*3,out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,2))(out)
    out = dis_block(num_channels*4,out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,3))(out)
    out = dis_block(num_channels*5,out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,4))(out)
    out = dis_block(num_channels*6,out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,5))(out)
    out = dis_block(num_channels*7,out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,6))(out)
    out = dis_block(num_channels*8,out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,7))(out)
    out = keras.layers.Flatten()(out)
    out = keras.layers.Dropout(0.2)(out)
    #out = keras.layers.Lambda(lambda x:hasNan(x,8))(out)
    #out = keras.layers.Lambda(lambda x: tf.math.multiply_no_nan(x, tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(x)), dtype=tf.float32)))(out)
    out = keras.layers.Dense(1, kernel_constraint=None)(out)
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