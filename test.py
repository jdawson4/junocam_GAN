# Author: Jacob Dawson
# This file just contains a few numpy/tensorflow functions
# I need to understand in order to get the rest of the system
# to work. You can safely ignore all of this.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from constants import *
import imageio

'''def hasNan(x, number):
    if tf.math.reduce_any((tf.math.is_nan(x))):
        print(number, "has a nan!")
    return x

a=tf.cast([1,2,np.nan,6.9,5.20,np.nan], tf.float16)
print(tf.math.multiply_no_nan(a, tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(a)), dtype=tf.float16)))
print(hasNan(a, 1))'''

'''a=tf.cast([-0.2,-20.0,256.0,125.0,300.0], tf.float16)
def makeImg(x):
    min = tf.reduce_min(x)
    if (min < 0):
        x += tf.math.abs(min)
    max = tf.reduce_max(x)
    x = (x/max) * 255.0
    return x
print(makeImg(a))
    
'''
'''
raw_imgs = keras.utils.image_dataset_from_directory(
    "raw_imgs/",
    labels = None,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = (image_size, image_size),
    shuffle=True,
    interpolation='bilinear',
    seed = seed
)
cookiecut_raw_imgs = keras.utils.image_dataset_from_directory(
    "cookiecut_raw_imgs/",
    labels = None,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = (image_size, image_size),
    shuffle=True,
    interpolation='bilinear',
    seed = seed
)
# and combine the cookiecut images with the cropped/zoomed ones:
raw_imgs = raw_imgs.concatenate(cookiecut_raw_imgs)

raw_imgs = raw_imgs.shuffle(100, seed=seed)

for i in range(10):
    random_selection = raw_imgs.take(1)
    raw_images = list(random_selection.as_numpy_iterator())[0]
    raw_image = tf.convert_to_tensor(raw_images[0],dtype=tf.float32)
    raw_image = raw_image.numpy().astype(np.uint8)
    imageio.imwrite('checkpoint_imgs/'+str(i)+'.png', raw_image)''''''
