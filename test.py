# Author: Jacob Dawson
# This file just contains a few numpy/tensorflow functions
# I need to understand in order to get the rest of the system
# to work. You can safely ignore all of this.

import numpy as np
import tensorflow as tf

'''def hasNan(x, number):
    if tf.math.reduce_any((tf.math.is_nan(x))):
        print(number, "has a nan!")
    return x

a=tf.cast([1,2,np.nan,6.9,5.20,np.nan], tf.float16)
print(tf.math.multiply_no_nan(a, tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(a)), dtype=tf.float16)))
print(hasNan(a, 1))'''

a=tf.cast([-0.2,-20.0,256.0,125.0,300.0], tf.float16)
def makeImg(x):
    min = tf.reduce_min(x)
    if (min < 0):
        x += tf.math.abs(min)
    max = tf.reduce_max(x)
    x = (x/max) * 255.0
    return x
print(makeImg(a))
    
