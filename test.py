# Author: Jacob Dawson
# This file just contains a few numpy/tensorflow functions
# I need to understand in order to get the rest of the system
# to work. You can safely ignore all of this.

import numpy as np
import tensorflow as tf

def hasNan(x, number):
    if tf.math.reduce_any((tf.math.is_nan(x))):
        print(number, "has a nan!")
    return x

a=tf.cast([1,2,np.nan,6.9,5.20,np.nan], tf.float16)
print(tf.math.multiply_no_nan(a, tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(a)), dtype=tf.float16)))
print(hasNan(a, 1))
