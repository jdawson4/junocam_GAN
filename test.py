import numpy as np
import tensorflow as tf

a=tf.cast([1,2,np.nan,6.9,5.20,np.nan], tf.float16)
print(tf.math.multiply_no_nan(a, tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(a)), dtype=tf.float16)))