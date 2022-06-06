# Author: Jacob Dawson
# The purpose of this file is so that I can make sense of the
# wasserstein loss function

import tensorflow as tf
def wasserstein_loss(y_true,y_pred):
    return tf.keras.backend.mean(y_true*y_pred)

a = tf.cast([1,1,1,1],tf.float32)
b = tf.cast([-1,-1,-1,-1],tf.float32)
c = tf.cast([0,1,-1,-1],tf.float32)
d = tf.cast([0,1,-0.5,-1],tf.float32)
e = tf.cast([0.1,0.9,-0.9,-1],tf.float32)
print(wasserstein_loss(a,a))
print(wasserstein_loss(a,b))
print(wasserstein_loss(b,a))
print(wasserstein_loss(b,b))

print()

print(wasserstein_loss(a,c))
print(wasserstein_loss(c,a))
print(wasserstein_loss(c,d))
print(wasserstein_loss(d,c))

#print(tf.clip_by_value(e,1))