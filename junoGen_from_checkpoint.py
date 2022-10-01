# Author: Jacob Dawson
# just a simple script for generating images using the trained GAN.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import imageio
from architecture import *
from constants import *

raw_imgs = tf.keras.preprocessing.image_dataset_from_directory(
    "raw_imgs/",
    labels = None,
    color_mode = 'rgb',
    batch_size = 1,
    image_size = (1600, 1600),
    shuffle=False, # same order as the directory so we can compare!
    seed = seed
)

generator = gen()
discriminator = dis()

def content_loss(fake, real):
    f=tf.cast(fake, tf.float32)
    r=tf.cast(real, tf.float32)
    ssim = chi * (1.0-tf.experimental.numpy.mean(tf.image.ssim(f,r,1.0)))
    l1 = ((1.0-chi) * tf.norm((f/(batch_size*255.0)) - (r/(batch_size*255.0))))
    return tf.cast(ssim,tf.float16)+tf.cast(l1,tf.float16)

# and here we create teh ConditionalGAN itself. Exciting!
class ConditionalGAN(keras.Model):
    def __init__(self,discriminator, generator):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.gen_loss_tracker = keras.metrics.Mean(name='generator_loss')
        self.dis_loss_tracker = keras.metrics.Mean(name='discriminator_loss')

    @property # no idea what this does
    def metrics(self):
        return [self.gen_loss_tracker, self.dis_loss_tracker]
    
    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, run_eagerly):
        super(ConditionalGAN, self).compile(run_eagerly=run_eagerly)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        #self.epoch_num = 0

    def train_step(self, data):
        raw_img_batch, user_img_batch = data

        # generate labels for the real and fake images
        batch_size = tf.shape(raw_img_batch)[0]
        true_image_labels = -tf.cast(tf.ones((batch_size,1)), tf.float16)
        fake_image_labels = tf.cast(tf.ones((batch_size,1)), tf.float16)
        # REMEMBER: TRUE IMAGES ARE -1, GENERATED IMAGES ARE +1

        # training here:
        with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
            gen_output = generator(raw_img_batch, training=True)
            disc_real_output = discriminator(user_img_batch, training=True)
            disc_generated_output = discriminator(gen_output, training=True)
            wganLoss = -self.g_loss_fn(fake_image_labels,disc_generated_output)
            wganLoss = tf.convert_to_tensor(wgan_lambda, dtype=tf.float16) * wganLoss
            contentLoss = content_loss(gen_output, raw_img_batch)
            contentLoss = tf.convert_to_tensor(content_lambda, dtype=tf.float16) * contentLoss
            total_g_loss = wganLoss + contentLoss
            d_loss = self.d_loss_fn(fake_image_labels,disc_generated_output) - self.d_loss_fn(true_image_labels,disc_real_output)
        grads = gtape.gradient(total_g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads,self.generator.trainable_weights)
        )
        self.gen_loss_tracker.update_state(total_g_loss)

        grads = dtape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        self.dis_loss_tracker.update_state(d_loss)
        
        return {
            'g_loss': self.gen_loss_tracker.result(),
            'd_loss': self.dis_loss_tracker.result(),
            'GAN_loss': wganLoss,
            'content_loss': contentLoss
        }

# okay... let's try to use this thing:
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator
)
def wasserstein_loss(y_true,y_pred):
    return tf.keras.backend.mean(y_true*y_pred)
cond_gan.compile(
    #d_optimizer = tf.keras.optimizers.RMSprop(learning_rate = dis_learn_rate),
    #g_optimizer = tf.keras.optimizers.RMSprop(learning_rate = gen_learn_rate),
    d_optimizer = tf.keras.optimizers.Adam(learning_rate = dis_learn_rate,beta_1=momentum),
    g_optimizer = tf.keras.optimizers.Adam(learning_rate = gen_learn_rate,beta_1=momentum),
    d_loss_fn = wasserstein_loss,
    g_loss_fn = wasserstein_loss,
    run_eagerly=True
)

cond_gan.built=True
cond_gan.load_weights("ckpts/ckpt66")
print("Checkpoint loaded.")

trained_gen = cond_gan.generator
print("Extracted generator.")

# if you decide you wanna save the generator as-is:
#trained_gen.save('junoGen',overwrite=True)

i = 0
for b in raw_imgs.__iter__():
    i+=1
    #print("Generating image", i)
    fake_images = trained_gen(b)
    fake_images = tf.cast(fake_images, tf.float16)
    fake_images = fake_images.numpy().astype(np.uint8)
    for fake_image in fake_images:
        # sorta weird to loop like this but keras outputs a list of length 1,
        # so just go with it
        imageio.imwrite('fake_images/'+str(i)+'.png', fake_image)
        