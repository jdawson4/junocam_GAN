# Author: Jacob Dawson
# Date: begun on May 31, 2022
#
# Purpose: the idea here is to implement a GAN and train it to automatically
# process images of space. There's lots of unprocessed space data out there,
# and I've always wanted to make a data-driven solution to automating the
# processing of it. One can access raw images taken by spacecraft, for instance
# from the JunoCam project, found here:
# https://www.missionjuno.swri.edu/junocam/processing/
#
# The idea for this script is therefore to implement something like a
# Conditional Generative Adverserial Network. Con-GANs, like traditional
# GANs, pit two deep learners against one another in something like an
# "evolutionary arms race", but unlike other GANs, they rely on some
# ground truth for the generative model to rely on rather than random
# noise. For more information: https://arxiv.org/abs/1411.1784
#
# The idea behind a GAN is that the first learner, the generator,
# creates images that are supposedly similar to natural process,
# and then the second model, the discriminator, tries to tell these
# artificial images apart from the real thing. In this case, then,
# we want the generator to be looking at raw (ugly) images of space
# and transforming them into something akin to what human tastes
# enjoy when looking at images of celestial bodies. Therefore, the
# discriminative model will attempt to distinguish between the images
# that the generator creates and images that actual humans have labored
# painstakingly to create by hand through color-correcting raw data.
# With any luck, the generator will be able to learn what steps a human being
# would take to transform a raw image of Jupiter into an aesthetically pleasing
# one!
#
# One other note: I am considering creating a custom loss function for the
# generative model such that the generator is encouraged to fool the
# discriminator (as per usual) AND to ensure that the images it creates
# aren't overly far from the raw data files. For this, I am taking inspiration
# from the famous "Style Transfer" paper: https://arxiv.org/abs/1508.06576
# which defines something that they call "content loss". Not entirely sure
# how I'll be implementing that, as I believe that they use a pretrained model
# in order to compare the latent space of certain layers against one another,
# but I have this in mind!
#
# Enough writing, on with the coding!

# IMPORTS
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from constants import *
from architecture import *
import imageio
#import tensorflow_datasets as tfds

physical_devices = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(physical_devices)
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# The data on my computer is nearly 600 MB...
# I'm not sure if this is a great idea:
raw_imgs = keras.preprocessing.image_dataset_from_directory(
    "raw_imgs/",
    labels = None,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = (image_size, image_size),
    shuffle=True,
    seed = seed
)
user_imgs = keras.preprocessing.image_dataset_from_directory(
    "user_imgs/",
    labels = None,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = (image_size, image_size), # force everything to be this size?
    shuffle=True,
    crop_to_aspect_ratio = True, # unsure about this one
    seed = seed
)

# maybe we should do image augmentation? Come back to this!
# TODO: CONSIDER IMAGE AUGMENTATION

# these are declared in architecture.py
generator = gen()
discriminator = dis()


# let's print some useful info here
print('\n')
print('######################################################################')
print('\n')
print('Architecture:\n')
discriminator.summary()
print('\n')
generator.summary()
print('\n')
print("Image size:", image_size)
print("Approx. raw imgs dataset size:", batch_size * raw_imgs.cardinality().numpy())
print("Approx. user imgs dataset size:", batch_size * user_imgs.cardinality().numpy())
print("Batch size:", batch_size)
#print("Latent dim used by generator is 1/"+str(latent_dim_smaller_by_factor), "of input")
print("Value of hyperparameter psi:", psi)
#print("Value of hyperparameter chi:", chi)
print('g learning rate', gen_learn_rate)
print('d learning rate', dis_learn_rate)
print("Intended number of epochs:",epochs)
print("Number of GPUs we're running on:", num_gpus)
print("Number of filters at each convolutional layer:", num_filters)
print('\n')
print('######################################################################')
print('\n')

#mse = tf.keras.losses.MeanSquaredError()
def content_loss(fake, real):
    #print(fake)
    #print(real)
    ssim = chi * (1-tf.experimental.numpy.mean(tf.image.ssim(fake,real,1.0)))
    l2 = (1-chi) * tf.math.reduce_mean(tf.math.squared_difference(fake, real))
    #print('ssim,',ssim)
    #print('l2,',l2)
    return tf.cast(ssim,tf.float32)+tf.cast(l2,tf.float32)
    # apparently this returns semantic dist?
    # note: SSIM measures from 0 to 1. 0 means poor quality, 1 means good
    # quality. We want loss to be 1-ssim, so that we encourage good quality,#
    # right? Yes. That's what others are doing, so I'll copy them.
    # This SHOULD fix an issue where it seems like the generator was coming
    # up with images that where spherical (and able to trick the discriminator)
    # but still fucked up (colors looked WEIRD.)
    # Others still are using SSIM + L2, which... I sorta like! I'll consider it.
    #return (1.0-(tf.experimental.numpy.mean(tf.image.ssim(fake,real,1.0))))
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
        #self.epoch_num+=1
        # alright, here's the thing I've been dreading.
        # the thing is, we need to pass in two types of data via data:
        # 1. raw images to feed to the generator
        # 2. user-made images to feed to the discriminator
        # I've just repackaged the data to be just that.
        # Remember: when we call fit, x should be set to the raw
        # images, and y should be set to the user-made ones.
        raw_img_batch, user_img_batch = data
        #raw_img_batch = (raw_img_batch) / 255.0
        #user_img_batch = (user_img_batch) / 255.0
        #raw_img_batch = raw_img_batch / 255.0
        #user_img_batch = user_img_batch / 255.0
        #print('raws',tf.math.reduce_max(raw_img_batch))
        #print('raws',tf.math.reduce_min(raw_img_batch))
        #print('raws',tf.math.reduce_mean(raw_img_batch))
        #print('users',tf.math.reduce_max(user_img_batch))
        #print('users',tf.math.reduce_min(user_img_batch))
        #print('users',tf.math.reduce_mean(user_img_batch))

        # generate labels for the real and fake images
        batch_size = tf.shape(raw_img_batch)[0]
        true_image_labels = -tf.cast(tf.ones((batch_size,1)), tf.float32)
        fake_image_labels = tf.cast(tf.ones((batch_size,1)), tf.float32)
        # REMEMBER: TRUE IMAGES ARE -1, GENERATED IMAGES ARE +1

        with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
            fake_images = self.generator(raw_img_batch)
            fake_images = tf.cast(fake_images,tf.float16)
            #print('fakes',tf.math.reduce_max(fake_images))
            #print('fakes',tf.math.reduce_min(fake_images))
            #print('fakes',tf.math.reduce_mean(fake_images))
            #fake_images = fake_images + raw_img_batch # act like a resnet
            #print(fake_images[0], 'fakes')
            #print(raw_img_batch[0], 'raws')
            g_predictions = self.discriminator(fake_images)
            d_predictions = self.discriminator(user_img_batch)
            d_loss = self.d_loss_fn(fake_image_labels,g_predictions) - self.d_loss_fn(true_image_labels,d_predictions)
            #d_loss *= -1
            g_loss1 = -self.g_loss_fn(fake_image_labels,g_predictions)
            #print('compared:')
            #print(true_image_labels)
            #print(d_predictions)
            #g_loss2 = content_loss(fake_images, raw_img_batch)
            #g_loss2 = tf.cast(g_loss2, tf.float32)
            #g_loss1 = tf.cast(g_loss1, tf.float32)
            #g_loss1 = tf.convert_to_tensor(1.0-psi, dtype=tf.float32) * g_loss1
            #g_loss2 = tf.convert_to_tensor(psi, dtype=tf.float32) * g_loss2
            #print(g_loss1)
            #print(g_loss2)
            #total_g_loss = (-tf.math.abs(g_loss1) + tf.math.abs(g_loss2)) # wtf is going on
            #print(total_g_loss)
            '''g_loss = self.g_loss_fn(fake_image_labels, g_predictions)
            g_loss *= -1'''
        grads = dtape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        grads = gtape.gradient(g_loss1, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads,self.generator.trainable_weights)
        )

        self.gen_loss_tracker.update_state(g_loss1)
        self.dis_loss_tracker.update_state(d_loss)

        return {
            'g_loss': self.gen_loss_tracker.result(),
            'd_loss': self.dis_loss_tracker.result(),
        }

# okay... let's try to use this thing:
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator
)
def wasserstein_loss(y_true,y_pred):
    return tf.keras.backend.mean(y_true*y_pred)
cond_gan.compile(
    d_optimizer = tf.keras.optimizers.RMSprop(learning_rate = dis_learn_rate),
    g_optimizer = tf.keras.optimizers.RMSprop(learning_rate = gen_learn_rate),
    d_loss_fn = wasserstein_loss,
    g_loss_fn = wasserstein_loss,
    run_eagerly=True
)

# only uncomment this code if you have a prepared checkpoint to use for output:
#cond_gan.built=True
#cond_gan.load_weights("ckpts/ckpt10")
#print("Checkpoint loaded, skipping training.")

# this is my janky, beautiful, disgusting, manual solution to the fit() problem
# instead of using keras' in-built fit() function, I'm doing each batch
# manually. However, this does mean that I can be quite specific with how my
# checkpoints and callbacks work and shit like that, so that's nice.
# Unfortunately, it prints out much uglier :(
for i in range(1,epochs+1):
    print("Epoch", str(i), end=' ')
    gl = 0.0
    dl = 0.0
    a = raw_imgs.__iter__()
    b = user_imgs.__iter__()
    num_batches = tf.get_static_value(raw_imgs.cardinality())
    for j in range(num_batches):
        x_batch = tf.cast(a.get_next(), tf.float16)
        y_batch = tf.cast(b.get_next(), tf.float16)
        metrics = cond_gan.train_on_batch(
            x=x_batch,
            y=y_batch,
            return_dict=True
        )
        gl += metrics['g_loss']
        dl += metrics['d_loss']
        #print(dl)
        #print(gl)
        #if (j%64==0):
        #    b_g_loss = metrics['g_loss']
        #    b_d_loss = metrics['d_loss']
        #    #print(f'Metrics for batch {j}:', metrics)
        #    print(f'Batch {j} g{b_g_loss:.4f} d{b_d_loss:.4f},', end=' ')
    gl = tf.cast(gl, tf.float32) # sometimes this breaks? Unsure why.
    dl = tf.cast(dl, tf.float32)
    gl /= num_batches
    dl /= num_batches
    print(f"g-loss: {gl:.10f}, d-loss: {dl:.10f}")
    if ((i%5)==0):
        # save a checkpoint every 5 epochs for a history of training
        cond_gan.save_weights("ckpts/ckpt"+str(i), overwrite=True, save_format='h5')
        if (i%10)==0:
            cond_gan.generator.save('junoGen',overwrite=True)
            # every few checkpoints, save model.
        # every few epochs, save a an image
        raw_image = (tf.expand_dims(x_batch[0],0))
        fake_image = cond_gan.generator(raw_image)[0]
        fake_image = tf.cast(fake_image, tf.float16)
        #print('fake', tf.math.reduce_mean(fake_image))
        #print('fake', tf.math.reduce_max(fake_image))
        #print('fake', tf.math.reduce_min(fake_image))
        #print('raw', tf.math.reduce_mean(raw_image))
        #print('raw', tf.math.reduce_max(raw_image))
        #print('raw', tf.math.reduce_min(raw_image))
        #fake_image = (fake_image) * 255.0
        #raw_image = (raw_image[0]) * 255.0
        #fake_images = fake_image + tf.cast(x_batch[0],tf.float16)+tf.cast(tf.ones(fake_images.shape), tf.float16)
        fake_image = fake_image.numpy().astype(np.uint8)
        #print(fake_image.shape)
        #print(raw_image.shape)
        raw_image = raw_image[0].numpy().astype(np.uint8)
        imageio.imwrite('checkpoint_imgs/'+str(i)+'.png', fake_image)
        imageio.imwrite('checkpoint_imgs/raw'+str(i)+'.png', raw_image)
        #print(fake_image, "fake image")
        #print(raw_image, "raw image")
cond_gan.save_weights("ckpts/finished", overwrite=True, save_format='h5')
cond_gan.generator.save('junoGen',overwrite=True)
# for good measure, save again once we're done training