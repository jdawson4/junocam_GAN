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
#import imageio
#import tensorflow_datasets as tfds

physical_devices = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(physical_devices)
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.autograph.set_verbosity(
#    level=0, alsologtostdout=False
#)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CONSTANTS
seed = 3 # my lucky number!
batch_size = 1 # unsure what my computer can handle haha
num_channels = 3 # rgb baby
image_size = 1024
# each raw image is 1600x1600. I think each output should be too
#latent_dim_smaller_by_factor = 2
# not 100% sure about this, but maybe the generator should be
# smaller in the middle?
psi = 0.3
# determines how much weight we give to "content loss" vs the
# "fooling the discriminator" loss in our generative loss function.
# 1 means that we only care about content loss; 0 means that we only
# care about fooling the discriminator
chi = 0.85 # how much we care about SSIM vs L2 when creating content loss
epochs = 50
num_filters = 8

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

#print("Raw imgs dataset size:", batch_size * raw_imgs.cardinality().numpy())
#print("User imgs dataset size:", batch_size * user_imgs.cardinality().numpy())

# ARCHITECTURE
discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((image_size, image_size, num_channels),dtype=tf.float16),
        keras.layers.Conv2D(num_filters, (3,3), strides = (2,2), padding = 'same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides=(2,2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides=(2,2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides=(2,2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides=(2,2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides=(2,2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides=(2,2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides=(2,2), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dense(1),
    ],
    name='discriminator',
)
# the architecture here summarizes thoughts into smaller and smaller
# latent spaces and then spits out a boolean.

# the generator is a little more complicated. I'm not sure, but I think that
# having some sort of bottleneck akin to encoding/decoding would be helpful.
#
# We control the size of this bottleneck via a constant declared above,
# latent_dim_smaller_by_factor, an extremely catchy name that I'm sure will
# catch on everywhere.
generator = keras.Sequential(
    [
        keras.layers.InputLayer((image_size,image_size,num_channels),dtype=tf.float16),
        keras.layers.Conv2D(num_filters, (3,3), strides = (1,1), padding='same'),
        keras.layers.LeakyReLU(alpha=0.2),
        #keras.layers.MaxPooling2D(pool_size = latent_dim_smaller_by_factor), # reduce to a smaller latent space. Maybe we shouldn't, who knows!
        #keras.layers.Conv2D(num_filters, (3,3), strides = (1,1), padding='same'), # do some convolutions in the smaller latent space!
        #keras.layers.LeakyReLU(alpha=0.2), # nonlinearity
        #keras.layers.Conv2DTranspose(num_channels, (2,2), strides = latent_dim_smaller_by_factor), # upscale back up to the original size!
        #keras.layers.LeakyReLU(alpha = 0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides = (1,1), padding = 'same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides = (1,1), padding = 'same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides = (1,1), padding = 'same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides = (1,1), padding = 'same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides = (1,1), padding = 'same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides = (1,1), padding = 'same'),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(num_filters, (3,3), strides = (1,1), padding = 'same'), # do some final convoluting
        keras.layers.LeakyReLU(alpha=0.2), # a final nonlinearity
        # outputs... how do outputs work again?
        keras.layers.Conv2D(num_channels, (2, 2), padding="same", activation="sigmoid"),
    ],
    name='generator'
)
# note that the generator might want to be significantly smaller
# (half the size?) of the discriminator. I think that this is a good idea
# because its outputs should theoretically need less "work" to arrive at
# (after all, it's just learning proper color correction)

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
print("Value of hyperparameter chi:", chi)
print("Intended number of epochs:",epochs)
print("Number of GPUs we're running on:", num_gpus)
print("Number of filters at each convolutional layer:", num_filters)
print('\n')
print('######################################################################')
print('\n')

# actually, I've thought about it, and we should really do this:
'''raw_imgs = raw_imgs.as_numpy_iterator()
user_imgs = user_imgs.as_numpy_iterator()
print(type(raw_imgs))
for x in raw_imgs:
    print(type(x))
    print(x.shape)
    break
# both datasets are now iterable tensorflow objects
# in which each iteration returns an ndarray of shape (4,1600,1600,3).
# That is, 4 images of size 1600x1600 in RGB. Great!'''
# just kidding! This is wrong and bad!
'''raw_imgs = raw_imgs.__iter__()
user_imgs = user_imgs.__iter__()
temp_raw_imgs = list()
temp_user_imgs = list()
for x in raw_imgs:
    temp_raw_imgs.append(x)
    temp_user_imgs.append(user_imgs.get_next())
raw_imgs = np.array(temp_raw_imgs, dtype = np.float32)
user_imgs = np.array(temp_user_imgs, dtype = np.float32)
print(raw_imgs.shape)
print(user_imgs.shape)'''
# this is also broken! Crap!
'''data = tf.data.Dataset.from_tensors((raw_imgs.__iter__(), user_imgs.__iter__()))
print(type(data))
for datum in data.__iter__():
    print(type(datum))
    print(len(datum))
    firstitem,seconditem = datum
    print(type(firstitem))
    print(type(seconditem))'''
# also bad!

# unfortunately we need to do this really quick.
# Not 100% sure how to go about this, might not even use it at all!
# Funny how these things work.
#def content_loss(fake, real):
#    print(type(fake))
#    print(fake.shape)
#    print(type(real))
#    print(real.shape)
#    return tf.keras.losses.MeanAbsolutePercentageError([fake], [real]) # no this doesn't return in the correct format
#content_loss = tf.keras.losses.MeanAbsolutePercentageError() # always returns infinity???
def content_loss(fake, real):
    ssim = chi *(1.0-(tf.experimental.numpy.mean(tf.image.ssim(fake,real,1.0))))
    mse = (1-chi) * (tf.keras.metrics.mean_squared_error(fake, real))
    return tf.cast(ssim,tf.float32)+tf.cast(mse,tf.float32)
    # apparently this returns semantic dist?
    # note: SSIM measures from 0 to 1. 0 means poor quality, 1 means good
    # quality. We want loss to be 1-ssim, so that we encourage good quality,#
    # right? Yes. That's what others are doing, so I'll copy them.
    # This SHOULD fix an issue where it seems like the generator was coming
    # up with images that where spherical (and able to trick the discriminator)
    # but still fucked up (colors looked WEIRD.)
    # Others still are using SSIM + L2, which... I sorta like! I'll consider it.

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
    
    def compile(self, d_optimizer, g_optimizer, loss_fn, run_eagerly):
        super(ConditionalGAN, self).compile(run_eagerly=run_eagerly)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
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
        #print(type(data))
        #print(len(data))
        #print(type(data[0]))
        raw_img_batch, user_img_batch = data
        #print(type(raw_img_batch))
        #print(type(user_img_batch))
        #print(raw_img_batch.dtype)
        #print(user_img_batch.dtype)

        # generate labels for the real and fake images
        batch_size = tf.shape(raw_img_batch)[0]
        true_image_labels = tf.zeros((batch_size,1))
        fake_image_labels = tf.ones((batch_size,1))
        # REMEMBER: TRUE IMAGES ARE 0, GENERATED IMAGES ARE 1
        
        # make generations:
        #generated_images = self.generator(raw_img_batch)
        #print(type(generated_images))
        #print(generated_images.dtype)
        #generated_images = tf.cast(generated_images, tf.float16)
        #generated_images = generated_images + raw_img_batch # this jerry-rigs the generator to work like a resnet

        # do we need to do this?
        #fake_images_and_labels = tf.concat([generated_images, fake_image_labels],-1)
        #real_images_and_labels = tf.concat([user_img_batch, true_image_labels],-1)
        
        # whatever. Make a thing for the discriminator to work with.
        #all_images = tf.concat([user_img_batch, generated_images],0)
        #all_labels = tf.concat([true_image_labels,fake_image_labels],0)
        #all_images_and_labels = tf.concat([all_images,all_labels],-1)

        '''# make discriminations
        with tf.GradientTape() as tape:
            predictions = self.discriminator(all_images)
            d_loss = self.loss_fn(all_labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # unfortunately, because of how gradienttape works, I now need to do
        # the generation process all over again. This isn't ideal, because the
        # discriminator was JUST trained on the generator's outputs. Hmm.
        with tf.GradientTape() as tape:
            fake_images = self.generator(raw_img_batch)
            fake_images = tf.cast(fake_images, tf.float16)
            fake_images = fake_images + raw_img_batch # more jerry-rigging
            #fake_image_and_labels = tf.concat([fake_images,fake_image_labels],-1)
            predictions = self.discriminator(fake_images) # might not need??
            g_loss1 = self.loss_fn(fake_image_labels, predictions)
            classic_g_loss = g_loss1
            g_loss2 = content_loss(fake_images, raw_img_batch)
            #print(g_loss2.dtype)
            #print(g_loss1)
            #print(g_loss2)
            g_loss2 = tf.cast(g_loss2, tf.float32)
            g_loss1 = tf.convert_to_tensor(1.0-psi, dtype=tf.float32) * g_loss1
            g_loss2 = tf.convert_to_tensor(psi, dtype=tf.float32) * g_loss2
            #total_g_loss = ((1-psi) * g_loss1) + (psi * g_loss2) # tf is unhappy?
            total_g_loss = tf.math.add(tf.math.abs(g_loss1), tf.math.abs(g_loss2)) # hideous. Let me use a plus sign.
            #print("Total_g_loss", total_g_loss)
        grads = tape.gradient(total_g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads,self.generator.trainable_weights)
        )'''

        # update: replace the above clusterfuck with a more optimized version.
        # This is possible because GradientTapes are actually really cool.
        with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
            fake_images = self.generator(raw_img_batch)
            fake_images = tf.cast(fake_images,tf.float16)
            fake_images = fake_images + raw_img_batch # act like a resnet
            all_images = tf.concat([user_img_batch, fake_images],0)
            all_labels = tf.concat([true_image_labels,fake_image_labels],0)
            all_predictions = self.discriminator(all_images)
            g_predictions = all_predictions[:batch_size]
            d_loss = self.loss_fn(all_labels, all_predictions)
            g_loss1 = self.loss_fn(fake_image_labels, g_predictions)
            g_loss2 = content_loss(fake_images, raw_img_batch)
            g_loss2 = tf.cast(g_loss2, tf.float32)
            g_loss1 = tf.convert_to_tensor(1.0-psi, dtype=tf.float32) * g_loss1
            g_loss2 = tf.convert_to_tensor(psi, dtype=tf.float32) * g_loss2
            total_g_loss = tf.math.add(tf.math.abs(g_loss1), tf.math.abs(g_loss2))
        grads = dtape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        grads = gtape.gradient(total_g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads,self.generator.trainable_weights)
        )

        self.gen_loss_tracker.update_state(total_g_loss)
        self.dis_loss_tracker.update_state(d_loss)

        return {
            'g_loss': self.gen_loss_tracker.result(),
            'd_loss': self.dis_loss_tracker.result(),
        }

# okay... let's try to use this thing:
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator
)
cond_gan.compile(
    d_optimizer = keras.optimizers.Adam(learning_rate = 0.0003),
    g_optimizer = keras.optimizers.Adam(learning_rate = 0.0003),
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True),
    run_eagerly=True
)

# only uncomment this code if you have a prepared checkpoint to use for output:
#cond_gan.built=True
#cond_gan.load_weights("ckpts/ckpt40")
#print("Checkpoint loaded, skipping training.")
#cond_gan.generator.save('junoGen',overwrite=True)

#a = raw_imgs.__iter__()
#b = user_imgs.__iter__()
#def stupid_data_thing():
#    return (tf.cast(a.get_next(), tf.float16), tf.cast(b.get_next(), tf.float16))

# this code doesn't work. I just realized that zip() would probably work to
# combine the two datasets in a useful way, but I came up with another
# (better?) solution first.
#cond_gan.fit(
#    tensorflow is being finicky. It won't let me insert both datasets into fit right here. Unfortunate.
#    x=stupid_data_thing(),
#    epochs=epochs,
#    batch_size = batch_size
#)

# this is my janky, beautiful, disgusting, manual solution to the fit() problem
# instead of using keras' in-built fit() function, I'm doing each batch
# manually. However, this does mean that I can be quite specific with how my
# checkpoints and callbacks work and shit like that, so that's nice.
# Unfortunately, it prints out much uglier :(
for i in range(1,epochs+1):
    print("Epoch", str(i))
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
        #if (j%32==0):
        #    print(f'Metrics for batch {j}:', metrics)
    gl = tf.cast(gl, tf.float32) # sometimes this breaks? Unsure why.
    dl = tf.cast(dl, tf.float32)
    gl /= num_batches
    dl /= num_batches
    print(f"g-loss: {gl:.10f}, d-loss: {dl:.10f}")
    if ((i%5)==0):
        # save a checkpoint every 5 epochs for a history of training
        cond_gan.save_weights("ckpts/ckpt"+str(i), overwrite=True, save_format='h5')
        if (i%20)==0:
            cond_gan.generator.save('junoGen',overwrite=True)
            # every 20 epochs, save g
cond_gan.generator.save('junoGen',overwrite=True)
# for good measure, save again once we're done training

'''
i = 0
trained_generator = cond_gan.generator
for b in raw_imgs.__iter__():
    i+=1
    #print("Generating image", i)
    fake_images = trained_generator(b)
    fake_images = tf.cast(fake_images, tf.float16)
    fake_images = fake_images + tf.cast(b,tf.float16) # more jerry-rigging
    fake_images *= 255.0
    fake_images = fake_images.numpy().astype(np.uint8)
    for fake_image in fake_images:
        imageio.imwrite('fake_images/'+str(i)+'.png', fake_image)
'''