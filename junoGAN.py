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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# The data on my computer is nearly 600 MB...
# I'm not sure if this is a great idea:
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
user_imgs = keras.utils.image_dataset_from_directory(
    "user_imgs/",
    labels = None,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = (image_size, image_size), # force everything to be this size?
    shuffle=True,
    interpolation='bilinear',
    crop_to_aspect_ratio = True, # unsure about this one
    seed = seed
)

style_image = tf.keras.preprocessing.image.load_img(
    'style_image.png',
    color_mode = 'rgb',
    target_size = (image_size,image_size),
    interpolation='bilinear'
)
style_image = tf.keras.utils.img_to_array(style_image)
# we will use this example image to compute style loss.
# Yes, I have been reduced to this.

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
#discriminator.summary()
print('\n')
#generator.summary()
print('\n')
print("Image size:", image_size)
print("Approx. raw imgs dataset size:", batch_size * raw_imgs.cardinality().numpy())
print("Approx. user imgs dataset size:", batch_size * user_imgs.cardinality().numpy())
print("Batch size:", batch_size)
#print("Latent dim used by generator is 1/"+str(latent_dim_smaller_by_factor), "of input")
print("Weight of content loss:", content_lambda)
print("Weight of WGAN loss:", wgan_lambda)
print("Weight of style loss:", style_lambda)
print("Value of hyperparameter chi:", chi)
print("Value of WGAN hyperparameter n_critic:", n_critic)
print('g learning rate', gen_learn_rate)
print('d learning rate', dis_learn_rate)
print("Intended number of epochs:",epochs)
print("Number of GPUs we're running on:", num_gpus)
#print("Number of filters at each convolutional layer:", num_filters)
print('\n')
print('######################################################################')
print('\n')

def content_loss(fake, real):
    f=tf.cast(fake, tf.float32)
    r=tf.cast(real, tf.float32)
    ssim = chi * (1.0-tf.experimental.numpy.mean(tf.image.ssim(f,r,1.0)))
    l1 = ((1.0-chi) * tf.norm((f/(batch_size*255.0)) - (r/(batch_size*255.0))))
    return tf.cast(ssim,tf.float32)+tf.cast(l1,tf.float32)

def gram_matrix(x):
    #x = tf.expand_dims(x, 0)
    #print(x.shape)
    x = tf.cast(tf.transpose(x, (2, 0, 1)), tf.float32)
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    #print('gram:',gram)
    return tf.cast(gram, tf.float32)

def style_loss(fake, style):
    S = gram_matrix(style)
    sum = tf.cast(0.0, tf.float32)
    for f in fake:
        C = gram_matrix(f)
        size = tf.cast(image_size * image_size, tf.float32)
        sum += tf.math.reduce_sum(tf.math.square(S-C)) / (4.0 * (num_channels ** 2) * (size ** 2))
    #print('style_loss:',sum / batch_size)
    return sum / batch_size

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

        # generate labels for the real and fake images
        batch_size = tf.shape(raw_img_batch)[0]
        true_image_labels = -tf.cast(tf.ones((batch_size,1)), tf.float32)
        fake_image_labels = tf.cast(tf.ones((batch_size,1)), tf.float32)
        # REMEMBER: TRUE IMAGES ARE -1, GENERATED IMAGES ARE +1

        fake_images = self.generator(raw_img_batch, training=True)
        fake_images = tf.cast(fake_images,tf.float16)
        for itr in range(n_critic):
            with tf.GradientTape() as dtape:
                g_predictions = self.discriminator(fake_images, training=True)
                d_predictions = self.discriminator(user_img_batch, training=True)
                d_loss = self.d_loss_fn(fake_image_labels,g_predictions) - self.d_loss_fn(true_image_labels,d_predictions)
            grads = dtape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )
            self.dis_loss_tracker.update_state(d_loss)
        #littleDiscrim = keras.Model(inputs = self.discriminator.input,
        #    outputs = self.discriminator.get_layer('conv2d_39').output
        #)

        with tf.GradientTape() as gtape:
            fake_images = self.generator(raw_img_batch, training=True)
            fake_images = tf.cast(fake_images,tf.float16)
            g_predictions = self.discriminator(fake_images, training=True)
            wganLoss = -self.g_loss_fn(fake_image_labels,g_predictions)
            contentLoss = content_loss(fake_images, raw_img_batch)
            contentLoss = tf.cast(contentLoss, tf.float32)
            wganLoss = tf.cast(wganLoss, tf.float32)
            #style_analysis = littleDiscrim(tf.expand_dims(style_image,0))[0]
            #fake_analysis = littleDiscrim(fake_images)
            #styleLoss = style_loss(fake_analysis,style_analysis)
            styleLoss = 0.0
            wganLoss = tf.convert_to_tensor(wgan_lambda, dtype=tf.float32) * wganLoss
            contentLoss = tf.convert_to_tensor(content_lambda, dtype=tf.float32) * contentLoss
            styleLoss = tf.convert_to_tensor(style_lambda, dtype=tf.float32) * styleLoss
            total_g_loss = (wganLoss + contentLoss + styleLoss)
        #print(wganLoss)
        #print(styleLoss)
        #print(contentLoss)
        grads = gtape.gradient(total_g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads,self.generator.trainable_weights)
        )
        self.gen_loss_tracker.update_state(total_g_loss)

        #print("")
        #print(tf.reduce_max(fake_images))
        #print(tf.reduce_max((raw_img_batch)))
        #print(tf.reduce_max((user_img_batch)))
        #print("")

        return {
            'g_loss': self.gen_loss_tracker.result(),
            'd_loss': self.dis_loss_tracker.result(),
            'GAN_loss': wganLoss,
            'content_loss': contentLoss,
            'style_loss': styleLoss
        }

# okay... let's try to use this thing:
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator
)
def wasserstein_loss(y_true,y_pred):
    #print(y_true, y_pred)
    #print(tf.keras.backend.mean(y_true*y_pred))
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

# only uncomment this code if you have a prepared checkpoint to use for output:
#cond_gan.built=True
#cond_gan.load_weights("ckpts/ckpt10")
#print("Checkpoint loaded, skipping training.")

'''# this is my janky, beautiful, disgusting, manual solution to the fit() problem
# instead of using keras' in-built fit() function, I'm doing each batch
# manually. However, this does mean that I can be quite specific with how my
# checkpoints and callbacks work and shit like that, so that's nice.
# Unfortunately, it prints out much uglier :(
for i in range(1,epochs+1):
    print("Epoch", str(i), end=' ')
    gl = 0.0
    dl = 0.0
    wloss = 0.0
    contloss = 0.0
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
        wloss += metrics['GAN_loss']
        contloss += metrics['content_loss']
        #if (j%64==0):
        #    b_g_loss = metrics['g_loss']
        #    b_d_loss = metrics['d_loss']
        #    #print(f'Metrics for batch {j}:', metrics)
        #    print(f'Batch {j} g{b_g_loss:.4f} d{b_d_loss:.4f},', end=' ')
    gl = tf.cast(gl, tf.float32) # sometimes this breaks? Unsure why.
    dl = tf.cast(dl, tf.float32)
    wloss = tf.cast(wloss, tf.float32)
    contloss = tf.cast(contloss, tf.float32)
    gl /= num_batches
    dl /= num_batches
    wloss /= num_batches
    contloss /= num_batches
    print(f"g-loss: {gl:.10f}, d-loss: {dl:.10f}, GAN loss: {wloss:.10f}, content loss: {contloss:.10f}")
    if ((i%5)==0):
        # save a checkpoint every 5 epochs for a history of training
        cond_gan.save_weights("ckpts/ckpt"+str(i), overwrite=True, save_format='h5')
        if (i%10)==0:
            cond_gan.generator.save('junoGen',overwrite=True)
            # every few checkpoints, save model.
        # every few epochs, save a an image
        raw_image = (tf.expand_dims(x_batch[0],0))
        fake_image = cond_gan.generator(raw_image, training=False)[0]
        #fake_image = tf.cast(fake_image, tf.float16) # do we need to do this??? Worried about floating point math.
        fake_image = fake_image.numpy().astype(np.uint8)
        raw_image = raw_image[0].numpy().astype(np.uint8)
        imageio.imwrite('checkpoint_imgs/'+str(i)+'.png', fake_image)
        imageio.imwrite('checkpoint_imgs/raw'+str(i)+'.png', raw_image)'''

class EveryKCallback(keras.callbacks.Callback):
    def __init__(self,data,epoch_interval=5):
        self.data = data
        self.epoch_interval = epoch_interval
    def on_epoch_end(self,epoch,logs=None):
        if ((epoch % self.epoch_interval)==0):
            random_selection = self.data.take(1)
            raw_images, _ = list(random_selection.as_numpy_iterator())[0]
            raw_image = tf.convert_to_tensor(raw_images[0],dtype=tf.float32)
            fake_image = self.model.generator(tf.expand_dims(raw_image,0),training=False)[0]
            raw_image = raw_image.numpy().astype(np.uint8)
            fake_image = fake_image.numpy().astype(np.uint8)
            imageio.imwrite('checkpoint_imgs/'+str(epoch)+'.png', fake_image)
            imageio.imwrite('checkpoint_imgs/'+str(epoch)+'raw.png', raw_image)

            self.model.save_weights("ckpts/ckpt"+str(epoch), overwrite=True, save_format='h5')
            if (epoch%(self.epoch_interval*2))==0:
                self.model.generator.save('junoGen',overwrite=True)
                # every few checkpoints, save model.

both_datasets = tf.data.Dataset.zip((raw_imgs,user_imgs))
cond_gan.fit(
    both_datasets,
    # data is already batched!
    epochs = epochs,
    verbose=1,
    callbacks=[EveryKCallback(both_datasets, epoch_interval=5)], # custom callbacks here!
    # validation doesnt really apply here?
    shuffle=False, # already shuffled by dataset api
)

cond_gan.save_weights("ckpts/finished", overwrite=True, save_format='h5')
cond_gan.generator.save('junoGen',overwrite=True)
# for good measure, save again once we're done training
