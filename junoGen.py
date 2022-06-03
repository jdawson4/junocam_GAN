# just a simple script for generating images using the trained GAN.

import tensorflow as tf
import numpy as np
import imageio

raw_imgs = tf.keras.preprocessing.image_dataset_from_directory(
    "raw_imgs/",
    labels = None,
    color_mode = 'rgb',
    batch_size = 1,
    image_size = (1024, 1024),
    shuffle=True,
    seed = 7
)

trained_gen = tf.keras.models.load_model('junoGAN')
i = 0
for b in raw_imgs.__iter__():
    i+=1
    #print("Generating image", i)
    fake_images = trained_gen(b)
    fake_images = tf.cast(fake_images, tf.float16)
    fake_images = fake_images + tf.cast(b,tf.float16) # more jerry-rigging
    fake_images *= 255.0
    fake_images = fake_images.numpy().astype(np.uint8)
    for fake_image in fake_images:
        imageio.imwrite('fake_images/'+str(i)+'.png', fake_image)