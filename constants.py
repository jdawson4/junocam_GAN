# Author: Jacob Dawson
# this file contains a bunch of constants that control the construction
# and processing of our data. These include hyperparameters and certain
# numbers that control preprocessing.
#
# CONSTANTS
seed = 3 # my lucky number!
batch_size = 1 # unsure what my computer can handle haha
num_channels = 3 # rgb baby
image_size = 800
# each raw image is 1600x1600. I think each output should be too
epochs = 300
num_filters = 8
n_critic = 5 # the number of times we'll train the discriminator before going on to the generator

# custom hyperparameters
psi = 0.01
# ^ determines how much weight we give to "content loss" vs the
# "fooling the discriminator" loss in our generative loss function.
# 1 means that we only care about content loss; 0 means that we only
# care about fooling the discriminator
chi = 0.5 # how much we care about SSIM vs L2 when creating content loss
# ^ not sure if we're even going to use L2 after all.

# learning rates:
# some people suggest these should be the same--I find that the discriminator
# optimizes faster, so I'll give the generator a stronger learning rate.
gen_learn_rate = 0.00005
dis_learn_rate = 0.00005
momentum = 0.5