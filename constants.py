# Author: Jacob Dawson
# this file contains a bunch of constants that control the construction
# and processing of our data. These include hyperparameters and certain
# numbers that control preprocessing.
#
# CONSTANTS
seed = 3 # my lucky number!
batch_size = 16 # unsure what my computer can handle haha
num_channels = 3 # rgb baby
image_size = 256
# each raw image is 1600x1600. I think each output should be too
epochs = 300
num_filters = 8

# custom hyperparameters
psi = 0.05
# ^ determines how much weight we give to "content loss" vs the
# "fooling the discriminator" loss in our generative loss function.
# 1 means that we only care about content loss; 0 means that we only
# care about fooling the discriminator
chi = 0.85 # how much we care about SSIM vs L2 when creating content loss

# learning rates:
# some people suggest these should be the same--I find that the discriminator
# optimizes faster, so I'll give the generator a stronger learning rate.
gen_learn_rate = 0.003
dis_learn_rate = 0.0003