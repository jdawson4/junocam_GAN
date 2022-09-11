# Author: Jacob Dawson
# this file contains a bunch of constants that control the construction
# and processing of our data. These include hyperparameters and certain
# numbers that control preprocessing.
#
# CONSTANTS
seed = 3 # my lucky number!
batch_size = 4 # unsure what my computer can handle haha
num_channels = 3 # rgb!
image_size = 400
# each raw image is 1600x1600. I think each output should be too
epochs = 100
#num_filters = 8 # this no longer filters into architecture--it's all based around num_channels now.
n_critic = 1 # the number of times we'll train the discriminator before going on to the generator
# ^ the original paper chose 5

# custom hyperparameters--determine things about loss:
chi = 0.9 # how much we care about SSIM vs L2 when creating content loss
# ^ not sure if we're even going to use L2 after all.
content_lambda = 0.001 # content loss weight
wgan_lambda = 0.5 # the weight we give to fooling the wgan
style_lambda = 0.0 # the weight we give to style loss

# learning rates:
gen_learn_rate = 0.0001
dis_learn_rate = 0.0004 # idea here comes from https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628, called TTUR
momentum = 0.9 # default for adam is 0.9, default for RMSProp is 0.0
