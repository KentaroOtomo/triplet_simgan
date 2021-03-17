''' 
	config file for testing the SimGAN
'''

import torch
import yaml

with open('config.yaml', 'r') as yml:
	config = yaml.safe_load(yml)


''' ---- Everything below is pretty much hyperparameters to change ----- '''


# Which GPU to use
cuda_num 		= 2

# Not terribly important for testing
batch_size 		= 1

# Directory to load the synthetic images from (path to the folder)
#synthetic_path 	= '/home/silvias15-local/data/eval_imgs_500/'
#synthetic_path  = '/home/silvias15/simgan-torch/data/UnityEyes3/'
# Directory to save the refined images to (path to the folder)
#save_refined_path 	= '../new_cyclegan_output/'
#save_refined_path = '/disks/dl310/silvias15/BCE_MSE_CCL_500_output/'
#save_refined_path   = '/home/silvias15/simgan-torch/output/'
# path to load the checkpoint too, the format of the checkpoints should match how it is saved during training. 
#If you want to use different format to load weights, you have to change load_weights in sub_simgan.py
checkpoint_path 	= config['checkpoint_path']['pathname']
#checkpoint_path = '/home/silvias15/simgan-torch/' + 'edge25,recon2.5/'

''' ----- No need to change anything below ----- '''

# prefix of saved discriminator weights
L_D1_path			= 'L_D1_%d.pkl'
G_D1_path			= 'G_D1_%d.pkl'
L_D2_path			= 'L_D2_%d.pkl'
G_D2_path			= 'G_D2_%d.pkl'
# prefix of saved refiner weights
R1_path			= 'R1_%d.pkl'
R2_path			= 'R2_%d.pkl'

Reg_path        = 'Reg_%d.pkl'
Gaze_path       = 'Gaze_%d.pkl'

# filename of optimizer information, this file 
# is located in the folder pointed to by checkpoint_path
optimizer_path	= 'optimizer_status.pkl'

# check if cuda is available
cuda_use 		= torch.cuda.is_available()

# train = false because we are testing !:)
train = False
